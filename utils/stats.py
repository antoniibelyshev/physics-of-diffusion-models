import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from typing import Generator, Optional

from .distance import compute_pw_dist_sqr
from .utils import dict_map, append_dict, get_default_device, add_dict
# from diffusion import DDPM  # Removed to avoid circular import


def compute_metric_stats_batch(
    dataloader: DataLoader[tuple[Tensor, ...]], 
    x0_traj: Tensor, 
    temp: Tensor,
    regularize: bool = False,
    adaptive_knn: bool = False,
    knn_k: int = 5,
    sigma_reg_scale: float = 1.0,
    precomputed_sigma_reg_sq: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    device = get_default_device()
    temp = temp.to(device)
    n_temps = len(temp)
    batch_size = x0_traj.shape[0]
    
    all_metric = []
    
    # Pre-calculate x0 on device to avoid repeated transfers
    all_x0 = []
    for x0, *_ in dataloader:
        all_x0.append(x0.to(device, non_blocking=True))
    all_x0_cat = torch.cat(all_x0, dim=0)
    
    # Estimate Tr(Sigma0) from dataset to check tail
    x0_flat = all_x0_cat.view(len(all_x0_cat), -1)
    dataset_tr_sigma0 = torch.var(x0_flat, dim=0).sum().item()
    D = x0_flat.shape[1]

    # Optional: compute adaptive per-point sigma_reg_sq using k-NN distances
    sigma_reg_sq_per_point: Optional[Tensor] = None
    if regularize and adaptive_knn:
        if precomputed_sigma_reg_sq is not None:
            sigma_reg_sq_per_point = precomputed_sigma_reg_sq.to(device)
        else:
            try:
                # Use sklearn NearestNeighbors for efficient batched k-NN
                from sklearn.neighbors import NearestNeighbors  # type: ignore
                x_np = x0_flat.cpu().numpy()
                # Note: n_neighbors = knn_k + 1 to include the point itself at distance 0
                nn = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
                nn.fit(x_np)
                dists, _ = nn.kneighbors(x_np, return_distance=True)
                # k-th neighbor distance (exclude the 0-th which is the point itself)
                d_k = dists[:, -1]  # shape (N,)
                sigma_reg_sq_per_point = torch.from_numpy((d_k ** 2).astype(np.float32)).to(device)
                # Apply scaling and dimension normalization
                sigma_reg_sq_per_point = sigma_reg_sq_per_point * sigma_reg_scale / float(D)
            except Exception as e:
                print(f"Warning: Adaptive k-NN regularization failed ({e}). Falling back to global sigma_reg_sq.")
                sigma_reg_sq_per_point = None
    
    # Check data range
    if x0_flat.min() < -2 or x0_flat.max() > 2:
        print(f"Warning: Data range [{x0_flat.min().item():.2f}, {x0_flat.max().item():.2f}] is unexpected (expected [-1, 1]).")
    
    print(f"Dataset D={D}, Tr(Sigma0)={dataset_tr_sigma0:.4f}")

    for i in range(n_temps):
        t = temp[i]
        # Generate xt = x0 + sqrt(t) * epsilon
        xt_i = torch.randn(*x0_traj.shape, device=device) * t.sqrt() + x0_traj.to(device)
        
        # dist = 0.5 * ||xt - x0||^2
        dist = 0.5 * compute_pw_dist_sqr(xt_i, all_x0_cat)
        
        # Subtract min dist for numerical stability in logsumexp
        min_dist = dist.min(dim=-1, keepdim=True).values
        energy_over_t = (dist - min_dist) / t
        
        log_weights = -energy_over_t - torch.logsumexp(-energy_over_t, dim=-1, keepdim=True)
        weights = log_weights.exp()
        
        # Var[Z] = E[Z^2] - (E[Z])^2 where Z = dist/T
        expected_energy_over_t = (weights * energy_over_t).sum(-1)
        expected_energy_sq_over_t = (weights * energy_over_t.pow(2)).sum(-1)
        
        var_score_at_y = torch.clamp(expected_energy_sq_over_t - expected_energy_over_t.pow(2), min=0)
        
        # Regularization for high dimensions:
        # In high dimensions, the discrete prior samples are very sparse.
        # The posterior collapses to a single mode too early (Var -> 0).
        # We assume each data point is actually a small Gaussian cluster with variance sigma_reg^2.
        # From derivation.tex, the rescaled metric for a Gaussian cluster is G_reg = 0.5 * sigma_reg^2 / (sigma_reg^2 + T)^2
        if regularize:
            if adaptive_knn and sigma_reg_sq_per_point is not None:
                # Compute effective local sigma as weights-average of per-point variances
                # weights: (batch, N), sigma_reg_sq_per_point: (N,)
                sigma_eff = torch.matmul(weights, sigma_reg_sq_per_point)  # (batch,)
                g_reg = 0.5 * sigma_eff * (sigma_eff + 2 * t) / (sigma_eff + t).pow(2)
                var_score_at_y = torch.maximum(var_score_at_y, g_reg)
            else:
                # Global small variance fallback
                sigma_reg_sq = torch.tensor(1e-3, device=device)
                g_reg = 0.5 * sigma_reg_sq * (sigma_reg_sq + 2 * t) / (sigma_reg_sq + t).pow(2)
                var_score_at_y = torch.maximum(var_score_at_y, g_reg)

        # Average over y (x0_traj)
        all_metric.append(var_score_at_y.mean().to(torch.float32).cpu())
        
    return {"metric_values": torch.stack(all_metric)}


def compute_metric_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
        regularize: bool = False,
        adaptive_knn: bool = False,
        knn_k: int = 5,
        sigma_reg_scale: float = 1.0,
) -> dict[str, Tensor]:
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    
    # If adaptive_knn, precompute per-point sigma_reg_sq once for all batches
    pre_sigma: Optional[Tensor] = None
    if regularize and adaptive_knn:
        device = get_default_device()
        all_x0 = []
        for x0, *_ in dataloader:
            all_x0.append(x0.to(device, non_blocking=True))
        all_x0_cat = torch.cat(all_x0, dim=0)
        x0_flat_all = all_x0_cat.view(len(all_x0_cat), -1)
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            x_np = x0_flat_all.cpu().numpy()
            nn = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
            nn.fit(x_np)
            dists, _ = nn.kneighbors(x_np, return_distance=True)
            d_k = dists[:, -1]
            pre_sigma = torch.from_numpy((d_k ** 2).astype(np.float32)).to(device)
            D = x0_flat_all.shape[1]
            pre_sigma = pre_sigma * sigma_reg_scale / float(D)
        except Exception as e:
            print(f"Warning: Precompute adaptive k-NN failed ({e}). Will compute on-the-fly or fallback.")
            pre_sigma = None

    with tqdm(total=n_samples, desc="Computing metric stats...") as pbar:
        while n_samples > 0:
            x0_traj = next(data_generator)[0]
            curr_batch_size = len(x0_traj)
            append_dict(batch_stats, compute_metric_stats_batch(
                dataloader, x0_traj, temp, 
                regularize=regularize, 
                adaptive_knn=adaptive_knn, 
                knn_k=knn_k, 
                sigma_reg_scale=sigma_reg_scale,
                precomputed_sigma_reg_sq=pre_sigma
            ))
            n_samples -= curr_batch_size
            pbar.update(curr_batch_size)

    # Average over y (x0_traj) and convert back to float32
    metric = torch.cat([v.unsqueeze(1) for v in batch_stats["metric_values"]], dim=1).mean(dim=1)
    
    # Calculate Tr(Sigma0) from the dataset for theoretical reference
    device = get_default_device()
    all_x0 = []
    for x0, *_ in dataloader:
        all_x0.append(x0.to(device, non_blocking=True))
    all_x0_cat = torch.cat(all_x0, dim=0)
    x0_flat = all_x0_cat.view(len(all_x0_cat), -1)
    dataset_tr_sigma0 = torch.var(x0_flat, dim=0).sum().item()
    
    return {
        "temp": temp,
        "metric": metric,
        "log_temp": temp.log(),
        "dataset_tr_sigma0": torch.tensor(dataset_tr_sigma0)
    }


@torch.no_grad()
def compute_model_metric_stats_batch(
    ddpm: torch.nn.Module, 
    x0_traj: Tensor, 
    temp: Tensor,
) -> dict[str, Tensor]:
    device = get_default_device()
    temp = temp.to(device)
    n_temps = len(temp)
    x0_traj = x0_traj.to(device)
    
    all_metric = []
    
    for i in range(n_temps):
        t = temp[i]
        # Generate xt = x0 + sqrt(t) * epsilon
        eps = torch.randn_like(x0_traj)
        xt_i = eps * t.sqrt() + x0_traj
        
        # Model predictions
        log_t = t.log().view(1)
        predictions = ddpm.get_predictions(xt_i, log_t)
        x0_pred = predictions.x0
        
        # G(lambda) = E [ 0.5 * ||x0 - x0_pred||^2 / T ]
        # This relationship holds because Var[dist/T] for a Gaussian-like posterior
        # is related to the expected error of the mean.
        # For small T, ||x0 - x0_pred||^2 / T is the dominant term.
        mse = torch.mean(torch.sum((x0_traj - x0_pred).view(len(x0_traj), -1)**2, dim=1))
        
        metric_val = 0.5 * mse / t
        all_metric.append(metric_val.cpu())
        
    return {"metric_values": torch.stack(all_metric)}


def compute_model_metric_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        ddpm: torch.nn.Module,
        temp: Tensor,
        n_samples: int,
) -> dict[str, Tensor]:
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    ddpm.eval()
    
    with tqdm(total=n_samples, desc="Computing model-based metric stats...") as pbar:
        while n_samples > 0:
            x0_traj = next(data_generator)[0]
            curr_batch_size = len(x0_traj)
            append_dict(batch_stats, compute_model_metric_stats_batch(ddpm, x0_traj, temp))
            n_samples -= curr_batch_size
            pbar.update(curr_batch_size)

    # Average over samples
    metric = torch.cat([v.unsqueeze(1) for v in batch_stats["metric_values"]], dim=1).mean(dim=1)
    
    # Calculate Tr(Sigma0) from the dataset for theoretical reference
    device = get_default_device()
    all_x0 = []
    for x0, *_ in dataloader:
        all_x0.append(x0.to(device, non_blocking=True))
    all_x0_cat = torch.cat(all_x0, dim=0)
    x0_flat = all_x0_cat.view(len(all_x0_cat), -1)
    dataset_tr_sigma0 = torch.var(x0_flat, dim=0).sum().item()
    
    return {
        "temp": temp,
        "metric": metric,
        "log_temp": temp.log(),
        "dataset_tr_sigma0": torch.tensor(dataset_tr_sigma0)
    }


def compute_average(p: Tensor, vals: Tensor) -> Tensor:
    return torch.matmul(p.unsqueeze(-2), vals.unsqueeze(-1)).view(*p.shape[:-1])


def compute_stats_batch(dataloader: DataLoader[tuple[Tensor, ...]], x0_traj: Tensor, temp: Tensor) -> dict[str, Tensor]:
    device = get_default_device()
    temp = temp.to(device)
    n_temps = len(temp)
    batch_size = x0_traj.shape[0]
    num_objects = len(dataloader.dataset) # type: ignore
    
    # Compute entropy one temperature at a time to save memory
    all_entropy = []
    
    for i in range(n_temps):
        t = temp[i]
        xt_i = torch.randn(*x0_traj.shape, device=device) * t.sqrt() + x0_traj.to(device)
        energy_i = torch.zeros(batch_size, num_objects, device=device)
        offset = 0
        for x0, *_ in dataloader:
            b = x0.shape[0]
            dist = 0.5 * compute_pw_dist_sqr(xt_i, x0.to(device, non_blocking=True))
            energy_i[:, offset:offset + b] = dist
            offset += b
        
        energy_i -= energy_i.min(-1, keepdim=True).values
        exp_i = -energy_i / t
        log_part_fun_i = exp_i.exp().sum(-1).log()
        exp_i -= log_part_fun_i.unsqueeze(-1)
        p_i = exp_i.exp()
        
        avg_energy_i = (p_i * energy_i).sum(-1)
        entropy_i = log_part_fun_i + avg_energy_i / t - np.log(num_objects)
        all_entropy.append(entropy_i.cpu())
        
    return {"entropy": torch.stack(all_entropy)}


def compute_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
) -> dict[str, Tensor]:
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    with tqdm(total=n_samples, desc="Computing stats...") as pbar:
        while n_samples > 0:
            x0_traj = next(data_generator)[0]
            append_dict(batch_stats, compute_stats_batch(dataloader, x0_traj, temp))
            n_samples -= len(x0_traj)
            pbar.update(len(x0_traj))

    stats = dict_map(lambda val: torch.cat(val, dim=1).mean(1), batch_stats)
    stats["temp"] = temp
    return stats


def extrapolate_entropy(temp: Tensor, entropy: Tensor, min_temp: float) -> tuple[Tensor, Tensor]:
    if temp[0] != min_temp:
        temp = torch.cat([torch.full((1,), min_temp), temp])
        entropy = torch.cat([torch.full((1,), entropy[0].item()), entropy])
    log_temp = temp.log()
    slope = (entropy[1:] - entropy[:-1]) / (log_temp[1:] - log_temp[:-1])
    idx = torch.argmax(slope)
    idx -= int(idx == len(temp))
    return temp, torch.cat(((log_temp[:idx] - log_temp[idx]) * slope[idx] + entropy[idx], entropy[idx:]), dim=0)
