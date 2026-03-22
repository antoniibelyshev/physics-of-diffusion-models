import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from typing import Generator

from .distance import compute_pw_dist_sqr
from .utils import dict_map, append_dict, get_default_device, add_dict


def compute_metric_stats_batch(dataloader: DataLoader[tuple[Tensor, ...]], x0_traj: Tensor, temp: Tensor) -> dict[str, Tensor]:
    device = get_default_device()
    temp = temp.to(device)
    n_temps = len(temp)
    batch_size = x0_traj.shape[0]
    num_objects = len(dataloader.dataset) # type: ignore
    D = x0_traj.view(batch_size, -1).shape[1]
    
    all_metric = []
    
    # Pre-calculate x0 on device to avoid repeated transfers
    all_x0 = []
    for x0, *_ in dataloader:
        all_x0.append(x0.to(device, non_blocking=True))
    all_x0_cat = torch.cat(all_x0, dim=0)

    for i in range(n_temps):
        t = temp[i]
        xt_i = torch.randn(*x0_traj.shape, device=device) * t.sqrt() + x0_traj.to(device)
        
        # dist = 0.5 * ||xt - x0||^2
        dist = 0.5 * compute_pw_dist_sqr(xt_i, all_x0_cat)
        
        # p(x|y, lambda) = exp(-energy / t) / sum(exp(-energy / t))
        # Robust log_weights calculation: log(p_k) = -E_k/t - logsumexp(-E_j/t)
        energy_over_t = dist / t
        log_weights = -energy_over_t - torch.logsumexp(-energy_over_t, dim=-1, keepdim=True)
        weights = log_weights.exp()
        
        # G(lambda) = E_y [ Var_{x|y, lambda} [ d ln p(y|x, lambda) / d lambda ] ]
        # d ln p(y|x, lambda) / d lambda = -0.5 * D + dist / t
        # Var_{x|y, lambda} [ -0.5 * D + dist / t ] = Var_{x|y, lambda} [ dist / t ]
        # Var[Z] = E[Z^2] - (E[Z])^2
        
        expected_dist_over_t = (weights * energy_over_t).sum(-1)
        expected_dist_sq_over_t = (weights * energy_over_t.pow(2)).sum(-1)
        
        var_score_at_y = expected_dist_sq_over_t - expected_dist_over_t.pow(2)
        
        # Average over y (x0_traj)
        all_metric.append(var_score_at_y.mean().cpu())
        
    return {"metric_values": torch.stack(all_metric)}


def compute_metric_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
) -> dict[str, Tensor]:
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    
    with tqdm(total=n_samples, desc="Computing metric stats...") as pbar:
        while n_samples > 0:
            x0_traj = next(data_generator)[0]
            curr_batch_size = len(x0_traj)
            append_dict(batch_stats, compute_metric_stats_batch(dataloader, x0_traj, temp))
            n_samples -= curr_batch_size
            pbar.update(curr_batch_size)

    # metric_values shape: (n_temps, n_batches)
    # We average metric values across batches
    metric = torch.cat([v.unsqueeze(1) for v in batch_stats["metric_values"]], dim=1).mean(dim=1)
    
    return {
        "temp": temp,
        "metric": metric,
        "log_temp": temp.log()
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
