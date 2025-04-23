import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from typing import Generator

from .distance import compute_pw_dist_sqr
from .utils import dict_map, append_dict, get_default_device, add_dict


def compute_average(p: Tensor, vals: Tensor) -> Tensor:
    return torch.matmul(p.unsqueeze(-2), vals.unsqueeze(-1)).view(*p.shape[:-1])


def compute_stats_batch(dataloader: DataLoader[tuple[Tensor, ...]], x0_traj: Tensor, temp: Tensor) -> dict[str, Tensor]:
    """Compute statistics for a batch of noisy samples against clean data.

    Args:
        dataloader: DataLoader containing clean images batches of shape (batch_size, *obj_size)
        x0_traj: Batch of trajectory starts with shape (batch_size, d), where d = prod(obj_size)
        temp: Temperature values with shape (n_temps)

    Returns:
        Dictionary containing:
            - "entropy": Entropy values of shape (n_temps, batch_size)
            - "heat_capacity": Heat capacity values of shape (n_temps, batch_size)

    Notes:
        Auxiliary tensors' shapes:
            - xt: (n_temps, batch_size, d)
            - energy: (n_temps, batch_size, num_objects)
            - exp: (n_temps, batch_size, num_objects)
            - log_part_fun: (n_temps, batch_size)
            - p: (n_temps, batch_size, num_objects)
            - avg_energy: (n_temps, batch_size)
            - var_energy: (n_temps, batch_size)
    """

    device = get_default_device()
    temp = temp.to(device)
    xt = torch.randn(len(temp), *x0_traj.shape, device=device) * temp.view(-1, *[1] * len(x0_traj.shape)).sqrt() + x0_traj.to(device)    
    num_objects = len(dataloader.dataset)  # type: ignore

    energy = torch.empty(*xt.shape[:2], num_objects, device=device)
    offset = 0
    for x0, *_ in dataloader:
        b = x0.shape[0]
        dist = 0.5 * compute_pw_dist_sqr(xt.view(-1, *xt.shape[2:]), x0.to(device, non_blocking=True))
        energy[..., offset:offset + b] = dist.view(*xt.shape[:2], b)
        offset += b
    
    del xt

    energy -= energy.min(-1, keepdim=True).values
    exp = -energy / temp.view(-1, 1, 1)
    log_part_fun = exp.exp().sum(-1).log()
    exp -= log_part_fun.unsqueeze(-1)
    p = exp.exp()

    del exp

    avg_energy = compute_average(p, energy)
    energy -= avg_energy.unsqueeze(-1)
    var_energy = compute_average(p, energy.square())

    del energy

    entropy = log_part_fun + avg_energy / temp.view(-1, 1) - np.log(num_objects)
    heat_capacity = var_energy / temp.square().view(-1, 1)

    return {"entropy": entropy.cpu(), "heat_capacity": heat_capacity.cpu()}


def compute_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
        unbiased: bool,
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


def compute_all_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
) -> dict[str, Tensor]:
    stats = compute_stats(dataloader, data_generator, temp, n_samples, False)
    # unbiased_stats = compute_stats(dataloader, data_generator, temp, n_samples, True)
    # stats = {**stats, **{key + "_unbiased": value for key, value in unbiased_stats.items() if key != "temp"}}
    # for suffix in ["", "_unbiased"]:
    #     stats["entropy" + suffix + "_extrapolated"] = extrapolate_entropy(temp, stats["entropy" + suffix])
    stats["entropy_extrapolated"] = extrapolate_entropy(temp, stats["entropy"])
    return stats


def extrapolate_entropy(temp: Tensor, entropy: Tensor) -> Tensor:
    log_temp = temp.log()
    slope = (entropy[1:] - entropy[:-1]) / (log_temp[1:] - log_temp[:-1])
    idx = torch.argmax(slope)
    idx -= int(idx == len(temp))
    return torch.cat(((log_temp[:idx] - log_temp[idx]) * slope[idx] + entropy[idx], entropy[idx:]), dim=0)
