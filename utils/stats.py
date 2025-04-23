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


def compute_stats_batch(dataloader: DataLoader[tuple[Tensor, ...]], xt: Tensor, temp: Tensor) -> dict[str, Tensor]:
    # energy = torch.cat([0.5 * compute_pw_dist_sqr(xt, x0.flatten(1).to(xt.device), final_device=xt.device) for x0, *_ in dataloader], dim=1)
    num_objects = len(dataloader.dataset)  # type: ignore
    energy = torch.empty(*xt.shape[:-1], num_objects, device=xt.device)

    # Fill in by batch
    offset = 0
    for x0, *_ in dataloader:
        b = x0.shape[0]
        dist = 0.5 * compute_pw_dist_sqr(xt, x0.flatten(1).to(xt.device, non_blocking=True), final_device=xt.device)
        energy[:, offset:offset + b] = dist
        offset += b

    energy -= energy.min(-1, keepdim=True).values
    exp = -energy / temp.unsqueeze(-1)
    log_part_fun = exp.exp().sum(-1).log()
    p = (exp - log_part_fun.unsqueeze(-1)).exp()

    avg_energy = compute_average(p, energy)
    var_energy = compute_average(p, (energy - avg_energy.unsqueeze(-1)).square())

    entropy = log_part_fun + avg_energy / temp - np.log(num_objects)
    heat_capacity = var_energy / temp.square()

    return {"entropy": entropy.cpu(), "heat_capacity": heat_capacity.cpu()}


def compute_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
        unbiased: bool,
) -> dict[str, Tensor]:
    device = get_default_device()
    temp = temp.to(device)
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    samples_left = n_samples
    while samples_left > 0:
        batch_stats: dict[str, list[Tensor]] = defaultdict(list)
        xt = next(data_generator)[0].flatten(1).to(device)
        for idx in trange(len(temp), desc=f"Computing {'unbiased ' if unbiased else ''}statistics..."):
            xt += torch.randn_like(xt) * (temp[idx] - (temp[idx - 1] if idx else 0)).sqrt()
            append_dict(batch_stats, compute_stats_batch(dataloader, xt, temp[idx]))
        add_dict(stats, dict_map(torch.stack, batch_stats))
        samples_left -= len(xt)

    stats = dict_map(lambda val: torch.stack(val).mean(0), batch_stats)
    stats["temp"] = temp.cpu()
    return stats


def compute_all_stats(
        dataloader: DataLoader[tuple[Tensor, ...]],
        data_generator: Generator[tuple[Tensor, ...], None, None],
        temp: Tensor,
        n_samples: int,
) -> dict[str, Tensor]:
    stats = compute_stats(dataloader, data_generator, temp, n_samples, False)
    unbiased_stats = compute_stats(dataloader, data_generator, temp, n_samples, True)
    stats = {**stats, **{key + "_unbiased": value for key, value in unbiased_stats.items() if key != "temp"}}
    for suffix in ["", "_unbiased"]:
        stats["entropy" + suffix + "_extrapolated"] = extrapolate_entropy(temp, stats["entropy" + suffix])
    return stats


def extrapolate_entropy(temp: Tensor, entropy: Tensor) -> Tensor:
    log_temp = temp.log()
    slope = (entropy[1:] - entropy[:-1]) / (log_temp[1:] - log_temp[:-1])
    idx = torch.argmax(slope)
    idx -= int(idx == len(temp))
    return torch.cat(((log_temp[:idx] - log_temp[idx]) * slope[idx] + entropy[idx], entropy[idx:]), dim=0)
