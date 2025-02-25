from torch import Tensor
from torch.nn.functional import pad
import torch
import numpy as np
from math import ceil
from tqdm import trange
from collections import defaultdict

from .distance import compute_pw_dist_sqr
from .utils import dict_map, append_dict, get_default_device


def get_noise(temp: Tensor, batch_size: int, obj_shape: tuple[int, ...], device: torch.device) -> Tensor:
    temp_diffs = (temp - pad(temp[:-1], (1, 0))).to(device).view(-1, *[1] * (len(obj_shape)))
    noise_steps = torch.randn(batch_size, len(temp), *obj_shape, device=device)
    return (noise_steps * temp_diffs.sqrt()).cumsum(1)


def compute_stats_batch(x0: Tensor, xt: Tensor, temp: Tensor) -> dict[str, Tensor]:
    energy = 0.5 * compute_pw_dist_sqr(x0, xt)
    min_energy = energy.min(0)[0]
    normalized_energy = energy - min_energy
    normalized_exponent = -normalized_energy / temp
    normalized_log_part_fun = normalized_exponent.exp().mean(0).log()
    p = (normalized_exponent - normalized_log_part_fun).exp()
    avg_normalized_energy = (p * normalized_energy).mean(0)
    var_energy = (p * (normalized_energy - avg_normalized_energy).square()).mean(0)

    entropy = normalized_log_part_fun + avg_normalized_energy / temp
    heat_capacity = var_energy / temp.square()

    return {"entropy": entropy, "heat_capacity": heat_capacity}


def compute_stats_traj_batch(x0: Tensor, traj_start: Tensor, temp: Tensor) -> dict[str, Tensor]:
    noise = get_noise(temp, len(traj_start), x0.shape[1:], x0.device)
    xt = traj_start + noise
    xt_flat = xt.flatten(0, 1)
    temp_flat = temp.unsqueeze(0).expand(len(xt), -1).flatten()

    batch_stats = compute_stats_batch(x0, xt_flat, temp_flat)
    traj_batch_stats = dict_map(lambda val: val.reshape(xt.shape[:2]).mean(0), batch_stats)
    traj_batch_stats["temp"] = temp
    return traj_batch_stats


def compute_stats(
        x0: Tensor,
        temp: Tensor,
        n_samples: int,
        batch_size: int,
        unbiased: bool,
) -> dict[str, Tensor]:
    device = get_default_device()
    x0 = x0.to(device)
    _x0 = x0
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    for _ in trange(ceil(n_samples / batch_size), desc="Computing statistics..."):
        if unbiased:
            idx = np.random.choice(range(len(x0)))
            traj_start = x0[torch.full((batch_size, 1), idx).long()]
            _x0 = torch.cat([x0[:idx], x0[idx + 1:]])
        else:
            traj_start = x0[np.random.choice(range(len(x0)), size=(batch_size, 1))]
        append_dict(batch_stats, compute_stats_traj_batch(_x0, traj_start, temp))

    stats = dict_map(lambda val: torch.stack(val).mean(0), batch_stats)
    return stats
