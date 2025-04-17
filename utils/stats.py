from torch import Tensor
from torch.nn.functional import pad
import torch
import numpy as np
from math import ceil
from tqdm import trange
from collections import defaultdict

from .distance import compute_pw_dist_sqr
from .utils import dict_map, append_dict, get_default_device


def compute_average(p: Tensor, vals: Tensor) -> Tensor:
    return torch.matmul(p.unsqueeze(-2), vals.unsqueeze(-1)).view(*p.shape[:-1])


def get_noise(temp: Tensor, batch_size: int, obj_shape: tuple[int, ...]) -> Tensor:
    temp_diffs = (temp - pad(temp[:-1], (1, 0))).view(-1, *[1] * (len(obj_shape)))
    noise_steps = torch.randn(batch_size, len(temp), *obj_shape, device=temp.device)
    return (noise_steps * temp_diffs.sqrt()).cumsum(1)


def compute_stats_batch(x0: Tensor, xt: Tensor, temp: Tensor) -> dict[str, Tensor]:
    energy = 0.5 * compute_pw_dist_sqr(xt, x0, final_device=x0.device)
    energy -= energy.min(1, keepdim=True).values
    exp = -energy / temp.unsqueeze(-1)
    log_part_fun = exp.exp().sum(1).log()
    p = (exp - log_part_fun.unsqueeze(-1)).exp()

    avg_energy = compute_average(p, energy)
    var_energy = compute_average(p, (energy - avg_energy.unsqueeze(-1)).square())

    entropy = log_part_fun + avg_energy / temp - np.log(len(x0))
    heat_capacity = var_energy / temp.square()

    return {"entropy": entropy.cpu(), "heat_capacity": heat_capacity.cpu()}


def compute_stats_traj_batch(x0: Tensor, traj_start: Tensor, temp: Tensor) -> dict[str, Tensor]:
    noise = get_noise(temp, len(traj_start), x0.shape[1:])
    xt = traj_start + noise
    xt_flat = xt.flatten(0, 1)
    temp_flat = temp.unsqueeze(0).expand(len(xt), -1).flatten()

    batch_stats = compute_stats_batch(x0, xt_flat, temp_flat)
    traj_batch_stats = dict_map(lambda val: val.reshape(xt.shape[:2]).mean(0), batch_stats)
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
    temp = temp.to(device)
    batch_stats: dict[str, list[Tensor]] = defaultdict(list)
    for _ in trange(ceil(n_samples / batch_size), desc=f"Computing {'unbiased ' if unbiased else ''}statistics..."):
        if unbiased:
            idx = np.random.choice(range(len(x0)))
            traj_start = x0[torch.full((batch_size, 1), idx).long()]
            _x0 = torch.cat([x0[:idx], x0[idx + 1:]])
        else:
            traj_start = x0[np.random.choice(range(len(x0)), size=(batch_size, 1))]
        append_dict(batch_stats, compute_stats_traj_batch(_x0, traj_start, temp))

    stats = dict_map(lambda val: torch.stack(val).mean(0), batch_stats)
    stats["temp"] = temp.cpu()
    return stats


def compute_all_stats(x0: Tensor, temp: Tensor, n_samples: int, batch_size: int) -> dict[str, Tensor]:
    stats = compute_stats(x0, temp, n_samples, batch_size, False)
    unbiased_stats = compute_stats(x0, temp, n_samples, batch_size, True)
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
