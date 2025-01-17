from torch import Tensor
import torch
from utils import compute_pw_dist_sqr, dict_map
import numpy as np
from tqdm import trange


def compute_stats_batch(x: Tensor, y: Tensor, temp: Tensor) -> dict[str, Tensor]:
    energy = 0.5 * compute_pw_dist_sqr(x, y)
    min_energy = energy.min(1)[0]
    energy -= min_energy[:, None]

    exp = -energy / temp
    max_exp = exp.max(1)[0]
    norm_exp = exp - max_exp[:, None]

    log_part_fun = (norm_exp.exp().mean(1).log() + max_exp)
    p = (exp - log_part_fun[:, None]).exp()

    avg_energy = (p * energy).mean(1)
    var_energy = (p * (energy - avg_energy[:, None]) ** 2).mean(1)

    return {"log_Z": log_part_fun, "U": avg_energy, "full_U": avg_energy + min_energy, "var_H": var_energy}


def compute_stats_traj_batch(x: Tensor, y: Tensor, temp: Tensor) -> dict[str, Tensor]:
    x_flat = x.flatten(0, 1)
    temp_flat = temp.unsqueeze(0).expand(x.shape[0], -1).flatten().unsqueeze(1)
    batch_stats = compute_stats_batch(x_flat, y, temp_flat)
    return dict_map(lambda val: val.reshape(x.shape[:2]).mean(0), batch_stats)


def compute_stats(
    y: Tensor,
    temp: Tensor,
    batch_size: int = 100,
    n_iter: int = 1000,
) -> dict[str, Tensor]:
    y = y.cuda()

    stats_lst: list[dict[str, Tensor]] = []
    for _ in trange(n_iter):
        y_sample = y[np.random.choice(range(len(y)), size=(batch_size, 1))]
        noise = get_noise(temp, batch_size, y.shape[1:], y.device)
        x = y_sample + noise
        stats_lst.append(compute_stats_traj_batch(x, y, temp))

    return {key: torch.stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}


def compute_stats_unbiased(
    y: Tensor,
    temp: Tensor,
    batch_size: int = 100,
    n_iter: int = 1000,
) -> dict[str, Tensor]:
    y = y.cuda()

    stats_lst: list[dict[str, Tensor]] = []
    for _ in trange(n_iter):
        idx = np.random.choice(range(len(y)))
        y_sample = y[idx][None, None]
        noise = get_noise(temp, batch_size, y.shape[1:], y.device)
        x = y_sample + noise
        stats_lst.append(compute_stats_traj_batch(x, torch.cat([y[:idx], y[idx + 1:]], 0), temp))

    return {key: torch.stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}


def get_noise(temp: Tensor, batch_size: int, obj_shape: tuple[int, ...], device: torch.device) -> Tensor:
    d_temp = (temp - torch.cat([torch.zeros_like(temp[:1]), temp[:-1]])).cuda().view(-1, *[1] * (len(obj_shape) - 1))
    return (torch.randn(batch_size, len(temp), *obj_shape, device=device) * d_temp.sqrt()).cumsum(1)
