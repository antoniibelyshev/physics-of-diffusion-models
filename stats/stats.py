from torch import Tensor
import torch
from utils import compute_pw_dist_sqr, dict_map
import numpy as np
from tqdm import trange


def compute_stats_batch(x: Tensor, y: Tensor, temp: Tensor) -> dict[str, Tensor]:
    H = 0.5 * compute_pw_dist_sqr(x, y)
    min_H = H.min(1)[0]
    H -= min_H[:, None]

    exp = -H / temp
    max_exp = exp.max(1)[0]
    norm_exp = exp - max_exp[:, None]

    log_Z = (norm_exp.exp().mean(1).log() + max_exp)
    p = (exp - log_Z[:, None]).exp()

    U = (p * H).mean(1)
    var_H = (p * (H - U[:, None]) ** 2).mean(1)

    return {"log_Z": log_Z, "U": U, "full_U": U + min_H, "var_H": var_H}


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
        dT = (temp - torch.cat([torch.zeros_like(temp[:1]), temp[:-1]])).cuda().view(-1, *[1] * (len(y.shape) - 1))
        noise = (torch.randn(batch_size, len(temp), *y.shape[1:], device=y.device) * dT.sqrt()).cumsum(1)
        x = y[np.random.choice(range(len(y)), size=(batch_size, 1))] + noise
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
        dT = (temp - torch.cat([torch.zeros_like(temp[:1]), temp[:-1]])).cuda().view(-1, *[1] * (len(y.shape) - 1))
        noise = (torch.randn(batch_size, len(temp), *y.shape[1:], device=y.device) * dT.sqrt()).cumsum(1)
        idx = np.random.choice(range(len(y)))
        x = y[idx][None, None].repeat(batch_size, *([1] * y.dim())) + noise
        stats_lst.append(compute_stats_traj_batch(x, torch.cat([y[:idx], y[idx + 1:]], 0), temp))

    return {key: torch.stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}
