from torch import Tensor
import torch
from utils import compute_pw_dist_sqr
import numpy as np
from tqdm import trange


def compute_free_energy(z: Tensor, y: Tensor, T: Tensor) -> Tensor:
    H = 0.5 * compute_pw_dist_sqr(z, y)

    exp = -H / T
    max_exp = exp.max(1)[0]
    exp -= max_exp[:, None]
    log_Z = exp.exp().mean(1).log() + max_exp

    return -log_Z * T


def sample_free_energy_direct_computation(
    y: Tensor,
    temp: Tensor,
    batch_size: int = 100,
    n_iter: int = 1000,
    average: bool = False,
) -> Tensor:
    y = y.cuda()

    if average:
        average_free_energy = torch.zeros_like(temp)
    else:
        free_energy_lst: list[Tensor] = []

    for _ in trange(n_iter):
        batch_free_energy_lst: list[Tensor] = []
        z = y[np.random.choice(range(len(y)), size=(batch_size,))]
        for i, T in enumerate(temp):
            dT = T - (temp[i - 1] if i else 0)
            z += np.sqrt(dT) * torch.randn_like(z)
            free_energy = compute_free_energy(z, y, T)
            if average:
                free_energy = free_energy.mean()
            batch_free_energy_lst.append(free_energy)

        if average:
            average_free_energy += torch.stack(batch_free_energy_lst, dim=0).cpu() / n_iter
        else:
            free_energy_lst.append(torch.stack(batch_free_energy_lst, dim=1).cpu())

    return average_free_energy if average else torch.cat(free_energy_lst, dim=0)
