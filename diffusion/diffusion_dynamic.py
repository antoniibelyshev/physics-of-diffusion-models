import torch
from torch import nn, Tensor, from_numpy
from torch.nn.functional import pad
import numpy as np
from typing import Callable

from config import Config
from utils import norm_sqr, get_inv_cdf


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_entropy_temp_schedule(stats_path: str) -> Callable[[Tensor], Tensor]:
    stats = np.load(stats_path)
    temp = stats["temp"]
    log_temp = np.log(temp)
    heat_capacity = stats["var_H"] / temp ** 2
    inv_cdf = get_inv_cdf(log_temp, heat_capacity)
    def entropy_temp_schedule(tau: Tensor) -> Tensor:
        return from_numpy(inv_cdf(tau.cpu().numpy())).exp().to(tau.device)
    return entropy_temp_schedule


def get_linear_beta_temp_schedule(min_temp: float, max_temp: float) -> Callable[[Tensor], Tensor]:
    scale = 1 + min_temp
    gamma = np.log((1 + max_temp) / scale)
    def linear_beta_temp_schedule(tau: Tensor) -> Tensor:
        return (tau.pow(2) * gamma).exp() * scale - 1 # type: ignore
    return linear_beta_temp_schedule


def get_cosine_temp_schedule(min_temp: float, max_temp: float) -> Callable[[Tensor], Tensor]:
    tau_min = 2 * np.arctan(min_temp ** 0.5) / np.pi
    tau_max = 2 * np.arctan(max_temp ** 0.5) / np.pi
    scale = 0.5 * np.pi * (tau_max - tau_min)
    shift = 0.5 * np.pi * tau_min
    def cosine_temp_schedule(tau: Tensor) -> Tensor:
        return (tau * scale + shift).tan().pow(2) # type: ignore
    return cosine_temp_schedule


def get_temp_schedule(config: Config) -> Callable[[Tensor], Tensor]:
    min_temp, max_temp = config.diffusion.temp_range
    if config.diffusion.noise_schedule == "linear_beta":
        return get_linear_beta_temp_schedule(min_temp, max_temp)
    elif config.diffusion.noise_schedule == "cosine":
        return get_cosine_temp_schedule(min_temp, max_temp)
    elif config.diffusion.noise_schedule.startswith("entropy"):
        return get_entropy_temp_schedule(config.flattening_temp_stats_path)
    else:
        raise ValueError(f"Unknown schedule type: {config.diffusion.noise_schedule}")


def get_alpha_bar(temp: Tensor) -> Tensor:
    return (temp + 1).pow(-1)


class DynamicCoeffs:
    def __init__(self, temp: Tensor) -> None:
        self.temp = temp
        self.alpha_bar = get_alpha_bar(temp)
        alpha_bar_prev = pad(self.alpha_bar[:-1], (*(0,) * (len(temp.shape) * 2 - 2), 1, 0), value=1.0)
        self.alpha = self.alpha_bar / alpha_bar_prev
        self.beta = 1 - self.alpha

        self.posterior_x0_coef = (alpha_bar_prev.sqrt() * self.beta) / (1 - self.alpha_bar)
        self.posterior_xt_coef = (self.alpha.sqrt() * (1 - alpha_bar_prev)) / (1 - self.alpha_bar)
        self.posterior_sigma = (1 - alpha_bar_prev) / (1 - self.alpha_bar) * self.beta

        self.dpm_xt_coef = self.alpha.pow(-0.5)
        self.dpm_eps_coef = -(1 - self.alpha_bar).sqrt() / self.alpha.sqrt() - (1 - alpha_bar_prev).sqrt()


class DDPMDynamic(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.obj_size = config.data.obj_size
        self.temp_schedule = get_temp_schedule(config)

    def get_temp(self, t: Tensor) -> Tensor:
        return self.temp_schedule(t).view(-1, *[1] * len(self.obj_size))

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        return get_alpha_bar(self.get_temp(t))

    def forward(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tau = torch.rand((len(x0),), device=x0.device)
        alpha_bar = self.get_alpha_bar(tau)
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + eps * (1 - alpha_bar).sqrt()
        return tau, eps, xt

    def get_true_score(self, xt: Tensor, t: Tensor, train_data: Tensor) -> Tensor:
        alpha_bar = self.get_alpha_bar(t)
        diffs = train_data.unsqueeze(1) * alpha_bar.sqrt() - xt
        n, bs, *obj_size = diffs.shape
        pows = -norm_sqr(diffs.view(n * bs, -1)).view(n, bs, *[1] * len(obj_size))
        pows /= 2 * (1 - alpha_bar)
        pows -= pows.max(0).values
        exps = pows.exp()
        diffs *= exps
        return diffs.sum(0) / (exps.sum(0) * (1 - alpha_bar)) # type: ignore
