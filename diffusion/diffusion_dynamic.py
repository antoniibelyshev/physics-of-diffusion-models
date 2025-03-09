import torch
from torch import nn, Tensor
from torch.nn.functional import sigmoid
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod

from config import Config
from utils import norm_sqr, interp1d_tensor_fun, get_diffusers_pipeline


class NoiseScheduler(nn.Module, ABC):
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x) # type: ignore

    @abstractmethod
    def forward(self, tau: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_tau(self, log_temp: Tensor) -> Tensor:
        pass

    @staticmethod
    def from_config(config: Config, *, noise_schedule_type: Optional[str] = None) -> "NoiseScheduler":
        if noise_schedule_type is None or noise_schedule_type == "original":
            noise_schedule_type = config.diffusion.noise_schedule_type
        if noise_schedule_type == "linear_beta":
            return LinearBetaNoiseScheduler(config)
        elif noise_schedule_type == "cosine":
            return CosineNoiseScheduler(config)
        elif noise_schedule_type.startswith("entropy"):
            return EntropyNoiseScheduler(config, noise_schedule_type)
        elif noise_schedule_type == "from_diffusers":
            return FromDiffusersNoiseScheduler(config)
        else:
            raise ValueError(f"Unknown schedule type: {config.diffusion.noise_schedule_type}")


class LinearBetaNoiseScheduler(NoiseScheduler):
    def __init__(self, config: Config):
        super().__init__()

        min_temp, max_temp = config.diffusion.temp_range
        self.scale = 1 + min_temp
        self.gamma = np.log((1 + max_temp) / self.scale)

    def forward(self, tau: Tensor) -> Tensor:
        return ((tau.pow(2) * self.gamma).exp() * self.scale - 1).log()  # type: ignore

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return (((log_temp.exp() + 1) / self.scale).log() / self.gamma).sqrt()  # type: ignore


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, config: Config):
        super().__init__()

        min_temp, max_temp = config.diffusion.temp_range
        tau_min = 2 * np.arctan(min_temp ** 0.5) / np.pi
        tau_max = 2 * np.arctan(max_temp ** 0.5) / np.pi
        self.scale = 0.5 * np.pi * (tau_max - tau_min)
        self.shift = 0.5 * np.pi * tau_min

    def forward(self, tau: Tensor) -> Tensor:
        return (tau * self.scale + self.shift).tan().log() * 2  # type: ignore

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return ((log_temp * 0.5).exp().atan() - self.shift) / self.scale  # type: ignore


class InterpolatedDiscreteTimeNoiseScheduler(NoiseScheduler):
    def __init__(self, config: Config, timestamps: Tensor, temp: Tensor):
        super().__init__()

        self._get_log_temp = interp1d_tensor_fun(timestamps.numpy(), temp.log().numpy())
        self._get_tau = interp1d_tensor_fun(temp.log().numpy(), timestamps.numpy())

    def forward(self, tau: Tensor) -> Tensor:
        return self._get_log_temp(tau)

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return self._get_tau(log_temp)


class EntropyNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, config: Config, noise_schedule_type: str):
        stats = np.load(config.get_noise_schedule_stats_path(noise_schedule_type))
        temp = stats["temp"]
        entropy = stats["entropy"]
        timestamps = entropy - entropy.min()
        timestamps /= timestamps.max()

        super().__init__(config, timestamps, temp)


class FromDiffusersNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, config: Config):
        scheduler = get_diffusers_pipeline(config).scheduler # type: ignore
        alpha_bar = scheduler.alphas_cumprod
        log_temp = get_log_temp_from_alpha_bar(alpha_bar)

        super().__init__(config, torch.linspace(0, 1, len(log_temp)), log_temp)


def get_log_temp_from_alpha_bar(alpha_bar: Tensor) -> Tensor:
    return (1 - alpha_bar).log() - alpha_bar.log() # type: ignore


def get_alpha_bar_from_log_temp(log_temp: Tensor) -> Tensor:
    return sigmoid(-log_temp)


class DiffusionDynamic(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.obj_size = config.data.obj_size
        self.noise_scheduler = NoiseScheduler.from_config(config)

    def get_log_temp(self, tau: Tensor) -> Tensor:
        return self.noise_scheduler(tau).view(-1, *[1] * len(self.obj_size))

    def get_alpha_bar(self, tau: Tensor) -> Tensor:
        return get_alpha_bar_from_log_temp(self.get_log_temp(tau))

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
