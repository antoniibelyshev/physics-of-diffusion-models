import torch
from torch import nn, Tensor, from_numpy
from torch.nn.functional import sigmoid
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from config import Config
from utils import norm_sqr, interp1d, get_diffusers_pipeline, fit_entropy_fun

import matplotlib.pyplot as plt


class NoiseScheduler(nn.Module, ABC):
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)  # type: ignore

    @abstractmethod
    def forward(self, tau: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_tau(self, log_temp: Tensor) -> Tensor:
        pass

    @staticmethod
    def from_config(config: Config, *, noise_schedule_type: Optional[str] = None) -> "NoiseScheduler":
        if noise_schedule_type is None or noise_schedule_type == "original":
            noise_schedule_type = config.ddpm.noise_schedule_type
        if noise_schedule_type == "linear_beta":
            return LinearBetaNoiseScheduler(config)
        elif noise_schedule_type == "cosine":
            return CosineNoiseScheduler(config)
        elif noise_schedule_type.startswith("entropy"):
            return EntropyNoiseScheduler(config, noise_schedule_type)
        elif noise_schedule_type == "from_diffusers":
            return FromDiffusersNoiseScheduler(config)
        else:
            raise ValueError(f"Unknown schedule type: {noise_schedule_type}")


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


ArrayT = np.typing.NDArray[np.float32]


class InterpolatedDiscreteTimeNoiseScheduler(NoiseScheduler):
    def __init__(self, timestamps: Tensor, log_temp: Tensor):
        super().__init__()

        self._get_log_temp = interp1d(timestamps, log_temp)
        self._get_tau = interp1d(log_temp, timestamps)

    def forward(self, tau: Tensor) -> Tensor:
        return self._get_log_temp(tau)

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return self._get_tau(log_temp)


class EntropyNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, config: Config, noise_schedule_type: str):
        stats = np.load(config.get_noise_schedule_stats_path(noise_schedule_type))
        temp = from_numpy(stats["temp"])
        entropy = from_numpy(stats["entropy"])


        if noise_schedule_type.endswith("_extrapolated"):
            if config.sample.extrapolation_type == "linear_middle":
                gamma = 0.2
                left_mask = entropy / entropy.min() > 1 - gamma
                left_log_temp = temp[left_mask].log()
                left_entropy = entropy[left_mask]
                right_mask = entropy / entropy.min() < gamma
                right_log_temp = temp[right_mask].log()
                right_entropy = entropy[right_mask]
                mid_mask = ~(left_mask | right_mask)
                mid_log_temp = temp[mid_mask].log()
                mid_entropy = entropy[mid_mask]
                mid_x = torch.stack([torch.ones_like(mid_log_temp), mid_log_temp], 1)
                beta = (mid_x.T @ mid_x).inverse() @ mid_x.T @ mid_entropy
                l_log_temp = np.log(config.sample.l_temp)
                l_entropy = torch.tensor([[1., l_log_temp]]).float() @ beta
                temp = torch.cat([left_log_temp - left_log_temp.max() + l_log_temp, right_log_temp]).exp()
                entropy = torch.cat([left_entropy - left_entropy.max() + l_entropy, right_entropy])
            else:
                logn_effective = config.sample.logn_effective or np.log(config.data.dataset_size)
                step_fun_type = config.sample.extrapolation_type
                entropy_fun = fit_entropy_fun(temp, entropy, np.log(config.data.dataset_size), logn_effective, step_fun_type)
                entropy = from_numpy(entropy_fun(temp.numpy())).float()
                temp = temp.clamp(min=config.sample.min_temp)


        timestamps = entropy - entropy.min()
        timestamps /= timestamps.max()

        super().__init__(timestamps, temp.clip(min=1e-4).log())


class FromDiffusersNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, config: Config):
        scheduler = get_diffusers_pipeline(config).scheduler  # type: ignore
        alpha_bar = scheduler.alphas_cumprod
        log_temp = get_log_temp_from_alpha_bar(alpha_bar)

        super().__init__(torch.linspace(0, 1, len(log_temp)), log_temp)


def get_log_temp_from_alpha_bar(alpha_bar: Tensor) -> Tensor:
    return (1 - alpha_bar).log() - alpha_bar.log()  # type: ignore


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
        return diffs.sum(0) / (exps.sum(0) * (1 - alpha_bar))  # type: ignore
