import torch
from torch import nn, Tensor, from_numpy
from torch.nn.functional import sigmoid
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod

from config import Config
from utils import norm_sqr, interp1d, get_diffusers_pipeline, compute_pw_dist_sqr, extrapolate_entropy


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
    def from_config(config: Config, *, noise_schedule_type: Optional[str] = None, noise_schedule_path: Optional[str] = None) -> "NoiseScheduler":
        noise_schedule_type = noise_schedule_type or config.ddpm.noise_schedule_type
        if noise_schedule_type == "linear_beta":
            return LinearBetaNoiseScheduler(*config.diffusion.temp_range)
        elif noise_schedule_type == "cosine":
            return CosineNoiseScheduler(*config.diffusion.temp_range)
        elif noise_schedule_type == "entropy":
            return EntropyNoiseScheduler(
                config.forward_stats_path,
                config.entropy_schedule.extrapolate,
                config.entropy_schedule.min_temp,
                config.entropy_schedule.max_temp,
            )
        elif noise_schedule_type == "log_snr":
            return LogSNRNoiseScheduler(*config.diffusion.temp_range)
        elif noise_schedule_type == "diffusers":
            scheduler = get_diffusers_pipeline(config).scheduler  # type: ignore
            return FromDiffusersNoiseScheduler(scheduler.alphas_cumprod)
        elif noise_schedule_type == "custom":
            if noise_schedule_path is None:
                raise ValueError("noise_schedule_path must be provided for custom noise schedule")
            return CustomNoiseScheduler(noise_schedule_path)
        else:
            raise ValueError(f"Unknown schedule type: {noise_schedule_type}")


class LinearBetaNoiseScheduler(NoiseScheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        self.scale = 1 + min_temp
        self.gamma = np.log((1 + max_temp) / self.scale)

    def forward(self, tau: Tensor) -> Tensor:
        return ((tau.pow(2) * self.gamma).exp() * self.scale - 1).log()  # type: ignore

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return (((log_temp.exp() + 1) / self.scale).log() / self.gamma).sqrt()  # type: ignore


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        tau_min = 2 * np.arctan(min_temp ** 0.5) / np.pi
        tau_max = 2 * np.arctan(max_temp ** 0.5) / np.pi
        self.scale = 0.5 * np.pi * (tau_max - tau_min)
        self.shift = 0.5 * np.pi * tau_min

    def forward(self, tau: Tensor) -> Tensor:
        return (tau * self.scale + self.shift).tan().log() * 2  # type: ignore

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return ((log_temp * 0.5).exp().atan() - self.shift) / self.scale  # type: ignore


class LogSNRNoiseScheduler(NoiseScheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        self.min_log_temp = np.log(min_temp)
        self.max_log_temp = np.log(max_temp)

    def forward(self, tau: Tensor) -> Tensor:
        return self.min_log_temp * (1 - tau) + self.max_log_temp * tau

    def get_tau(self, log_temp: Tensor) -> Tensor:
        return (log_temp - self.min_log_temp) / (self.max_log_temp - self.min_log_temp)


ArrayT = np.typing.NDArray[np.float32]


class InterpolatedDiscreteTimeNoiseScheduler(NoiseScheduler):
    def __init__(self, timestamps: Tensor, log_temp: Tensor):
        super().__init__()

        self.register_buffer("timestamps", timestamps)
        self.register_buffer("log_temp", log_temp)
        self._update_interpolators()

    def _update_interpolators(self) -> None:
        self._get_log_temp = interp1d(self.timestamps, self.log_temp)
        self._get_tau = interp1d(self.log_temp, self.timestamps)

    def forward(self, tau: Tensor) -> Tensor:
        if self.timestamps.device != self.log_temp.device or self.timestamps.device != tau.device:
             self._update_interpolators()
        return self._get_log_temp(tau)

    def get_tau(self, log_temp: Tensor) -> Tensor:
        if self.timestamps.device != self.log_temp.device or self.timestamps.device != log_temp.device:
             self._update_interpolators()
        return self._get_tau(log_temp)


class CustomNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, path: str):
        if path.endswith(".npz"):
            stats = np.load(path)
            log_temp = from_numpy(stats["log_temp"])
            if "timestamps" in stats:
                timestamps = from_numpy(stats["timestamps"])
            else:
                timestamps = torch.linspace(0, 1, len(log_temp))
        else:
            log_temp = torch.load(path)
            timestamps = torch.linspace(0, 1, len(log_temp))

        super().__init__(timestamps, log_temp)


class EntropyNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(
        self,
        forward_stats_path: str,
        extrapolate: bool,
        min_temp: float,
        max_temp: float,
    ):
        stats = np.load(forward_stats_path)
        temp = from_numpy(stats["temp"])
        entropy = from_numpy(stats["entropy"])

        if extrapolate:
            temp, entropy = extrapolate_entropy(temp, entropy, min_temp)

            mask = temp <= max_temp

            temp = temp[mask]
            entropy = entropy[mask]

        timestamps = entropy - entropy.min()
        timestamps /= timestamps.max()

        super().__init__(timestamps, temp.log())


class FromDiffusersNoiseScheduler(InterpolatedDiscreteTimeNoiseScheduler):
    def __init__(self, alpha_bar: Tensor):
        log_temp = get_log_temp_from_alpha_bar(alpha_bar)

        super().__init__(torch.linspace(0, 1, len(log_temp)), log_temp)


def get_log_temp_from_alpha_bar(alpha_bar: Tensor) -> Tensor:
    return (1 - alpha_bar).log() - alpha_bar.log()  # type: ignore


def get_alpha_bar_from_log_temp(log_temp: Tensor) -> Tensor:
    return sigmoid(-log_temp)


class DiffusionDynamic(nn.Module):
    def __init__(self, obj_size: tuple[int, ...], noise_scheduler: NoiseScheduler) -> None:
        super().__init__()

        self.obj_size = obj_size
        self.noise_scheduler = noise_scheduler

    @classmethod
    def from_config(cls, config: Config) -> "DiffusionDynamic":
        return cls(
            obj_size=config.dataset_config.obj_size,
            noise_scheduler=NoiseScheduler.from_config(config),
        )

    def get_log_temp(self, tau: Tensor) -> Tensor:
        return self.noise_scheduler(tau).view(-1, *[1] * len(self.obj_size))

    def get_alpha_bar(self, tau: Tensor) -> Tensor:
        return get_alpha_bar_from_log_temp(self.get_log_temp(tau))

    def forward(self, x0: Tensor, tau: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        tau = torch.rand((len(x0),), device=x0.device) if tau is None else tau
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
    
    @torch.autocast("cuda", enabled=False)  # type: ignore
    def get_true_posterior_mean_x0(self, xt: Tensor, tau: Tensor, data: Tensor) -> Tensor:
        xt = xt.float()
        data = data.float()
        alpha_bar = self.get_alpha_bar(tau)
        h = 0.5 * compute_pw_dist_sqr(xt, alpha_bar.sqrt() * data)
        h = h - h.min(1, keepdim=True).values
        exp = -h / (1 - alpha_bar).view(-1, 1)
        p = exp.exp()
        p = p / p.sum(1, keepdim=True)
        return torch.matmul(p, data.view(len(data), -1)).view(-1, *data.shape[1:])
