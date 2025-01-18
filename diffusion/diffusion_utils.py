from typing import Callable
from torch import Tensor, from_numpy
from scipy.interpolate import interp1d # type: ignore
from scipy.integrate import cumulative_trapezoid # type: ignore
import numpy as np

from config import Config


# def dict_to_device(dct: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
#     return {k: v.to(device) for k, v in dct.items()}


def get_flattening_temp_schedule(stats_path: str) -> Callable[[Tensor], Tensor]:
    stats = np.load(stats_path)
    stats_temp = stats["temp"]
    stats_log_temp = np.log(stats_temp)
    heat_capacity = stats["var_H"] / stats_temp ** 2
    cdf_values = cumulative_trapezoid(heat_capacity, stats_log_temp)
    cdf_values /= cdf_values[-1]
    inverse_cdf_interp = interp1d(
        cdf_values,
        0.5 * (stats_log_temp[1:] + stats_log_temp[:-1]),
        kind='linear',
        fill_value="extrapolate"
    )
    return lambda t: from_numpy(inverse_cdf_interp(t.cpu().numpy())).exp().to(t.device)


def get_linear_beta_temp_schedule(beta0: float, beta1: float) -> Callable[[Tensor], Tensor]:
    def linear_beta_temp_schedule(t: Tensor) -> Tensor:
        return (beta0 * t + beta1 * t ** 2 / 2).exp() - 1 # type: ignore
    return linear_beta_temp_schedule


def get_temp_schedule(config: Config) -> Callable[[Tensor], Tensor]:
    if config.ddpm.schedule_type == "linear_beta":
        return get_linear_beta_temp_schedule(config.ddpm.beta0, config.ddpm.beta1)
    elif config.ddpm.schedule_type.startswith("flattening"):
        return get_flattening_temp_schedule(config.flattening_temp_stats_path)
    else:
        raise ValueError(f"Unknown schedule type: {config.ddpm.schedule_type}")
