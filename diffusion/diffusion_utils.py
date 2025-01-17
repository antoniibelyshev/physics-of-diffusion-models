from scipy.integrate import cumulative_trapezoid # type: ignore
from scipy.interpolate import interp1d # type: ignore
from torch import Tensor, from_numpy
from torch.nn.functional import pad
import torch
import numpy as np

from config import Config


def dict_to_device(dct: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in dct.items()}


def get_coeffs_primitives(config: Config) -> dict[str, Tensor]:
    ddpm_config = config.ddpm
    total_time = ddpm_config.total_time

    if ddpm_config.schedule_type == "linear_beta":
        beta = torch.linspace(ddpm_config.beta_min, ddpm_config.beta_max, total_time)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(dim=0)
        temp = (1 - alpha_bar) / alpha_bar
    else:
        temp = get_flattening_temp(config.flattening_temp_stats_path, total_time)
        alpha_bar = 1 / (temp + 1)
        alpha = alpha_bar / pad(alpha_bar[:-1], (1, 0), value=1.0)
        beta = 1 - alpha_bar

    alpha_bar_shifted = pad(alpha_bar[:-1], (1, 0), value=1.0)

    posterior_x0_coef = (alpha_bar_shifted.sqrt() * beta) / (1 - alpha_bar)
    posterior_xt_coef = (alpha.sqrt() * (1 - alpha_bar_shifted)) / (1 - alpha_bar)
    posterior_sigma = (1 - alpha_bar_shifted) / (1 - alpha_bar) * beta

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "alpha_bar_shifted": alpha_bar_shifted,
        "posterior_x0_coef": posterior_x0_coef,
        "posterior_xt_coef": posterior_xt_coef,
        "posterior_sigma": posterior_sigma,
        "temp": temp,
    }


def get_flattening_temp(stats_filename: str, total_time: int) -> Tensor:
    stats = np.load(stats_filename)
    stats_log_temp = np.log(stats["temp"])
    heat_capacity = stats["C"]
    cdf_values = cumulative_trapezoid(heat_capacity, stats_log_temp)
    cdf_values /= cdf_values[-1]
    inverse_cdf_interp = interp1d(cdf_values, stats_log_temp, kind='linear', fill_value="extrapolate")
    return from_numpy(np.exp(inverse_cdf_interp(torch.linspace(0, 1, total_time))))
