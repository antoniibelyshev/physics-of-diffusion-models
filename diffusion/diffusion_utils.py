from torch import Tensor
import torch


def dict_to_device(dct: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in dct.items()}


def get_coeffs_primitives(
    T: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 2e-2,
    beta: Tensor | None = None,
) -> dict[str, Tensor]:
    if beta is None:
        beta = torch.linspace(beta_min, beta_max, T)

    alpha = 1 - beta
    alpha_bar = alpha.cumprod(dim=0)
    alpha_bar_shifted = torch.cat([torch.Tensor([1]), alpha_bar[:-1]])

    temp = (1 - alpha_bar) / alpha_bar

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
