from typing import Callable, TypeVar
from torch import nn
from denoising_diffusion_pytorch import Unet # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d # type: ignore


T = TypeVar("T")
V = TypeVar("V")
ArrayT = NDArray[np.float32]


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def replace_activations(
    module: nn.Module,
    new_activation: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2),
    activations_to_replace: tuple[type, ...] = (nn.ReLU, nn.SiLU, nn.GELU),
) -> nn.Module:
    if isinstance(module, activations_to_replace):
        return new_activation()
    for name, child in module.named_children():
        setattr(module, name, replace_activations(child, new_activation, activations_to_replace))
    return module


def get_unet(base_channels: int, dim_mults: list[int], channels: int, use_lrelu: bool = True) -> Unet:
    unet = Unet(
        dim=base_channels,
        dim_mults=dim_mults,
        channels=channels,
        flash_attn=True,
    )

    if use_lrelu:
        replace_activations(unet)

    return unet


def compute_cdf(x: ArrayT, p: ArrayT) -> ArrayT:
    cdf = np.cumsum(np.append(0, 0.5 * (p[1:] + p[:-1]) / (x[1:] - x[:-1])))
    return cdf / cdf[-1] # type: ignore


def get_cdf(x: ArrayT, p: ArrayT) -> Callable[[ArrayT], ArrayT]:
    cdf = compute_cdf(x, p)
    return interp1d(x, cdf, kind='linear', fill_value="extrapolate") # type: ignore


def get_inv_cdf(x: ArrayT, p: ArrayT) -> Callable[[ArrayT], ArrayT]:
    cdf = compute_cdf(x, p)
    return interp1d(cdf, x, kind='linear', fill_value="extrapolate") # type: ignore
