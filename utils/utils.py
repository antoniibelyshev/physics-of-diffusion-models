import torch
from torch import nn, Tensor, from_numpy
from torch.autograd.functional import jacobian
from denoising_diffusion_pytorch import Unet # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d # type: ignore
from typing import Callable, TypeVar


T = TypeVar("T")
V = TypeVar("V")
ArrayT = NDArray[np.float32]


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def append_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, T]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + [val]


def extend_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, list[T]]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + val


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


def numpy_fun_to_tensor_fun(fun: Callable[[ArrayT], ArrayT]) -> Callable[[Tensor], Tensor]:
    return lambda tensor: from_numpy(fun(tensor.cpu().numpy())).float().to(tensor.device)


def compute_cdf(x: ArrayT, p: ArrayT) -> ArrayT:
    cdf = np.cumsum(np.append(0, 0.5 * (p[1:] + p[:-1]) / (x[1:] - x[:-1])))
    return cdf / cdf[-1] # type: ignore


def get_cdf(x: ArrayT, p: ArrayT) -> Callable[[Tensor], Tensor]:
    cdf = compute_cdf(x, p)
    return numpy_fun_to_tensor_fun(interp1d(x, cdf, kind='linear', fill_value="extrapolate"))


def get_inv_cdf(x: ArrayT, p: ArrayT) -> Callable[[Tensor], Tensor]:
    cdf = compute_cdf(x, p)
    return numpy_fun_to_tensor_fun(interp1d(cdf, x, kind='linear', fill_value="extrapolate"))


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x_: Tensor) -> Tensor:
        return func(x_).sum(dim=0)

    return jacobian(_func_sum, x).permute(1, 0, 2) # type: ignore
