import torch
from torch import nn, Tensor, from_numpy
from torch.autograd.functional import jacobian
from denoising_diffusion_pytorch import Unet # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d # type: ignore
import argparse
from yaml import safe_load
from typing import Callable, Optional, ParamSpec, TypeVar, Concatenate, Any
from functools import wraps

from config import Config


T = TypeVar("T")
V = TypeVar("V")


# dict operations

def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def append_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, T]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + [val]


def extend_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, list[T]]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + val


# nn functions

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


def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x_: Tensor) -> Tensor:
        return func(x_).sum(dim=0)

    return jacobian(_func_sum, x).permute(1, 0, 2) # type: ignore


# Smooth cdf

ArrayT = NDArray[np.float32]


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


# Config

def parse_args_from_config(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    def flatten_config(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        items: list[Any] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_config(config.model_dump())
    for key, value in flat_config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", help=f"Enable {key}")
            parser.add_argument(f"--no-{key}", dest=key, action="store_false", help=f"Disable {key}")
        else:
            arg_type = type(value)
            parser.add_argument(f"--{key}", type=arg_type, help=f"Set config value for {key}")

    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> None:
    for arg_key, arg_value in vars(args).items():
        if arg_value is None:
            continue

        keys = arg_key.split(".")
        sub_config = config

        for key in keys[:-1]:
            sub_config = getattr(sub_config, key)

        setattr(sub_config, keys[-1], arg_value)


def load_config(config_path: Optional[str] = None) -> Config:
    with open(config_path or "config/config.yaml", "r") as f:
        config_dict = safe_load(f)
    return Config(**config_dict)


P = ParamSpec("P")
R = TypeVar("R")


def with_config(
        config_path: Optional[str] = None,
        *,
        parse_args: bool = False,
) -> Callable[[Callable[Concatenate[Config, P], R]], Callable[P, R]]:
    config = load_config(config_path)
    if parse_args:
        script_args = parse_args_from_config(config)
        update_config_from_args(config, script_args)
    def decorator(func: Callable[Concatenate[Config, P], R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(config, *args, **kwargs)
        return wrapper
    return decorator


# Other

def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
