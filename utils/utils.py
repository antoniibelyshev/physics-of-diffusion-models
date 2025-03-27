import torch
from diffusers import DDPMPipeline
from torch import nn, Tensor, searchsorted
from torch.autograd.functional import jacobian
from denoising_diffusion_pytorch import Unet  # type: ignore
import numpy as np
from numpy.typing import NDArray
import argparse
from yaml import safe_load
from typing import Callable, Optional, ParamSpec, TypeVar, Concatenate, Any
from functools import wraps, partial
from scipy.optimize import curve_fit  # type: ignore

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

    return jacobian(_func_sum, x).permute(1, 0, 2)  # type: ignore


def get_diffusers_pipeline(config: Config) -> DDPMPipeline:
    return DDPMPipeline.from_pretrained(config.ddpm.get_diffusers_model_id(config.data.dataset_name))  # type: ignore


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
            parser.set_defaults(**{key: value})
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


ArrayT = NDArray[np.float32]


def interp1d(x_vals: Tensor, y_vals: Tensor) -> Callable[[Tensor], Tensor]:
    def interpolate(x: Tensor) -> Tensor:
        device = x.device
        _x = x.to(x_vals.device)
        left_idx = searchsorted(x_vals, _x).clamp(0, len(x_vals) - 2)

        xl, xr = x_vals[left_idx], x_vals[left_idx + 1]
        yl, yr = y_vals[left_idx], y_vals[left_idx + 1]

        wl = torch.where(xl == xr, 0.5, (xr - _x) / (xr - xl))
        return (wl * yl + (1 - wl) * yr).to(device)  # type: ignore

    return interpolate


def compute_cdf(x: ArrayT, non_normalized_p: ArrayT) -> ArrayT:
    cdf = np.cumsum(np.append(0, 0.5 * (non_normalized_p[1:] + non_normalized_p[:-1]) / (x[1:] - x[:-1])))
    return cdf / cdf[-1]  # type: ignore


def entropy_fun_gompertz(temp: ArrayT, b: float, logn: float) -> ArrayT:
    return logn * np.exp(-b / logn / temp) - logn  # type: ignore


def fit_entropy_fun(temp: Tensor, entropy: Tensor, n: float, n_effective: float) -> Callable[[Tensor], Tensor]:
    (b,), _ = curve_fit(partial(entropy_fun_gompertz, logn=np.log(n)), temp.numpy(), entropy.numpy(), p0=[1])
    return partial(entropy_fun_gompertz, b=b, logn=np.log(n_effective))  # type: ignore
