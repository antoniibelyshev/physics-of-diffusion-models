import torch
from diffusers import DDPMPipeline
from torch import Tensor, searchsorted
from torch.autograd.functional import jacobian
from denoising_diffusion_pytorch import Unet  # type: ignore
import numpy as np
from numpy.typing import NDArray
import argparse
from yaml import safe_load
from typing import Callable, Optional, ParamSpec, TypeVar, Concatenate, Any
from functools import wraps
from scipy.optimize import curve_fit  # type: ignore
from pydantic import BaseModel

from config import Config

T = TypeVar("T")
V = TypeVar("V")


# dict operations

def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def append_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, T]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + [val]


def add_dict(prev_dict: dict[str, Tensor], new_dict: dict[str, Tensor]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] += val


def extend_dict(prev_dict: dict[str, list[T]], new_dict: dict[str, list[T]]) -> None:
    for key, val in new_dict.items():
        prev_dict[key] = prev_dict[key] + val


# nn functions

def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x_: Tensor) -> Tensor:
        return func(x_).sum(dim=0)

    return jacobian(_func_sum, x).permute(1, 0, 2)  # type: ignore


def get_diffusers_pipeline(config: Config) -> DDPMPipeline:
    return DDPMPipeline.from_pretrained(config.dataset_config.diffusers_model_id)  # type: ignore


# Config

def flatten_config(config: BaseModel, parent_key: str = "") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in config:
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, BaseModel):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_args_from_config(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    flat_config = flatten_config(config)
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


def _load_yaml_file(path: str) -> dict:
    with open(path, "r") as f:
        return safe_load(f)


def _materialize_group_paths(config_root: dict, base_dir: str) -> dict:
    """
    Replace any top-level string values that point to YAML files with the loaded dicts.
    This lets config/config.yaml stay high-level and delegate per-group parameters
    into separate YAML files located in a sub-directory.
    """
    materialized: dict[str, Any] = {}
    for key, val in config_root.items():
        if isinstance(val, str) and val.endswith((".yml", ".yaml")):
            import os
            path = val if os.path.isabs(val) else os.path.join(base_dir, val)
            group_dict = _load_yaml_file(path)
            materialized[key] = group_dict
        else:
            materialized[key] = val
    return materialized


def load_config(config_path: Optional[str] = None) -> Config:
    cfg_path = config_path or "config/config.yaml"
    root = _load_yaml_file(cfg_path)

    # Resolve group file references relative to the config file directory
    import os
    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    root = _materialize_group_paths(root, base_dir)

    return Config(**root)


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
        idx_right = searchsorted(x_vals, _x).clamp(1, len(x_vals) - 1)

        xl, xr = x_vals[idx_right - 1], x_vals[idx_right]
        yl, yr = y_vals[idx_right - 1], y_vals[idx_right]

        wl = torch.where(xl == xr, 0.5, (xr - _x) / (xr - xl))
        return (wl * yl + (1 - wl) * yr).to(device)  # type: ignore

    return interpolate


def compute_cdf(x: ArrayT, non_normalized_p: ArrayT) -> ArrayT:
    cdf = np.cumsum(np.append(0, 0.5 * (non_normalized_p[1:] + non_normalized_p[:-1]) / (x[1:] - x[:-1])))
    return cdf / cdf[-1]  # type: ignore


def parse_value(value: Any) -> Any:
    if value == "None":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    return value
