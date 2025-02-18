from .config import Config
from yaml import safe_load
from typing import Callable, Optional, ParamSpec, TypeVar, Concatenate
from functools import wraps


def load_config(config_path: Optional[str] = None) -> Config:
    with open(config_path or "config/config.yaml", "r") as f:
        config_dict = safe_load(f)
    return Config(**config_dict)


P = ParamSpec("P")
R = TypeVar("R")


def with_config(
        config_path: Optional[str] = None
) -> Callable[[Callable[Concatenate[Config, P], R]], Callable[P, R]]:
    config = load_config(config_path)
    def decorator(func: Callable[Concatenate[Config, P], R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(config, *args, **kwargs)
        return wrapper
    return decorator
