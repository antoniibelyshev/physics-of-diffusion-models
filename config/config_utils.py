from .config import Config
from yaml import safe_load
from typing import Callable, Any, Optional


def load_config(config_path: Optional[str] = None) -> Config:
    with open(config_path or "config/config.yaml", "r") as f:
        config_dict = safe_load(f)
    if isinstance(config_dict["data"].get("obj_size"), list):
        config_dict["data"]["obj_size"] = tuple(config_dict["data"]["obj_size"])
    config = Config(**config_dict)
    return config


def with_config(config_path: Optional[str] = None) -> Callable[[Callable[[Config], Any]], Callable[[], Any]]:
    def decorator(func: Callable[[Config], Any]) -> Callable[[], Any]:
        def wrapper() -> Any:
            config = load_config(config_path)
            return func(config)
        return wrapper
    return decorator