from typing import Callable, TypeVar
import torch
from torch import Tensor


T = TypeVar("T")
V = TypeVar("V")


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def get_time_evenly_spaced(n_intervals: int, timestamp: float = 1.0, *, eps: float = 0.1) -> Tensor:
    eps /= n_intervals
    return torch.linspace(eps, timestamp - eps, int(timestamp * n_intervals))
