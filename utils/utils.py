from typing import Callable, TypeVar
import torch
from torch import Tensor


T = TypeVar("T")
V = TypeVar("V")


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def get_time_evenly_spaced(n_intervals: int, timestamp: float = 1.0, *, a: float = 0.1) -> Tensor:
    return torch.linspace(timestamp, a / n_intervals, int(timestamp * n_intervals))
