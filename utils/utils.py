from typing import Callable, TypeVar
import torch
from torch import Tensor


T = TypeVar("T")
V = TypeVar("V")


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def get_time_evenly_spaced(n_intervals: int, *, min_t: float = 1e-5) -> Tensor:
    return torch.linspace(min_t, 1.0 - min_t, n_intervals)
