from typing import Callable, TypeVar
import torch
from torch import Tensor, nn


T = TypeVar("T")
V = TypeVar("V")


def dict_map(func: Callable[[T], V], d: dict[str, T]) -> dict[str, V]:
    return {key: func(val) for key, val in d.items()}


def get_time_evenly_spaced(n_intervals: int, *, min_t: float = 1e-5) -> Tensor:
    return torch.linspace(min_t, 1.0 - min_t, n_intervals)


def replace_activations(
    module: nn.Module,
    new_activation: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2),
    activations_to_replace = (nn.ReLU, nn.SiLU, nn.GELU),
) -> nn.Module:
    if isinstance(module, activations_to_replace):
        return new_activation()
    for name, child in module.named_children():
        setattr(module, name, replace_activations(child, new_activation, activations_to_replace))
    return module
