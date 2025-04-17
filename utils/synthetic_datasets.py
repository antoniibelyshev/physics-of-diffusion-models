import torch
from torch import Tensor
from typing import Optional


def generate_simplex(d: int) -> Tensor:
    return torch.cat((torch.eye(d), torch.full((1, d), (1 - (1 + d) ** 0.5) / d)), 0)


def generate_cross_polytope(d: int) -> Tensor:
    return torch.cat((torch.eye(d), -torch.eye(d)))


def sample_on_hypersphere(d: int, n: Optional[int] = None) -> Tensor:
    samples = torch.randn(n or 10 * d, d)
    samples /= torch.norm(samples, dim=1, keepdim=True) / d ** 0.5
    return samples


def generate_gaussian(d: int, n: int = 1000) -> Tensor:
    return torch.randn(n, d)


def generate_dataset(name: str = "hypersphere", d: int = 100) -> Tensor:
    match name:
        case "simplex":
            return generate_simplex(d)
        case "cross-polytope":
            return generate_cross_polytope(d)
        case "hypersphere":
            return sample_on_hypersphere(d)
        case "gaussian":
            return generate_gaussian(d)
        case _:
            raise ValueError(f"Invalid name: {name}")
