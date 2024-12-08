import numpy as np
from numpy.typing import NDArray
from typing import Optional


def generate_simplex(d: int) -> NDArray[np.float64]:
    return np.concatenate((np.eye(d), np.ones((1, d)) * (1 - np.sqrt(1 + d)) / d), axis=0)


def generate_cross_polytope(d: int) -> NDArray[np.float64]:
    return np.concatenate((np.eye(d), -np.eye(d)), axis=0)


def sample_on_hypersphere(d: int, n: Optional[int] = None) -> NDArray[np.float64]:
    samples = np.random.randn(n or 10 * d, d)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return samples


def generate_dataset(name: str = "hypersphere", d: int = 100) -> NDArray[np.float64]:
    match name:
        case "simplex":
            return generate_simplex(d)
        case "cross-polytope":
            return generate_cross_polytope(d)
        case "hypersphere":
            return sample_on_hypersphere(d)
        case _:
            raise ValueError(f"Invalid name: {name}")
