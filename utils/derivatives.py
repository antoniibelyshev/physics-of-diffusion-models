import numpy as np
from numpy.typing import NDArray
from typing import TypeVar


T = TypeVar("T", bound=np.generic)


def finite_diff_derivative(x: NDArray[T], fx: NDArray[T]) -> NDArray[T]:
    return (fx[1:] - fx[:-1]) / (x[1:] - x[:-1]) # type: ignore
