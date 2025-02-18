import numpy as np
from numpy.typing import NDArray


ArrayT = NDArray[np.float32]


def finite_diff_derivative(x: ArrayT, fx: ArrayT) -> ArrayT:
    return (fx[1:] - fx[:-1]) / (x[1:] - x[:-1])
