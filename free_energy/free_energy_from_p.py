from torch import Tensor
import numpy as np


def free_energy_from_px(log_px: Tensor, temp: Tensor, d: int = 1024) -> Tensor:
    return -temp * (np.log(2 * np.pi) * d / 2 + temp.log() * d / 2 + log_px) # type: ignore


def free_energy_from_pz(log_pz: Tensor, temp: Tensor, d: int = 1024) -> Tensor:
    return -temp * (np.log(2 * np.pi) * d / 2 + temp.log() * d / 2 + log_pz) # type: ignore
