import numpy as np
from torch import Tensor
from .scheduler import Scheduler

class CosineScheduler(Scheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        tau_min = 2 * np.arctan(min_temp ** 0.5) / np.pi
        tau_max = 2 * np.arctan(max_temp ** 0.5) / np.pi
        self.scale = 0.5 * np.pi * (tau_max - tau_min)
        self.shift = 0.5 * np.pi * tau_min

    def log_temp_from_tau(self, tau: Tensor) -> Tensor:
        return (tau * self.scale + self.shift).tan().log() * 2  # type: ignore

    def tau_from_log_temp(self, log_temp: Tensor) -> Tensor:
        return ((log_temp * 0.5).exp().atan() - self.shift) / self.scale  # type: ignore
