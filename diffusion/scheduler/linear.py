import numpy as np
from torch import Tensor
from .scheduler import Scheduler

class LinearBetaScheduler(Scheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        self.scale = 1 + min_temp
        self.gamma = np.log((1 + max_temp) / self.scale)

    def log_temp_from_tau(self, tau: Tensor) -> Tensor:
        return ((tau.pow(2) * self.gamma).exp() * self.scale - 1).log()  # type: ignore

    def tau_from_log_temp(self, log_temp: Tensor) -> Tensor:
        return (((log_temp.exp() + 1) / self.scale).log() / self.gamma).sqrt()  # type: ignore
