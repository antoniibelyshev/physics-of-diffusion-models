import numpy as np
from torch import Tensor
from .scheduler import Scheduler

class LogSNRScheduler(Scheduler):
    def __init__(self, min_temp: float, max_temp: float):
        super().__init__()

        self.min_log_temp = np.log(min_temp)
        self.max_log_temp = np.log(max_temp)

    def log_temp_from_tau(self, tau: Tensor) -> Tensor:
        return self.min_log_temp * (1 - tau) + self.max_log_temp * tau

    def tau_from_log_temp(self, log_temp: Tensor) -> Tensor:
        return (log_temp - self.min_log_temp) / (self.max_log_temp - self.min_log_temp)
