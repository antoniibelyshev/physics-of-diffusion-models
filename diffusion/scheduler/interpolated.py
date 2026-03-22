from torch import Tensor
from utils import interp1d
from .scheduler import Scheduler

class InterpolatedDiscreteTimeScheduler(Scheduler):
    def __init__(self, timestamps: Tensor, log_temp: Tensor):
        super().__init__()

        self.timestamps = timestamps
        self.log_temp = log_temp
        self._update_interpolators()

    def _update_interpolators(self) -> None:
        self._log_temp_from_tau = interp1d(self.timestamps, self.log_temp)
        self._tau_from_log_temp = interp1d(self.log_temp, self.timestamps)

    def log_temp_from_tau(self, tau: Tensor) -> Tensor:
        return self._log_temp_from_tau(tau.to(self.log_temp.device)).to(tau.device)

    def tau_from_log_temp(self, log_temp: Tensor) -> Tensor:
        return self._tau_from_log_temp(log_temp.to(self.log_temp.device)).to(log_temp.device)
