from torch import Tensor
from .ddpm import DDPM
from ..scheduler import Scheduler


class DDPMTrue(DDPM):
    def __init__(self, scheduler: Scheduler, parametrization: str, train_data: Tensor):
        super().__init__(scheduler, parametrization)
        self.register_buffer("train_data", train_data)

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.scheduler.true_posterior_mean_x0(xt, tau, self.train_data)  # type: ignore
