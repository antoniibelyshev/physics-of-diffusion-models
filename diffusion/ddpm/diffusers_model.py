from torch import nn, Tensor
from .ddpm import DDPM
from ..scheduler import Scheduler


class DDPMDiffusers(DDPM):
    def __init__(
        self,
        scheduler: Scheduler,
        parametrization: str,
        unet: nn.Module,
        time_scale: float,
    ) -> None:
        super().__init__(scheduler, parametrization)
        self.unet = unet
        self.time_scale = time_scale

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.unet(xt, (tau * self.time_scale)).sample # type: ignore
