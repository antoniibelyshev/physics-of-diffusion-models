from torch import nn, Tensor
import torch
from denoising_diffusion_pytorch import Unet # type: ignore
from .ddpm_dynamic import DDPMDynamic
from typing import Optional


class DDPM(nn.Module):
    def __init__(self, beta: Optional[Tensor] = None, parametrization: str = "x0"):
        super().__init__()
        self.dynamic = DDPMDynamic(beta)

        assert parametrization in ["x0", "eps", "score"]
        self.parametrization = parametrization

    def get_val(self, val: str, xt: Tensor, t: Tensor) -> Tensor:
        assert val in ["x0", "eps", "score"]

        return getattr(self.dynamic, f"get_{val}_from_{self.parametrization}")(xt, self(xt, t), t) # type: ignore


class DDPMUnet(DDPM):
    def __init__(self, beta: Optional[Tensor] = None, parametrization: str = "x0"):
        super().__init__(beta, parametrization)

        self.unet = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 4),
            channels = 1,
            flash_attn = True,
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.unet(xt, t) # type: ignore
    

class DDPMTrue(DDPM):
    def __init__(self, train_data: Tensor, beta: Optional[Tensor] = None):
        super().__init__(beta, "score")

        self.register_buffer("train_data", train_data)

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.dynamic.get_true_score(xt, t, self.train_data)
