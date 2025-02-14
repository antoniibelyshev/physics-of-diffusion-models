import torch
from torch import nn, Tensor
from utils import get_unet
from config import Config


class GANGenerator(nn.Module):
    def __init__(self, config: Config, t: float = 1e-3):
        super().__init__()

        self.unet = get_unet(config.gan.base_channels_g, config.gan.dim_mults_g, config.data.obj_size[0], True)
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x, torch.full((len(x),), self.t, device=x.device)) # type: ignore
