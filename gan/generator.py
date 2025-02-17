import torch
from torch import nn, Tensor
from utils import get_unet
from config import Config


class GANGenerator(nn.Module):
    def __init__(self, config: Config, t: float = 1e-3):
        super().__init__()

        self.unet = get_unet(config.gan.base_channels_g, config.gan.dim_mults_g, 2 * config.data.obj_size[0], True)
        self.t = t
        self.img_channels = config.data.obj_size[0]

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.sigmoid(self.unet(
            torch.cat([x, torch.randn_like(x)], dim=1),
            torch.full((len(x),), self.t, device=x.device),
        ))[:, :self.img_channels] # type: ignore
