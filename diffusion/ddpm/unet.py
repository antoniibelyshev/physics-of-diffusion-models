from typing import Any, Optional
import torch
from torch import nn, Tensor, compile
from diffusers import UNet2DModel
from diffusers.models.attention_processor import AttnProcessor2_0

from .ddpm import DDPM
from ..scheduler import Scheduler
from utils import dict_map, parse_value


class DDPMUnet(DDPM):
    def __init__(
        self,
        scheduler: Scheduler,
        parametrization: str,
        image_size: tuple[int, int],
        unet_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(scheduler, parametrization)

        unet = UNet2DModel(
            sample_size=image_size[0],
            **dict_map(parse_value, unet_config or {}),
        )  # type: ignore
        set_processor_recursively(unet, AttnProcessor2_0) # type: ignore
        torch.set_float32_matmul_precision('high')
        self.unet = compile(unet, mode="reduce-overhead") # type: ignore

    def forward(self, xt: Tensor, tau: Tensor | int) -> Tensor:
        return self.unet(xt, tau).sample  # type: ignore


def set_processor_recursively(module: nn.Module, processor_class: type) -> None:
    for submodule in module.children():
        if hasattr(submodule, "set_processor"):
            submodule.set_processor(processor_class())
        else:
            set_processor_recursively(submodule, processor_class)
