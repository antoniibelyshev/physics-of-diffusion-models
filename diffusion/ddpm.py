from abc import abstractmethod
from typing import Any, Optional

import torch
from torch import nn, Tensor, load, compile
from diffusers import UNet2DModel
from diffusers.models.attention_processor import AttnProcessor2_0

from utils import get_diffusers_pipeline
from .diffusion_dynamic import DiffusionDynamic
from config import Config
from utils import get_data_tensor, dict_map, parse_value


class DDPMPredictions:
    x0: Tensor
    eps: Tensor
    score: Tensor

    def __init__(self, pred: Tensor, xt: Tensor, alpha_bar: Tensor, parametrization: str) -> None:
        match parametrization:
            case "x0":
                self.x0 = pred
                self.eps = (xt - pred * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt()
                self.score = -self.eps / (1 - alpha_bar).sqrt()
            case "eps":
                self.x0 = (xt - pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                self.eps = pred
                self.score = -self.eps / (1 - alpha_bar).sqrt()
            case "score":
                self.x0 = (xt + pred * (1 - alpha_bar)) / alpha_bar.sqrt()
                self.eps = -pred * (1 - alpha_bar).sqrt()
                self.score = pred


class DDPM(nn.Module):
    def __init__(self, dynamic: DiffusionDynamic, parametrization: str):
        super().__init__()
        self.dynamic = dynamic
        self.parametrization = parametrization
        assert self.parametrization in ["x0", "eps", "score"]
    
    def get_predictions(self, xt: Tensor, log_temp: Tensor) -> DDPMPredictions:
        tau = self.dynamic.noise_scheduler.get_tau(log_temp).clip(0, 1)
        return DDPMPredictions(self(xt, tau), xt, self.dynamic.get_alpha_bar(tau), self.parametrization)

    @abstractmethod
    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        pass

    @classmethod
    def from_config(cls, config: Config, pretrained: bool = False) -> "DDPM":
        dynamic = DiffusionDynamic.from_config(config)
        match config.ddpm.model_name:
            case "unet":
                ddpm = DDPMUnet(
                    dynamic=dynamic,
                    parametrization=config.ddpm.parametrization,
                    image_size=config.dataset_config.image_size,
                    unet_config=config.ddpm.unet_config,
                )
                if pretrained:
                    checkpoint = load(config.ddpm_checkpoint_path)
                    if "model_state_dict" in checkpoint:
                        ddpm.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        ddpm.load_state_dict(checkpoint)
                return ddpm
            case "true":
                return DDPMTrue(
                    dynamic=dynamic,
                    parametrization=config.ddpm.parametrization,
                    train_data=get_data_tensor(config),
                )
            case "diffusers":
                pipeline = get_diffusers_pipeline(config)
                set_processor_recursively(pipeline.unet, AttnProcessor2_0)  # type: ignore
                torch.set_float32_matmul_precision("high")
                unet = compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)  # type: ignore
                time_scale = pipeline.scheduler.timesteps.max()  # type: ignore
                return DDPMDiffusers(
                    dynamic=dynamic,
                    parametrization="eps",
                    unet=unet,
                    time_scale=time_scale,
                )
            case _:
                raise ValueError(f"Unknown model name: {config.ddpm.model_name}")


class DDPMUnet(DDPM):
    def __init__(
        self,
        dynamic: DiffusionDynamic,
        parametrization: str,
        image_size: tuple[int, int],
        unet_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(dynamic, parametrization)

        unet = UNet2DModel(
            sample_size=image_size[0],
            **dict_map(parse_value, unet_config or {}),
        )  # type: ignore
        set_processor_recursively(unet, AttnProcessor2_0) # type: ignore
        torch.set_float32_matmul_precision('high')
        self.unet = compile(unet, mode="reduce-overhead") # type: ignore

    def forward(self, xt: Tensor, tau: Tensor | int) -> Tensor:
        return self.unet(xt, tau).sample  # type: ignore


class DDPMTrue(DDPM):
    def __init__(self, dynamic: DiffusionDynamic, parametrization: str, train_data: Tensor):
        super().__init__(dynamic, parametrization)
        self.register_buffer("train_data", train_data)

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.dynamic.get_true_posterior_mean_x0(xt, tau, self.train_data)  # type: ignore


def set_processor_recursively(module: nn.Module, processor_class: type) -> None:
    for submodule in module.children():
        if hasattr(submodule, "set_processor"):
            submodule.set_processor(processor_class())
        else:
            set_processor_recursively(submodule, processor_class)


class DDPMDiffusers(DDPM):
    def __init__(
        self,
        dynamic: DiffusionDynamic,
        parametrization: str,
        unet: nn.Module,
        time_scale: float,
    ) -> None:
        super().__init__(dynamic, parametrization)
        self.unet = unet
        self.time_scale = time_scale

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.unet(xt, (tau * self.time_scale)).sample # type: ignore
