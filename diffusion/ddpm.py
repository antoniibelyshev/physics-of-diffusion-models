from abc import abstractmethod

import torch
from torch import nn, Tensor, load, compile
from diffusers.models.attention_processor import AttnProcessor2_0

from utils import get_diffusers_pipeline
from .diffusion_dynamic import DiffusionDynamic
from config import Config
from utils import get_data_tensor, get_unet


class DDPMPredictions:
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
    def __init__(self, config: Config):
        super().__init__()
        self.dynamic = DiffusionDynamic(config)
        self.parametrization = config.ddpm.parametrization
        assert self.parametrization in ["x0", "eps", "score"]
    
    def get_predictions(self, xt: Tensor, log_temp: Tensor) -> DDPMPredictions:
        tau = self.dynamic.noise_scheduler.get_tau(log_temp)
        return DDPMPredictions(self(xt, tau), xt, self.dynamic.get_alpha_bar(tau), self.parametrization)

    @abstractmethod
    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        pass

    @classmethod
    def from_config(cls, config: Config, pretrained: bool = False) -> "DDPM":
        match config.ddpm.model_name:
            case "unet":
                ddpm = DDPMUnet(config)
                if pretrained:
                    ddpm.load_state_dict(load(config.ddpm_checkpoint_path))
                return ddpm
            case "true":
                return DDPMTrue(config)
            case "diffusers":
                return DDPMDiffusers(config)
            case _:
                raise ValueError(f"Unknown model name: {config.ddpm.model_name}")


class DDPMUnet(DDPM):
    def __init__(self, config: Config):
        super().__init__(config)

        self.unet = get_unet(config.ddpm.dim, config.ddpm.dim_mults, config.data.obj_size[0], config.ddpm.use_lrelu)

    def forward(self, xt: Tensor, tau: Tensor | int) -> Tensor:
        return self.unet(xt, tau) # type: ignore


class DDPMTrue(DDPM):
    def __init__(self, config: Config):
        assert config.ddpm.parametrization == "score"
        super().__init__(config)

        self.register_buffer("train_data", get_data_tensor(config))

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.dynamic.get_true_score(xt, tau, self.train_data)


def set_processor_recursively(module: nn.Module, processor_class: type) -> None:
    for submodule in module.children():
        if hasattr(submodule, "set_processor"):
            submodule.set_processor(processor_class())
        else:
            set_processor_recursively(submodule, processor_class)


class DDPMDiffusers(DDPM):
    def __init__(self, config: Config) -> None:
        config.ddpm.parametrization = "eps"
        super().__init__(config)

        pipeline = get_diffusers_pipeline(config)
        # pipeline.unet.set_attn_processor(AttnProcessor2_0())
        set_processor_recursively(pipeline.unet, AttnProcessor2_0) # type: ignore
        torch.set_float32_matmul_precision('high')
        self.unet = compile(pipeline.unet, mode="reduce-overhead", fullgraph=True) # type: ignore
        # self.unet = pipeline.unet # type: ignore
        self.n_steps = len(pipeline.scheduler.timesteps) # type: ignore

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.unet(xt, tau * self.n_steps).sample # type: ignore
