from torch import nn, Tensor, load
from denoising_diffusion_pytorch import Unet # type: ignore
from .ddpm_dynamic import DDPMDynamic
from base_config import BaseConfig
from utils import get_data_tensor


class DDPM(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.dynamic = DDPMDynamic(config.data.obj_size, beta_min = config.ddpm.beta_min, beta_max = config.ddpm.beta_max, T = config.ddpm.T)

        assert config.ddpm.parametrization in ["x0", "eps", "score"]
        self.parametrization = config.ddpm.parametrization
    
    def get_predictions(self, xt: Tensor, t: Tensor) -> dict[str, Tensor]:
        res = self(xt, t)
        predictions = {self.parametrization: res}

        match self.parametrization:
            case "x0":
                predictions["eps"] = self.dynamic.get_eps_from_x0(xt, res, t)
                predictions["score"] = self.dynamic.get_score_from_x0(xt, res, t)
            case "eps":
                predictions["x0"] = self.dynamic.get_x0_from_eps(xt, res, t)
                predictions["score"] = self.dynamic.get_score_from_eps(xt, res, t)
            case "score":
                predictions["x0"] = self.dynamic.get_x0_from_score(xt, res, t)
                predictions["eps"] = self.dynamic.get_eps_from_score(xt, res, t)

        return predictions


class DDPMUnet(DDPM):
    def __init__(self, config: BaseConfig):
        super().__init__(config)

        self.unet = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 4),
            channels = 1,
            flash_attn = True,
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.unet(xt, t) # type: ignore


class DDPMTrue(DDPM):
    train_data: Tensor
    
    def __init__(self, config: BaseConfig):
        assert config.ddpm.parametrization == "score"
        super().__init__(config)

        self.register_buffer("train_data", get_data_tensor(config))

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.dynamic.get_true_score(xt, t, self.train_data)


def get_ddpm(config: BaseConfig, pretrained: bool = False) -> DDPM:
    if config.ddpm.model_name == "unet":
        ddpm = DDPMUnet(config)
        if pretrained:
            ddpm.load_state_dict(load(config.checkpoint_path))
        return ddpm
    elif config.ddpm.model_name == "true":
        return DDPMTrue(config)
    else:
        raise ValueError(f"Unknown model name: {config.ddpm.model_name}")
