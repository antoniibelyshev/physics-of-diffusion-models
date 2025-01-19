from torch import nn, Tensor, load
from denoising_diffusion_pytorch import Unet # type: ignore
from .ddpm_dynamic import DDPMDynamic, DynamicParams
from config import Config
from utils import get_data_tensor


class DDPMPredictions:
    def __init__(self, pred: Tensor, xt: Tensor, dynamic_params: DynamicParams, parametrization: str) -> None:
        alpha_bar = dynamic_params.alpha_bar
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
                self.x0 = (xt - pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                self.eps = -pred * (1 - alpha_bar).sqrt()
                self.score = pred


class DDPM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dynamic = DDPMDynamic(config)
        self.parametrization = config.ddpm.parametrization
        assert self.parametrization in ["x0", "eps", "score"]
    
    def get_predictions(self, xt: Tensor, t: Tensor | int) -> DDPMPredictions:
        return DDPMPredictions(self(xt, t), xt, self.dynamic.get_dynamic_params(t), self.parametrization)


class DDPMUnet(DDPM):
    def __init__(self, config: Config):
        super().__init__(config)

        self.unet = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 4),
            channels = 1,
            flash_attn = True,
        )

    def forward(self, xt: Tensor, t: Tensor | int) -> Tensor:
        return self.unet(xt, self.dynamic.get_dynamic_params(t, unsqueeze=False).temp) # type: ignore


class DDPMTrue(DDPM):
    train_data: Tensor

    def __init__(self, config: Config):
        assert config.ddpm.parametrization == "score"
        super().__init__(config)

        self.register_buffer("train_data", get_data_tensor(config))

    def forward(self, xt: Tensor, t: Tensor | int) -> Tensor:
        return self.dynamic.get_true_score(xt, t, self.train_data)


def get_ddpm(config: Config, pretrained: bool = False) -> DDPM:
    if config.ddpm.model_name == "unet":
        ddpm = DDPMUnet(config)
        if pretrained:
            ddpm.load_state_dict(load(config.ddpm_checkpoint_path))
        return ddpm
    elif config.ddpm.model_name == "true":
        return DDPMTrue(config)
    else:
        raise ValueError(f"Unknown model name: {config.ddpm.model_name}")
