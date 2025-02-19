from torch import nn, Tensor, load, searchsorted, clip, where, abs, from_numpy
from pytorch_diffusion import Diffusion # type: ignore
from typing import Callable

from .diffusion_dynamic import DDPMDynamic
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
        self.dynamic = DDPMDynamic(config)
        self.parametrization = config.ddpm.parametrization
        assert self.parametrization in ["x0", "eps", "score"]
    
    def get_predictions(self, xt: Tensor, tau: Tensor) -> DDPMPredictions:
        return DDPMPredictions(self(xt, tau), xt, self.dynamic.get_alpha_bar(tau), self.parametrization)


class DDPMUnet(DDPM):
    def __init__(self, config: Config):
        super().__init__(config)

        self.unet = get_unet(config.ddpm.dim, config.ddpm.dim_mults, config.data.obj_size[0], config.ddpm.use_lrelu)

    def forward(self, xt: Tensor, tau: Tensor | int) -> Tensor:
        return self.unet(xt, tau) # type: ignore


class DDPMTrue(DDPM):
    train_data: Tensor

    def __init__(self, config: Config):
        assert config.ddpm.parametrization == "score"
        super().__init__(config)

        self.register_buffer("train_data", get_data_tensor(config))

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.dynamic.get_true_score(xt, tau, self.train_data)


class DDPMPytorchDiffusion(DDPM):
    def __init__(self, config: Config) -> None:
        config.ddpm.parametrization = "eps"
        super().__init__(config)
        self.model = Diffusion.from_pretrained(config.data.dataset_name)
        self.tau_to_t = self.get_tau_to_t()

    def get_tau_to_t(self) -> Callable[[Tensor], Tensor]:
        temp_pd = from_numpy(self.model.sqrt_recipm1_alphas_cumprod).pow(2).cuda()

        def tau_to_t(tau: Tensor) -> Tensor:
            temp = self.dynamic.get_temp(tau).squeeze()
            idx = clip(searchsorted(temp_pd, temp), 1, len(temp_pd) - 1)
            left = temp_pd[idx - 1]
            right = temp_pd[idx]
            closest_idx = where(abs(temp - left) <= abs(temp - right), idx - 1, idx)
            return closest_idx.reshape(-1)

        return tau_to_t

    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        return self.model.model(xt, self.tau_to_t(tau)) # type: ignore


def get_ddpm(config: Config, pretrained: bool = False) -> DDPM:
    match config.ddpm.model_name:
        case "unet":
            ddpm = DDPMUnet(config)
            if pretrained:
                ddpm.load_state_dict(load(config.ddpm_checkpoint_path))
            return ddpm
        case "true":
            return DDPMTrue(config)
        case "pytorch_diffusion":
            return DDPMPytorchDiffusion(config)
        case _:
            raise ValueError(f"Unknown model name: {config.ddpm.model_name}")
