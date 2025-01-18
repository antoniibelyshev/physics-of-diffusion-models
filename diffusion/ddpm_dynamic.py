from .diffusion_utils import get_temp_schedule
import torch
from torch import Tensor, nn
from torch.nn.functional import pad
from utils import norm_sqr
from config import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicParams:
    def __init__(self, temp: Tensor) -> None:
        self.temp = temp
        self.alpha_bar = 1 / (temp + 1)
        alpha_bar_shifted = pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        alpha = self.alpha_bar / alpha_bar_shifted
        self.beta = 1 - alpha
        self.posterior_x0_coef = (alpha_bar_shifted.sqrt() * self.beta) / (1 - self.alpha_bar)
        self.posterior_xt_coef = (alpha.sqrt() * (1 - alpha_bar_shifted)) / (1 - self.alpha_bar)
        self.posterior_sigma = (1 - alpha_bar_shifted) / (1 - self.alpha_bar) * self.beta


class DDPMDynamic(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.obj_size = config.data.obj_size
        self.temp_schedule = get_temp_schedule(config)

    def get_dynamic_params(self, t: Tensor, unsqueeze: bool = True) -> DynamicParams:
        if unsqueeze:
            t = t.view(-1, *(1,) * len(self.obj_size))
        return DynamicParams(self.temp_schedule(t))

    def sample_from_posterior_q(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        posterior_x0_coef = self.get_coef_from_time("posterior_x0_coef", t)
        posterior_xt_coef = self.get_coef_from_time("posterior_xt_coef", t)
        posterior_sigma = self.get_coef_from_time("posterior_sigma", t)

        sample = posterior_x0_coef * x0 + posterior_xt_coef * xt
        eps = torch.randn(x0.shape).to(x0.device)
        return sample + eps * posterior_sigma.sqrt() # type: ignore

    def forward(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = torch.rand((len(x0),), device=x0.device)

        alpha_bar = self.get_dynamic_params(t).alpha_bar
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + eps * (1 - alpha_bar).sqrt()

        return t, eps, xt

    def get_true_score(self, xt: Tensor, t: Tensor, train_data: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        diffs = train_data.unsqueeze(1) * alpha_bar.sqrt() - xt
        ts, bs, *obj_shape = diffs.shape
        pows = -norm_sqr(diffs.view(bs * ts, -1)).view(ts, bs, *[1] * len(obj_shape))
        pows /= 2 * (1 - alpha_bar)
        pows -= pows.max(0).values
        exps = pows.exp()
        weighted_diffs = diffs * exps / (1 - alpha_bar)
        return weighted_diffs.sum(0) / exps.sum(0) # type: ignore
