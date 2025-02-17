from .diffusion_utils import get_temp_schedule, get_alpha_bar
import torch
from torch import Tensor, nn
from utils import norm_sqr
from config import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMDynamic(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.obj_size = config.data.obj_size
        self.temp_schedule = get_temp_schedule(config)
        self.continuous_time = config.ddpm_training.continuous_time
        self.timestamps = torch.linspace(config.ddpm.min_t, 1, config.sample.n_steps, device=DEVICE)

    def get_temp(self, t: Tensor) -> Tensor:
        return self.temp_schedule(t).view(-1, *[1] * len(self.obj_size))

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        return get_alpha_bar(self.get_temp(t))

    def forward(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.continuous_time:
            t = torch.rand((len(x0),), device=x0.device)
        else:
            t = self.timestamps[torch.randint(0, len(self.timestamps), (len(x0),), device=x0.device)]

        alpha_bar = self.get_alpha_bar(t)
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + eps * (1 - alpha_bar).sqrt()

        return t, eps, xt

    def get_true_score(self, xt: Tensor, t: Tensor, train_data: Tensor) -> Tensor:
        alpha_bar = self.get_alpha_bar(t)
        diffs = train_data.unsqueeze(1) * alpha_bar.sqrt() - xt
        n, bs, *obj_size = diffs.shape
        pows = -norm_sqr(diffs.view(n * bs, -1)).view(n, bs, *[1] * len(obj_size))
        pows /= 2 * (1 - alpha_bar)
        pows -= pows.max(0).values
        exps = pows.exp()
        diffs *= exps
        return diffs.sum(0) / (exps.sum(0) * (1 - alpha_bar)) # type: ignore
