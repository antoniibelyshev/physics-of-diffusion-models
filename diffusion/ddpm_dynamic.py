from .diffusion_utils import get_coeffs_primitives
import torch
from torch import Tensor, nn
from typing import Any
from utils import norm_sqr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMDynamic(nn.Module):
    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor
    alpha_bar_shifted: Tensor
    posterior_x0_coef: Tensor
    posterior_xt_coef: Tensor
    posterior_sigma: Tensor
    temp: Tensor

    def __init__(self, obj_size: tuple[int, ...], **coeffs_kwargs: Any) -> None:
        super().__init__()

        self.obj_size = obj_size
        coeffs_primitives = get_coeffs_primitives(**coeffs_kwargs)

        for name, tensor in coeffs_primitives.items():
            self.register_buffer(name, tensor)

        self.T = len(next(iter(coeffs_primitives.values())))

    def sample_time_on_device(self, batch_size: int = 1, device: torch.device = DEVICE) -> Tensor:
        return torch.randint(0, self.T, (batch_size,), device=device)

    def get_coef_from_time(self, coef_name: str, t: Tensor) -> Tensor:
        coef_t = getattr(self, coef_name)[t]
        assert isinstance(coef_t, Tensor)
        return coef_t.view(-1, *([1] * len(self.obj_size)))

    def sample_from_posterior_q(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        posterior_x0_coef = self.get_coef_from_time("posterior_x0_coef", t)
        posterior_xt_coef = self.get_coef_from_time("posterior_xt_coef", t)
        posterior_sigma = self.get_coef_from_time("posterior_sigma", t)

        sample = posterior_x0_coef * x0 + posterior_xt_coef * xt
        eps = torch.randn(x0.shape).to(x0.device)
        return sample + eps * posterior_sigma.sqrt()
    
    def get_x0_from_eps(self, xt: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt() # type: ignore

    def get_x0_from_score(self, xt: Tensor, score: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - score * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt() # type: ignore

    def get_eps_from_x0(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - x0 * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt() # type: ignore

    def get_eps_from_score(self, xt: Tensor, score: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return -score * (1 - alpha_bar).sqrt() # type: ignore

    def get_score_from_x0(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - x0 * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt() # type: ignore

    def get_score_from_eps(self, xt: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return -eps / (1 - alpha_bar).sqrt() # type: ignore
    
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

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x0 = batch["x0"]
        t = batch["t"]

        alpha_bar = self.get_coef_from_time("alpha_bar", t)

        if "eps" in batch.keys():
            eps = batch["eps"]
        else:
            eps = torch.randn(x0.shape).to(x0.device)

        xt = alpha_bar.sqrt() * x0 + eps * (1 - alpha_bar).sqrt()

        return {
            "xt": xt,
            "eps": eps
        }
