from .diffusion_utils import get_coeffs_primitives
import torch
from torch import Tensor, LongTensor, nn
from typing import Optional
from utils import norm_sqr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMDynamic(nn.Module):
    def __init__(self, beta: Optional[Tensor] = None):
        super().__init__()
        coeffs_primitives = get_coeffs_primitives(beta=beta)

        for name, tensor in coeffs_primitives.items():
            self.register_buffer(name, tensor)

        self.T = len(next(iter(coeffs_primitives.values())))

    def sample_time_on_device(self, batch_size: int = 1, device: torch.device = DEVICE) -> LongTensor:
        return torch.randint(0, self.T, (batch_size,), device=device) # type: ignore

    def get_coef_from_time(self, coef_name: str, t: LongTensor) -> Tensor:
        return getattr(self, coef_name)[t].unsqueeze(1).unsqueeze(2).unsqueeze(3) # type: ignore

    def sample_from_posterior_q(self, xt: Tensor, x0: Tensor, t: LongTensor) -> Tensor:
        posterior_x0_coef = self.get_coef_from_time("posterior_x0_coef", t)
        posterior_xt_coef = self.get_coef_from_time("posterior_xt_coef", t)
        posterior_sigma = self.get_coef_from_time("posterior_sigma", t)

        sample = posterior_x0_coef * x0 + posterior_xt_coef * xt
        eps = torch.randn(x0.shape).to(x0.device)
        return sample + eps * posterior_sigma.sqrt()
    
    def get_x0_from_x0(self, xt: Tensor, x0: Tensor, t: LongTensor) -> Tensor:
        return x0
    
    def get_x0_from_eps(self, xt: Tensor, eps: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt() # type: ignore

    def get_x0_from_score(self, xt: Tensor, score: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - score * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt() # type: ignore

    def get_eps_from_x0(self, xt: Tensor, x0: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - x0 * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt() # type: ignore

    def get_eps_from_eps(self, xt: Tensor, eps: Tensor, t: LongTensor) -> Tensor:
        return eps

    def get_eps_from_score(self, xt: Tensor, score: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return -score * (1 - alpha_bar).sqrt() # type: ignore

    def get_score_from_x0(self, xt: Tensor, x0: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return (xt - x0 * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt() # type: ignore

    def get_score_from_eps(self, xt: Tensor, eps: Tensor, t: LongTensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        return -eps / (1 - alpha_bar).sqrt() # type: ignore
    
    def get_score_from_score(self, xt: Tensor, score: Tensor, t: LongTensor) -> Tensor:
        return score
    
    def get_true_score(self, xt: Tensor, t: LongTensor, train_data: Tensor) -> Tensor:
        alpha_bar = self.get_coef_from_time("alpha_bar", t)
        diffs = train_data.unsqueeze(1) * alpha_bar.sqrt() - xt
        ts, bs, *obj_shape = diffs.shape
        pows = -norm_sqr(diffs.view(bs * ts, -1)).view(ts, bs, *[1] * len(obj_shape))
        pows /= 2 * (1 - alpha_bar)
        pows -= pows.max(0).values
        exps = pows.exp()
        weighted_diffs = diffs * exps / (1 - alpha_bar)
        return weighted_diffs.sum(0) / exps.sum(0) # type: ignore

    def forward(self, batch: dict[str, Tensor | LongTensor]) -> dict[str, Tensor]:
        x0 = batch["x0"]
        t = batch["t"]

        assert type(t) == LongTensor
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
