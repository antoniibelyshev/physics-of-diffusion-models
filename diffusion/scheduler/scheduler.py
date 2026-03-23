from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np
from torch import Tensor
from torch.nn.functional import sigmoid

from config import Config
from utils import norm_sqr, compute_pw_dist_sqr


def log_temp_from_alpha_bar(alpha_bar: Tensor) -> Tensor:
    return (1 - alpha_bar).log() - alpha_bar.log()  # type: ignore


def alpha_bar_from_log_temp(log_temp: Tensor) -> Tensor:
    return sigmoid(-log_temp)


def cast_log_temp(log_temp: Tensor, target: Tensor) -> Tensor:
    return log_temp.view(-1, *[1] * (target.ndim - 1))


class Scheduler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log_temp_from_tau(self, tau: Tensor) -> Tensor:
        pass

    @abstractmethod
    def tau_from_log_temp(self, log_temp: Tensor) -> Tensor:
        pass

    def alpha_bar_from_tau(self, tau: Tensor) -> Tensor:
        return alpha_bar_from_log_temp(self.log_temp_from_tau(tau))

    def add_noise(self, x0: Tensor, tau: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        tau = torch.rand((len(x0),), device=x0.device) if tau is None else tau
        alpha_bar = cast_log_temp(self.alpha_bar_from_tau(tau), x0)
        eps = torch.randn_like(x0)
        xt = alpha_bar.sqrt() * x0 + eps * (1 - alpha_bar).sqrt()
        return tau, eps, xt

    def true_score(self, xt: Tensor, tau: Tensor, train_data: Tensor) -> Tensor:
        alpha_bar = cast_log_temp(self.alpha_bar_from_tau(tau), xt)
        diffs = train_data.unsqueeze(1) * alpha_bar.sqrt() - xt
        n, bs, *obj_size = diffs.shape
        pows = -norm_sqr(diffs.reshape(n * bs, -1)).view(n, bs, *[1] * len(obj_size))
        pows /= 2 * (1 - alpha_bar)
        pows -= pows.max(0).values
        exps = pows.exp()
        diffs *= exps
        return diffs.sum(0) / (exps.sum(0) * (1 - alpha_bar))  # type: ignore

    @torch.autocast(device_type="cuda", enabled=False)  # type: ignore
    @torch.autocast(device_type="mps", enabled=False)  # type: ignore
    def true_posterior_mean_x0(self, xt: Tensor, tau: Tensor, data: Tensor) -> Tensor:
        xt = xt.float()
        data = data.float()
        alpha_bar = cast_log_temp(self.alpha_bar_from_tau(tau), xt)
        h = 0.5 * compute_pw_dist_sqr(xt, alpha_bar.sqrt() * data.view(len(data), *xt.shape[1:]))
        h = h - h.min(1, keepdim=True).values
        exp = -h / (1 - alpha_bar).view(-1, 1)
        p = exp.exp()
        p = p / p.sum(1, keepdim=True)
        return torch.matmul(p, data.view(len(data), -1)).view_as(xt)
