from abc import abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor, load, compile

from ..scheduler import Scheduler, cast_log_temp
from config import Config
from utils import get_data_tensor, get_diffusers_pipeline


class DDPMPredictions:
    def __init__(self, pred: Tensor, xt: Tensor, alpha_bar: Tensor, parametrization: str) -> None:
        self.pred = pred
        self.parametrization = parametrization
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
    def __init__(self, scheduler: Scheduler, parametrization: str):
        super().__init__()
        self.scheduler = scheduler
        self.parametrization = parametrization
        assert self.parametrization in ["x0", "eps", "score"]
    
    def get_predictions(self, xt: Tensor, log_temp: Tensor) -> DDPMPredictions:
        tau = self.scheduler.tau_from_log_temp(log_temp).clip(0, 1)
        alpha_bar = cast_log_temp(self.scheduler.alpha_bar_from_tau(tau), xt)
        return DDPMPredictions(self(xt, tau), xt, alpha_bar, self.parametrization)

    @abstractmethod
    def forward(self, xt: Tensor, tau: Tensor) -> Tensor:
        pass
