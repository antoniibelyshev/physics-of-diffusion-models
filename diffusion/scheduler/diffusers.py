import torch
from torch import Tensor
from .scheduler import log_temp_from_alpha_bar
from .interpolated import InterpolatedDiscreteTimeScheduler

class FromDiffusersScheduler(InterpolatedDiscreteTimeScheduler):
    def __init__(self, alpha_bar: Tensor):
        log_temp = log_temp_from_alpha_bar(alpha_bar)

        super().__init__(torch.linspace(0, 1, len(log_temp)), log_temp)
