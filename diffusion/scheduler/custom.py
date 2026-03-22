import torch
import numpy as np
from torch import from_numpy
from .interpolated import InterpolatedDiscreteTimeScheduler

class CustomScheduler(InterpolatedDiscreteTimeScheduler):
    def __init__(self, path: str):
        if path.endswith(".npz"):
            stats = np.load(path)
            log_temp = from_numpy(stats["log_temp"])
            if "timestamps" in stats:
                timestamps = from_numpy(stats["timestamps"])
            else:
                timestamps = torch.linspace(0, 1, len(log_temp))
        else:
            log_temp = torch.load(path)
            timestamps = torch.linspace(0, 1, len(log_temp))

        super().__init__(timestamps, log_temp)
