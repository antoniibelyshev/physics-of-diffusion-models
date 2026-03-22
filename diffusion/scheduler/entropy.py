import numpy as np
from torch import from_numpy
import torch
from utils import extrapolate_entropy
from .interpolated import InterpolatedDiscreteTimeScheduler

class EntropyScheduler(InterpolatedDiscreteTimeScheduler):
    def __init__(
        self,
        forward_stats_path: str,
        extrapolate: bool,
        min_temp: float,
        max_temp: float,
    ):
        stats = np.load(forward_stats_path)
        temp = from_numpy(stats["temp"])
        entropy = from_numpy(stats["entropy"])

        if extrapolate:
            temp, entropy = extrapolate_entropy(temp, entropy, min_temp)

            mask = temp <= max_temp

            temp = temp[mask]
            entropy = entropy[mask]

        timestamps = entropy - entropy.min()
        timestamps /= timestamps.max()

        super().__init__(timestamps, temp.log())
