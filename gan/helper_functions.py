from torch import Tensor, randn_like
from torch.nn.functional import softplus
from math import sqrt


def cross_entropy_loss(logits: Tensor, p: float) -> Tensor:
    return -p * softplus(-logits).mean() - (1 - p) * softplus(logits).mean()


def add_noise(x: Tensor, temp: float) -> Tensor:
    return (x + randn_like(x) * sqrt(temp)) / sqrt(1 + temp)
