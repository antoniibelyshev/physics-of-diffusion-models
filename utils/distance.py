from torch import Tensor
from typing import Optional


def compute_gram_matrix(x: Tensor, y: Tensor) -> Tensor:
    return x @ y.T


def norm_sqr(x: Tensor) -> Tensor:
    return (x.unsqueeze(1) @ x.unsqueeze(2)).squeeze(2).squeeze(1)


def compute_pw_dist_sqr(x: Tensor, y: Optional[Tensor] = None, to_cpu: bool = True) -> Tensor:
    flattened_x = x.view(x.shape[0], -1)
    flattened_y = y.view(y.shape[0], -1) if y is not None else flattened_x

    x_norm_sqr = norm_sqr(flattened_x)
    y_norm_sqr = norm_sqr(flattened_y)
    gram_matrix = compute_gram_matrix(flattened_x, flattened_y)

    pw_dist_sqr = x_norm_sqr.unsqueeze(1) - 2 * gram_matrix + y_norm_sqr
    return pw_dist_sqr.cpu() if to_cpu else pw_dist_sqr
