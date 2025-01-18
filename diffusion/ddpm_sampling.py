from tqdm import tqdm
from torch import Tensor
import torch
from .ddpm import DDPM
from typing import Optional, Callable, Any
import numpy as np
from torch.autograd.functional import jacobian

from utils import get_time_evenly_spaced


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x_: Tensor) -> Tensor:
        return func(x_).sum(dim=0)
        
    return jacobian(_func_sum, x).permute(1, 0, 2) # type: ignore


@torch.no_grad()
def sde_step(xt: Tensor, t: Tensor, ddpm: DDPM) -> Tensor:
    x0 = ddpm.get_predictions(xt, t).x0
    return ddpm.dynamic.sample_from_posterior_q(xt, x0, t)


def ode_step(xt: Tensor, t: Tensor, ddpm: DDPM) -> Tensor:
    beta = ddpm.dynamic.get_dynamic_params(t).beta
    score = ddpm.get_predictions(xt, t).score
    return xt + 0.5 * beta * (xt + score) # type: ignore


def step(xt: Tensor, t: Tensor, ddpm: DDPM, step_type: str = "sde") -> Tensor:
    match step_type:
        case "sde":
            return sde_step(xt, t, ddpm)
        case "ode":
            return ode_step(xt, t, ddpm)
        case _:
            raise ValueError("Invalid type")


def sample(
    ddpm: DDPM,
    num_steps: int,
    n_samples: Optional[int] = None,
    *,
    device: torch.device = DEVICE,
    step_type: str = "sde",
    track_ll: bool = False,
    x_start: Optional[Tensor] = None,
    init_ll: Optional[Tensor] = None,
    timestamp: int = 1,
) -> dict[str, Tensor]:
    if x_start is not None:
        shape = tuple(x_start.shape)
        xt = x_start.to(device)
    elif n_samples is not None:
        shape = (n_samples, *ddpm.dynamic.obj_size)
        xt = torch.randn(shape, device=device)
    else:
        raise ValueError("Either shape or x_start must be provided")

    ddpm.to(device)
    ddpm.eval()

    states: list[Tensor] = []
    ll_lst: list[Tensor] | None = None
    if track_ll:
        assert step_type == "ode", "Log-likelihood tracking is only supported for ODE sampling"
        if init_ll is None:
            init_ll = -0.5 * (xt.pow(2).sum(dim=tuple(range(1, len(shape) + 1))).cpu() + np.log(2 * np.pi) * np.prod(shape[1:]))
        ll_lst = [init_ll]

    for t in tqdm(get_time_evenly_spaced(num_steps, timestamp)):
        states.append(xt.cpu())
        if track_ll:
            def next_val(x: Tensor) -> Tensor:
                return step(x.view(shape), t, ddpm, step_type).view(shape[0], -1)

            assert ll_lst is not None, "ll_lst should not be None"
            ll_lst.append(ll_lst[-1] - torch.logdet(batch_jacobian(next_val, xt.view(shape[0], -1))).cpu())

        with torch.no_grad():
            xt = step(xt, t, ddpm, step_type)

    res = {"x": xt.cpu(), "states": torch.stack(states[::-1], dim=1)}
    if track_ll:
        assert ll_lst is not None, "ll_lst should not be None"
        res["ll"] = torch.stack(ll_lst[::-1], dim=1)

    return res


def get_samples(ddpm: DDPM, kwargs: dict[str, Any], n_repeats: int) -> dict[str, Tensor]:
    results = sample(ddpm, **kwargs)
    for _ in range(n_repeats - 1):
        for key, val in sample(ddpm, **kwargs).items():
            results[key] = torch.cat([results[key], val], dim=0)
    return results
