from tqdm.auto import trange
from tqdm import tqdm
from torch import Tensor
import torch
from .ddpm import DDPM
from typing import Any, Generator, Optional, Callable
import numpy as np
from torch.autograd.functional import jacobian


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x: Tensor) -> Tensor:
        return func(x).sum(dim=0)
        
    return jacobian(_func_sum, x).permute(1, 0, 2) # type: ignore


@torch.no_grad()
def sde_step(xt: Tensor, t: Tensor, ddpm: DDPM) -> Tensor:
    x0 = ddpm.get_val("x0", xt, t)
    return ddpm.dynamic.sample_from_posterior_q(xt, x0, t) - xt
    # beta = ddpm.dynamic.get_coef_from_time("beta", t)
    # s = ddpm.get_val("score", xt, t)
    # return 0.5 * beta * (xt + 2 * s) + beta.sqrt() * torch.randn_like(xt)


def ode_step(xt: Tensor, t: Tensor, ddpm: DDPM) -> Tensor:
    beta = ddpm.dynamic.get_coef_from_time("beta", t)
    s = ddpm.get_val("score", xt, t)
    return 0.5 * beta * (xt + s)


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
    shape: Optional[tuple[int, int, int, int]] = None,
    *,
    device: torch.device = DEVICE,
    step_type: str = "sde",
    track_ll: bool = False,
    n_substeps: int = 1,
    x_start: Optional[Tensor] = None,
    init_ll: Optional[Tensor] = None,
    timestamp: Optional[int] = None,
) -> dict[str, Tensor]:
    timestamp = timestamp or ddpm.dynamic.T
    dt = 1 / n_substeps

    if x_start is not None:
        shape_ = x_start.shape
        xt = x_start.to(device)
    elif shape is not None:
        shape_ = torch.Size(shape)
        xt = torch.randn(shape, device=device)
    else:
        raise ValueError("Either shape or x_start must be provided")

    ddpm.to(device)
    ddpm.eval()

    iterator: Generator[Tensor, None, None] = (
        torch.tensor([t] * shape_[0], device=device).long()
        for t in range(timestamp - 1, -1, -1)
    )

    states: list[Tensor] = []
    ll_lst: list[Tensor] | None = None
    if track_ll:
        assert step_type == "ode", "Log-likelihood tracking is only supported for ODE sampling"

        ll_lst = [init_ll if init_ll is not None else -0.5 * (xt.pow(2).sum(dim=(1, 2, 3)).cpu() + np.log(2 * np.pi) * np.prod(shape_[1:]))]

    for t in tqdm(iterator, total=timestamp):
        for _ in range(n_substeps):
            states.append(xt.cpu())
            if track_ll:
                def next_val(x: Tensor) -> Tensor:
                    return x + dt * step(x.view(shape_), t, ddpm, step_type).view(shape_[0], -1)

                assert ll_lst is not None, "ll_lst should not be None"
                ll_lst.append(ll_lst[-1] - torch.logdet(batch_jacobian(next_val, xt.view(shape_[0], -1))).cpu())

            with torch.no_grad():
                xt = xt + dt * step(xt, t, ddpm, step_type)

    res = {"x": xt.cpu(), "states": torch.stack(states[::-1], dim=1)}
    if track_ll:
        assert ll_lst is not None, "ll_lst should not be None"
        res["ll"] = torch.stack(ll_lst[::-1], dim=1)

    return res