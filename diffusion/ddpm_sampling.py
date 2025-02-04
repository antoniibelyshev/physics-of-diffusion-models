from tqdm import trange
from torch import Tensor, from_numpy
import torch
from .ddpm import DDPM, get_ddpm
from .diffusion_utils import DynamicCoeffs
from config import Config
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
def sde_step(xt: Tensor, idx: int, t: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs) -> Tensor:
    x0 = ddpm.get_predictions(xt, t).x0
    posterior_x0_coef = dynamic_coeffs.posterior_x0_coef[idx]
    posterior_xt_coef = dynamic_coeffs.posterior_xt_coef[idx]
    posterior_sigma = dynamic_coeffs.posterior_sigma[idx]
    eps = torch.randn(x0.shape).to(x0.device)
    return posterior_x0_coef * x0 + posterior_xt_coef * xt + eps * posterior_sigma.sqrt() # type: ignore


def ode_step(xt: Tensor, idx: int, t: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs) -> Tensor:
    beta = dynamic_coeffs.beta[idx]
    score = ddpm.get_predictions(xt, t).score
    return xt + 0.5 * beta * (xt + score) # type: ignore


def step(xt: Tensor, idx: int, t: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs, step_type: str = "sde") -> Tensor:
    match step_type:
        case "sde":
            return sde_step(xt, idx, t, ddpm, dynamic_coeffs)
        case "ode":
            return ode_step(xt, idx, t, ddpm, dynamic_coeffs)
        case _:
            raise ValueError("Invalid type")


def sample(
        ddpm: DDPM,
        n_steps: int,
        n_samples: Optional[int] = None,
        *,
        device: torch.device = DEVICE,
        step_type: str = "sde",
        track_ll: bool = False,
        x_start: Optional[Tensor] = None,
        init_ll: Optional[Tensor] = None,
        idx_start: int | None = None,
        min_t: float = 1e-10,
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
            init_ll = -0.5 * (xt.pow(2).sum(dim=tuple(range(1, len(shape)))).cpu() + np.log(2 * np.pi) * np.prod(shape[1:]))
        ll_lst = [init_ll]

    t_grid = get_time_evenly_spaced(n_steps, min_t = min_t).to(device)
    dynamic_coeffs = DynamicCoeffs(ddpm.dynamic.get_temp(t_grid))

    for idx in trange(idx_start or len(t_grid) - 1, -1, -1):
        t = t_grid[idx][None]
        states.append(xt.cpu())
        if track_ll:
            def next_val(x: Tensor) -> Tensor:
                return step(x.view(shape), idx, t, ddpm, dynamic_coeffs, step_type).view(shape[0], -1)

            assert ll_lst is not None, "ll_lst should not be None"
            ll_lst.append(ll_lst[-1] - torch.logdet(batch_jacobian(next_val, xt.view(shape[0], -1))).cpu())

        with torch.no_grad():
            xt = step(xt, idx, t, ddpm, dynamic_coeffs, step_type)

    res = {"x": xt.cpu(), "states": torch.stack(states[::-1], dim=1), "temp": dynamic_coeffs.temp.cpu()}
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


def get_and_save_samples(config: Config, *, save: bool = True) -> dict[str, Tensor]:
    ddpm = get_ddpm(config, pretrained = True)
    kwargs = config.sample.kwargs
    kwargs["n_steps"] = config.sample.n_steps
    kwargs["n_samples"] = config.sample.n_samples
    kwargs["min_t"] = config.ddpm.min_t
    save_path = config.samples_path
    if (idx_start := config.sample.idx_start) is not None:
        kwargs["idx_start"] = idx_start
        samples = np.load(config.samples_path)
        kwargs["x_start"] =  from_numpy(samples["states"][idx_start])
        if kwargs["track_ll"]:
            kwargs["init_ll"] = from_numpy(samples["ll"][idx_start])
        save_path = config.samples_from_timestamp_path
    
    samples = get_samples(ddpm, kwargs, config.sample.n_repeats)
    if save:
        np.savez(save_path, **samples)  # pyright: ignore
    return samples # type: ignore
