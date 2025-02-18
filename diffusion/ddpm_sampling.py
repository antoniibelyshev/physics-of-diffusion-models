from tqdm import trange
import torch
from torch import Tensor, from_numpy
from torch.autograd.functional import jacobian
import numpy as np
from typing import Optional, Callable, Any

from config import Config
from utils import norm_sqr
from .ddpm import DDPM, get_ddpm
from .diffusion_dynamic import DynamicCoeffs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_jacobian(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    def _func_sum(x_: Tensor) -> Tensor:
        return func(x_).sum(dim=0)

    return jacobian(_func_sum, x).permute(1, 0, 2) # type: ignore


@torch.no_grad()
def sde_step(xt: Tensor, idx: int, tau: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs) -> Tensor:
    x0 = ddpm.get_predictions(xt, tau).x0
    posterior_x0_coef = dynamic_coeffs.posterior_x0_coef[idx]
    posterior_xt_coef = dynamic_coeffs.posterior_xt_coef[idx]
    posterior_sigma = dynamic_coeffs.posterior_sigma[idx]
    eps = torch.randn(x0.shape).to(x0.device)
    return posterior_x0_coef * x0 + posterior_xt_coef * xt + eps * posterior_sigma.sqrt() # type: ignore


def ode_step(xt: Tensor, idx: int, tau: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs) -> Tensor:
    beta = dynamic_coeffs.beta[idx]
    alpha = dynamic_coeffs.alpha[idx]
    score = ddpm.get_predictions(xt, tau).score
    return xt / alpha.sqrt() + 0.5 * beta * score # type: ignore


def dpm_step(xt: Tensor, idx: int, tau: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs) -> Tensor:
    xt_coef = dynamic_coeffs.dpm_xt_coef[idx]
    eps_coef = dynamic_coeffs.dpm_eps_coef[idx]
    return xt * xt_coef + ddpm.get_predictions(xt, tau).eps * eps_coef # type: ignore


def step(xt: Tensor, idx: int, tau: Tensor, ddpm: DDPM, dynamic_coeffs: DynamicCoeffs, step_type: str) -> Tensor:
    match step_type:
        case "sde":
            return sde_step(xt, idx, tau, ddpm, dynamic_coeffs)
        case "ode":
            return ode_step(xt, idx, tau, ddpm, dynamic_coeffs)
        case "dpm":
            return dpm_step(xt, idx, tau, ddpm, dynamic_coeffs)
        case _:
            raise ValueError(f"Invalid step type: {step_type}")


def sample(
        ddpm: DDPM,
        n_steps: int,
        n_samples: Optional[int] = None,
        *,
        device: torch.device = DEVICE,
        step_type: str = "sde",
        track_states: bool = False,
        track_ll: bool = False,
        x_start: Optional[Tensor] = None,
        init_ll: Optional[Tensor] = None,
        idx_start: Optional[int] = None,
        verbose: bool = True,
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

    states: list[Tensor] | None = [] if track_states else None
    ll_lst: list[Tensor] | None = None
    if track_ll:
        assert step_type == "ode", "Log-likelihood tracking is only supported for ODE sampling"
        if init_ll is None:
            init_ll = -0.5 * norm_sqr(xt.flatten(1).cpu() + np.log(2 * np.pi) * np.prod(shape[1:]))
        ll_lst = [init_ll]

    tau_grid = torch.linspace(0, 1, n_steps, device=device)
    dynamic_coeffs = DynamicCoeffs(ddpm.dynamic.get_temp(tau_grid))

    for idx in (trange if verbose else range)(idx_start or len(tau_grid) - 1, -1, -1):
        if states is not None:
            states.append(xt.cpu())

        tau = tau_grid[idx][None]

        if ll_lst is not None:
            def next_val(x: Tensor) -> Tensor:
                return step(x.view(shape), idx, tau, ddpm, dynamic_coeffs, step_type).view(shape[0], -1)

            ll_lst.append(ll_lst[-1] - torch.logdet(batch_jacobian(next_val, xt.view(shape[0], -1))).cpu())

        with torch.no_grad():
            xt = step(xt, idx, tau, ddpm, dynamic_coeffs, step_type)

    res = {"x": xt.cpu(), "temp": dynamic_coeffs.temp.cpu()}
    if states is not None:
        res["states"] = torch.stack(states[::-1], dim=1)
    if ll_lst is not None:
        res["ll"] = torch.stack(ll_lst[::-1], dim=1)
    return res


def get_samples(config: Config) -> dict[str, Tensor]:
    ddpm = get_ddpm(config, pretrained = True)
    kwargs: dict[str, Any] = {
        "n_steps": config.sample.n_steps,
        "n_samples": config.sample.n_samples,
        "device": DEVICE,
        "step_type": config.sample.step_type,
        "track_states": config.sample.track_states,
        "track_ll": config.sample.track_ll,
    }
    if (idx_start := config.sample.idx_start) != -1:
        kwargs["idx_start"] = idx_start
        samples = np.load(config.samples_path)
        kwargs["x_start"] =  from_numpy(samples["states"][idx_start])
        if kwargs["track_ll"]:
            kwargs["init_ll"] = from_numpy(samples["ll"][idx_start])

    verbose = config.sample.n_repeats > 10
    kwargs["verbose"] = not verbose
    samples = sample(ddpm, **kwargs)
    for _ in (trange if verbose else range)(config.sample.n_repeats - 1):
        for key, val in sample(ddpm, **kwargs).items():
            samples[key] = torch.cat([samples[key], val], dim=0) if key != "temp" else val
    return samples # type: ignore
