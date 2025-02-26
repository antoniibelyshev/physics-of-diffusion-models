from tqdm import trange
import torch
from torch import Tensor
import numpy as np
from math import ceil
from typing import Iterable
from collections import defaultdict

from config import Config
from utils import norm_sqr, get_default_device, batch_jacobian, dict_map, append_dict
from .ddpm import DDPM, DDPMPredictions
from .diffusion_dynamic import NoiseScheduler, get_alpha_bar_from_log_temp


class SamplingCoeffs:
    def __init__(self, log_temp: Tensor, prev_log_temp: Tensor) -> None:
        alpha_bar = get_alpha_bar_from_log_temp(log_temp)
        prev_alpha_bar = get_alpha_bar_from_log_temp(prev_log_temp)
        alpha = alpha_bar / prev_alpha_bar
        beta = 1 - alpha

        self.ddpm_x0_coef = (prev_alpha_bar.sqrt() * beta) / (1 - alpha_bar)
        self.ddpm_xt_coef = (alpha.sqrt() * (1 - prev_alpha_bar)) / (1 - alpha_bar)
        self.ddpm_noise_coef = ((1 - prev_alpha_bar) / (1 - alpha_bar) * beta).sqrt()

        self.ddim_xt_coef = alpha.pow(-0.5)
        self.ddim_eps_coef = -self.ddim_xt_coef * beta / ((1 - alpha_bar).sqrt() + (1 - alpha_bar - beta).sqrt())


def ddpm_step(xt: Tensor, coeffs: SamplingCoeffs, predictions: DDPMPredictions) -> Tensor:
    x0 = predictions.x0
    noise = torch.randn_like(xt)
    return coeffs.ddpm_x0_coef * x0 + coeffs.ddpm_xt_coef * xt + coeffs.ddpm_noise_coef * noise  # type: ignore


def ddim_step(xt: Tensor, coeffs: SamplingCoeffs, predictions: DDPMPredictions) -> Tensor:
    eps = predictions.eps
    return coeffs.ddim_xt_coef * xt + coeffs.ddim_eps_coef * eps # type: ignore


STEPS_DICT = {
    "ddpm": ddpm_step,
    "ddim": ddim_step,
}


def is_ode_step(step_type: str) -> bool:
    match step_type:
        case "ddpm":
            return False
        case "ddim":
            return True
        case _:
            raise ValueError(f"Unknown step type {step_type}")


def get_range(*rng_args: int, verbose: bool) -> Iterable[int]:
    return (trange if verbose else range)(*rng_args)


class DDPMSampler:
    def __init__(self, config: Config) -> None:
        device = get_default_device()
        self.ddpm = DDPM.from_config(config, pretrained=True).to(device)
        self.ddpm.eval()
        self.device = device
        self.n_steps = config.sample.n_steps
        tau = torch.linspace(0, 1, self.n_steps, device=device).reshape(-1, 1)
        noise_scheduler = NoiseScheduler.from_config(config, noise_schedule_type=config.sample.noise_schedule_type)
        self.log_temp = noise_scheduler(tau)
        self.clean_log_temp = torch.full((1,), -torch.inf, device=device)
        self.n_samples = config.sample.n_samples
        self.batch_size = config.sample.batch_size
        self.n_repeats = ceil(self.n_samples / self.batch_size)
        self.outer_verbose = self.n_repeats >= 5
        self.inner_verbose = not self.outer_verbose
        try:
            self.step = STEPS_DICT[config.sample.step_type]
        except KeyError:
            raise KeyError(f"Unknown step type: {config.sample.step_type}")
        self.track_states = config.sample.track_states
        self.track_ll = config.sample.track_ll
        assert not self.track_ll or is_ode_step(config.sample.step_type), "LL tracking is possible only for ode steps"
        self.obj_size = config.data.obj_size

    def batch_sample(self, batch_size: int) -> dict[str, Tensor]:
        sample_shape = batch_size, *self.obj_size
        xt = torch.randn(*sample_shape, device=self.device)
        states: list[Tensor] | None = [] if self.track_states else None
        ll_lst: list[Tensor] | None = None
        if self.track_ll:
            ll_lst = [-0.5 * norm_sqr(xt.flatten(1).cpu() + np.log(2 * np.pi) * np.prod(self.obj_size))]

        for idx in get_range(len(self.log_temp) - 1, -1, -1, verbose=self.inner_verbose):
            if states is not None:
                states.append(xt.cpu())

            log_temp = self.log_temp[idx]
            prev_log_temp = self.log_temp[idx - 1] if idx > 0 else self.clean_log_temp
            coeffs = SamplingCoeffs(log_temp, prev_log_temp)

            if ll_lst is not None:
                def next_x(x: Tensor) -> Tensor:
                    return self.step(
                        x.view(sample_shape),
                        coeffs,
                        self.ddpm.get_predictions(xt, log_temp)
                    ).view(sample_shape[0], -1)

                ll_lst.append(ll_lst[-1] - torch.logdet(batch_jacobian(next_x, xt.view(sample_shape[0], -1))).cpu())

            with torch.no_grad():
                xt = self.step(xt, coeffs, self.ddpm.get_predictions(xt, log_temp))

        res = {"x": xt.cpu()}
        if states is not None:
            res["states"] = torch.stack(states[::-1], dim=1)
        if ll_lst is not None:
            res["ll"] = torch.stack(ll_lst[::-1], dim=1)
        return res

    def sample(self) -> dict[str, Tensor]:
        res: dict[str, list[Tensor]] = defaultdict(list)
        total_samples = 0
        for _ in get_range(self.n_repeats, verbose=self.outer_verbose):
            append_dict(res, self.batch_sample(min(self.batch_size, self.n_samples - total_samples)))
            total_samples += self.batch_size
        samples = dict_map(torch.cat, res)
        return samples


def get_samples(config: Config) -> dict[str, Tensor]:
    sampler = DDPMSampler(config)
    return sampler.sample()
