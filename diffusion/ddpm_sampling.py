from tqdm import trange
import torch
from torch import Tensor
from math import ceil
from typing import Iterable
from collections import defaultdict

from config import Config
from utils import get_default_device, dict_map, append_dict, get_dataset, compute_dataset_average
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
        max_log_temp = self.ddpm.dynamic.get_log_temp(torch.ones(1)).item()
        noise_scheduler = NoiseScheduler.from_config(
            config, noise_schedule_type=config.sample.noise_schedule_type
        )
        tau = torch.linspace(0, 1, config.sample.n_steps + 1, device=device)[:-1].unsqueeze(1)
        self.log_temp = noise_scheduler(tau).clip(max=max_log_temp)
        self.clean_log_temp = torch.full((1,), -torch.inf, device=device)
        self.n_samples = config.sample.n_samples
        self.batch_size = config.sample.batch_size
        self.n_repeats = ceil(self.n_samples / self.batch_size)
        # try:
        #     self.step = STEPS_DICT[config.sample.step_type]
        # except KeyError:
        #     raise KeyError(f"Unknown step type: {config.sample.step_type}")
        self.step_type = config.sample.step_type

        self.obj_size = config.dataset_config.obj_size
        self.sampling_dtype = torch.float16 if config.sample.precision == "half" else torch.float32

        # self.x0_uniform = compute_dataset_average(config).to(device)

    def step(self, xt: Tensor, log_temp: Tensor, prev_log_temp: Tensor):
        ddpm_predictions = self.ddpm.get_predictions(xt, log_temp)
        alpha_bar = get_alpha_bar_from_log_temp(log_temp)
        prev_alpha_bar = get_alpha_bar_from_log_temp(prev_log_temp)
        alpha = alpha_bar / prev_alpha_bar
        beta = 1 - alpha

        if self.step_type == "ddpm":
            x0_coef = (prev_alpha_bar.sqrt() * beta) / (1 - alpha_bar)
            xt_coef = (alpha.sqrt() * (1 - prev_alpha_bar)) / (1 - alpha_bar)
            noise_coef = ((1 - prev_alpha_bar) / (1 - alpha_bar) * beta).sqrt()

            return ddpm_predictions.x0 * x0_coef + xt * xt_coef + torch.randn_like(xt) * noise_coef

        elif self.step_type == "ddim" and self.ddpm.parametrization == "x0":
            if prev_alpha_bar.item() < 0.5:
                s = ((1 - prev_alpha_bar) / (1 - alpha_bar)).sqrt()
            else:
                s = ((1 / prev_alpha_bar - 1) / (1 / prev_alpha_bar - alpha)).sqrt()

            x0_coef = prev_alpha_bar.sqrt() * (1 - s)
            xt_coef = alpha.pow(-0.5) * s

            return ddpm_predictions.x0 * x0_coef + xt * xt_coef

        elif self.step_type == "ddim" and self.ddpm.parametrization == "eps":
            xt_coef = alpha.pow(-0.5)
            eps_coef = -xt_coef * beta / ((1 - alpha_bar).sqrt() + (1 - alpha_bar - beta).sqrt())
            return xt * xt_coef + ddpm_predictions.eps * eps_coef

        raise ValueError

    def batch_sample(self, batch_size: int) -> dict[str, Tensor]:
        sample_shape = batch_size, *self.obj_size
        xt = torch.randn(*sample_shape, device=self.device)

        for idx in range(len(self.log_temp) - 1, -1, -1):
            log_temp = self.log_temp[idx]
            prev_log_temp = self.log_temp[idx - 1] if idx > 0 else self.clean_log_temp
            coeffs = SamplingCoeffs(log_temp, prev_log_temp)

            with torch.no_grad(), torch.autocast("cuda", dtype=self.sampling_dtype):
                # if idx == len(self.log_temp) - 1:
                #     prev_alpha_bar = get_alpha_bar_from_log_temp(prev_log_temp)
                #     xt = self.x0_uniform * prev_alpha_bar.sqrt() + xt * (1 - prev_alpha_bar).sqrt()
                # else:
                xt = self.step(xt, coeffs, self.ddpm.get_predictions(xt, log_temp))

        res = {"x": xt.cpu()}
        return res

    def sample(self) -> dict[str, Tensor]:
        res: dict[str, list[Tensor]] = defaultdict(list)
        total_samples = 0
        for _ in trange(self.n_repeats):
            append_dict(res, self.batch_sample(min(self.batch_size, self.n_samples - total_samples)))
            total_samples += self.batch_size
        samples = dict_map(torch.cat, res)
        return samples


def get_samples(config: Config) -> dict[str, Tensor]:
    sampler = DDPMSampler(config)
    return sampler.sample()
