from tqdm import trange
import torch
from torch import Tensor
from math import ceil
from typing import Iterable, Optional
from collections import defaultdict

from config import Config
from utils import get_default_device, dict_map, append_dict, get_dataset, compute_dataset_average
from .ddpm import DDPM, DDPMPredictions
from .diffusion_dynamic import NoiseScheduler, get_alpha_bar_from_log_temp


class DDPMSampler:
    def __init__(self, config: Config, ddpm: Optional[DDPM] = None, min_temp: Optional[float] = None) -> None:
        device = get_default_device()

        self.ddpm = DDPM.from_config(config, pretrained=True).to(device) if ddpm is None else ddpm
        self.ddpm.eval()

        config.entropy_schedule.min_temp = min_temp or config.entropy_schedule.min_temp

        self.device = device
        max_log_temp = self.ddpm.dynamic.get_log_temp(torch.ones(1)).item()
        noise_scheduler = NoiseScheduler.from_config(
            config, noise_schedule_type=config.sample.noise_schedule_type
        )
        if config.sample.noise_schedule_type == "entropy":
            tau = torch.linspace(0, 1, config.sample.n_steps + 1, device=device)[:-1].unsqueeze(1)
        else:
            tau = torch.linspace(0, 1, config.sample.n_steps, device=device).unsqueeze(1)
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
        self.track_states = config.sample.track_states

        # self.x0_uniform = compute_dataset_average(config).to(device)

    def step(self, xt: Tensor, log_temp: Tensor, prev_log_temp: Tensor) -> Tensor:
        predictions = self.ddpm.get_predictions(xt, log_temp)
        alpha_bar = get_alpha_bar_from_log_temp(log_temp)
        prev_alpha_bar = get_alpha_bar_from_log_temp(prev_log_temp)

        if self.step_type == "ddpm":
            alpha = alpha_bar / prev_alpha_bar
            beta = 1 - alpha
            
            x0_coef = (prev_alpha_bar.sqrt() * beta) / (1 - alpha_bar)
            xt_coef = (alpha.sqrt() * (1 - prev_alpha_bar)) / (1 - alpha_bar)
            noise_coef = ((1 - prev_alpha_bar) / (1 - alpha_bar) * beta).sqrt()

            return predictions.x0 * x0_coef + xt * xt_coef + torch.randn_like(xt) * noise_coef
        
        elif self.step_type == "ddim":
            # x_{t-1} = sqrt(alpha_bar_prev) * x0 + sqrt(1 - alpha_bar_prev) * eps
            return prev_alpha_bar.sqrt() * predictions.x0 + (1 - prev_alpha_bar).sqrt() * predictions.eps

        raise ValueError(f"unknown step type: {self.step_type}")

    def batch_sample(self, batch_size: int) -> dict[str, Tensor]:
        sample_shape = batch_size, *self.obj_size
        xt = torch.randn(*sample_shape, device=self.device)
        
        states: Optional[list[Tensor]] = [] if self.track_states else None

        for idx in range(len(self.log_temp) - 1, -1, -1):
            log_temp = self.log_temp[idx]
            prev_log_temp = self.log_temp[idx - 1] if idx > 0 else self.clean_log_temp

            with torch.no_grad(), torch.autocast("cuda", dtype=self.sampling_dtype):
                xt = self.step(xt, log_temp, prev_log_temp)
                if states is not None:
                    states.append(xt.cpu())

        res = {"x": xt.cpu()}
        if states is not None:
            res["states"] = torch.stack(states[::-1])
        return res

    def sample(self) -> dict[str, Tensor]:
        res: dict[str, list[Tensor]] = defaultdict(list)
        total_samples = 0
        for _ in trange(self.n_repeats):
            append_dict(res, self.batch_sample(min(self.batch_size, self.n_samples - total_samples)))
            total_samples += self.batch_size
        samples = dict_map(torch.cat, res)
        return samples


def get_samples(config: Config, min_temp: Optional[float] = None) -> dict[str, Tensor]:
    sampler = DDPMSampler(config, min_temp = min_temp)
    return sampler.sample()

