from tqdm import trange
import torch
from torch import Tensor
from math import ceil
from typing import Iterable, Optional
from collections import defaultdict

from config import Config
from utils import get_default_device, dict_map, append_dict, get_dataset, compute_dataset_average
from .ddpm import DDPM, DDPMPredictions
from .diffusion_dynamic import NoiseScheduler, get_alpha_bar_from_log_temp, InterpolatedDiscreteTimeNoiseScheduler


class DDPMSampler:
    def __init__(
        self,
        ddpm: DDPM,
        noise_scheduler: NoiseScheduler,
        n_steps: int,
        batch_size: int,
        n_samples: int,
        obj_size: tuple[int, ...],
        step_type: str = "ddim",
        precision: str = "full",
        track_states: bool = False,
        device: str = get_default_device(),
        log_temp: Optional[Tensor] = None,
    ) -> None:
        self.ddpm = ddpm.to(device)
        self.ddpm.eval()

        self.device = device
        max_log_temp = self.ddpm.dynamic.get_log_temp(torch.ones(1)).item()
        
        if log_temp is not None:
            self.log_temp = log_temp.to(device).clip(max=max_log_temp)
        else:
            tau = torch.linspace(0, 1, n_steps + 1, device=device)[1:].unsqueeze(1)
            self.log_temp = noise_scheduler(tau).clip(max=max_log_temp)

        self.clean_log_temp = torch.full((1,), -torch.inf, device=device)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_repeats = ceil(self.n_samples / self.batch_size)
        self.step_type = step_type

        self.obj_size = obj_size
        self.sampling_dtype = torch.float16 if precision == "half" else torch.float32
        self.track_states = track_states

    @classmethod
    def from_config(cls, config: Config, ddpm: Optional[DDPM] = None, min_temp: Optional[float] = None) -> "DDPMSampler":
        device = get_default_device()
        ddpm = DDPM.from_config(config, pretrained=True).to(device) if ddpm is None else ddpm
        
        if min_temp is not None:
             config.entropy_schedule.min_temp = min_temp

        noise_scheduler = NoiseScheduler.from_config(
            config, 
            noise_schedule_type=config.sample.noise_schedule_type,
            noise_schedule_path=config.sample.noise_schedule_path,
        )

        log_temp = None
        if config.sample.noise_schedule_type == "custom" and isinstance(noise_scheduler, InterpolatedDiscreteTimeNoiseScheduler):
             log_temp = noise_scheduler.log_temp
        
        return cls(
            ddpm=ddpm,
            noise_scheduler=noise_scheduler,
            n_steps=config.sample.n_steps,
            batch_size=config.sample.batch_size,
            n_samples=config.sample.n_samples,
            obj_size=config.dataset_config.obj_size,
            step_type=config.sample.step_type,
            precision=config.sample.precision,
            track_states=config.sample.track_states,
            device=device,
            log_temp=log_temp,
        )

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

            # For the last step (prev_log_temp = -inf), noise_coef should be 0
            # because prev_alpha_bar = 1, so (1 - prev_alpha_bar) = 0.
            # However, depending on floating point precision, it might be safer to explicitly zero it.
            noise = torch.randn_like(xt) if prev_log_temp > -torch.inf else 0
            return predictions.x0 * x0_coef + xt * xt_coef + noise * noise_coef
        
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

            with torch.no_grad(), torch.autocast(self.device, dtype=self.sampling_dtype) if self.device != "cpu" else torch.no_grad():
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
    sampler = DDPMSampler.from_config(config, min_temp=min_temp)
    return sampler.sample()

