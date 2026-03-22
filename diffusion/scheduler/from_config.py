from typing import Optional
from config import Config
from .scheduler import Scheduler

def scheduler_from_config(config: Config, *, noise_schedule_type: Optional[str] = None, noise_schedule_path: Optional[str] = None) -> Scheduler:
    from .linear import LinearBetaScheduler
    from .cosine import CosineScheduler
    from .entropy import EntropyScheduler
    from .log_snr import LogSNRScheduler
    from .diffusers import FromDiffusersScheduler
    from .custom import CustomScheduler
    from .metric import MetricScheduler
    from utils import get_diffusers_pipeline, compute_dataset_average

    noise_schedule_type = noise_schedule_type or config.ddpm.noise_schedule_type

    if noise_schedule_type == "linear_beta":
        return LinearBetaScheduler(*config.diffusion.temp_range)
    elif noise_schedule_type == "cosine":
        return CosineScheduler(*config.diffusion.temp_range)
    elif noise_schedule_type == "entropy":
        return EntropyScheduler(
            config.forward_stats_path,
            config.entropy_schedule.extrapolate,
            config.entropy_schedule.min_temp,
            config.entropy_schedule.max_temp,
        )
    elif noise_schedule_type == "log_snr":
        return LogSNRScheduler(*config.diffusion.temp_range)
    elif noise_schedule_type == "metric":
        return MetricScheduler(config.metric_stats_path)
    elif noise_schedule_type == "diffusers":
        scheduler = get_diffusers_pipeline(config).scheduler  # type: ignore
        return FromDiffusersScheduler(scheduler.alphas_cumprod)
    elif noise_schedule_type == "custom":
        if noise_schedule_path is None:
            raise ValueError("noise_schedule_path must be provided for custom noise schedule")
        return CustomScheduler(noise_schedule_path)
    else:
        raise ValueError(f"Unknown schedule type: {noise_schedule_type}")
