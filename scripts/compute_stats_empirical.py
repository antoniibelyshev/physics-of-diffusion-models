import torch
import numpy as np

from utils import with_config, get_dataset, get_data_generator
from diffusion import DDPM
from config import Config


@with_config()
def main(config: Config) -> None:
    dataset = get_dataset(config)
    data_generator = get_data_generator(dataset, batch_size=config.ddpm_training.batch_size)
    ddpm = DDPM.from_config(config, pretrained=True).cuda()

    temp_range = torch.logspace(
        np.log10(config.diffusion.min_temp),
        np.log10(config.diffusion.max_temp),
        config.empirical_stats.n_temps,
    )
    d_entropy_d_log_temp = []
    for temp in temp_range:
        errors = []
        tau = ddpm.dynamic.noise_scheduler.get_tau(temp.log()).cuda().repeat(config.ddpm_training.batch_size)
        for _ in range(config.empirical_stats.n_steps_per_temp):
            x0 = next(data_generator)[0].cuda()
            xt = ddpm.dynamic(x0, tau)
            errors.append((ddpm.get_predictions(xt, tau).x0 - x0).square().sum(list(range(1, len(x0.shape)))).cpu())

        d_entropy_d_log_temp.append(0.5 * torch.cat(errors).mean() / temp)

    np.savez(config.empirical_stats_path, temp=temp_range.numpy(), d_entropy_d_log_temp=d_entropy_d_log_temp)
