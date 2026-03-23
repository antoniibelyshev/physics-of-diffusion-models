import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm, trange
from typing import Generator

from utils import with_config, get_dataset, get_data_tensor, get_data_generator, get_default_device
from diffusion import DDPM
from config import Config


def compute_entropy_derivative(
        data_generator: Generator[tuple[Tensor, ...], None, None],
        ddpm: DDPM,
        temp_range: Tensor,
        config: Config,
        device: str = get_default_device(),
) -> Tensor:
    d_entropy_d_log_temp = []
    for temp in tqdm(temp_range):
        errors = []
        log_temp = temp.log().to(device)[None]
        tau = ddpm.scheduler.tau_from_log_temp(log_temp)
        for _ in range(config.empirical_stats.n_steps_per_temp):
            x0 = next(data_generator)[0]
            tau_val, eps, xt = ddpm.scheduler.add_noise(x0.to(device), tau)
            predictions = ddpm.get_predictions(xt, log_temp)
            x0_pred = predictions.x0.cpu()
            errors.append(torch.norm(x0_pred - x0).square() / len(x0))

        d_entropy_d_log_temp.append((0.5 * sum(errors) / len(errors) / temp).cpu())

    return torch.stack(d_entropy_d_log_temp)


@with_config(parse_args=(__name__ == "__main__"))
@torch.no_grad()
def main(config: Config) -> None:
    device = get_default_device()
    for dataset_name in config.available_datasets:
        print(dataset_name)
        config.dataset_name = dataset_name
        dataset = get_dataset(config)
        data_generator = get_data_generator(dataset, batch_size=config.empirical_stats.batch_size)
        from diffusion import ddpm_from_config
        ddpm = ddpm_from_config(config, pretrained=True).to(device)

        temp_range = torch.logspace(
            np.log10(config.diffusion.min_temp),
            np.log10(config.diffusion.max_temp),
            config.empirical_stats.n_temps,
        )
        
        d_entropy_d_log_temp = compute_entropy_derivative(data_generator, ddpm, temp_range, config, device=device)
        
        d_log_temp = temp_range[1].log() - temp_range[0].log()
        
        entropy = 0.5 * (d_entropy_d_log_temp[1:] + d_entropy_d_log_temp[:-1]).cumsum(0) * d_log_temp
        entropy -= entropy[-1].item()
        entropy = torch.nn.functional.pad(entropy, (0, 1), value=0)

        sigma = temp_range.sqrt()
        rescaled_entropy = 0.5 * (d_entropy_d_log_temp[1:] * sigma[1:] + d_entropy_d_log_temp[:-1] * sigma[:-1]).cumsum(0) * d_log_temp
        rescaled_entropy -= rescaled_entropy[-1].item()
        rescaled_entropy = torch.nn.functional.pad(rescaled_entropy, (0, 1), value=0)

        np.savez(
            config.empirical_stats_path,
            temp=temp_range.numpy(),
            entropy=entropy,
            rescaled_entropy=rescaled_entropy,
            d_entropy_d_log_temp=d_entropy_d_log_temp.numpy()
        )


if __name__ == "__main__":
    main()
