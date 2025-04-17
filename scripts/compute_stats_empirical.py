import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm, trange
from typing import Generator

from utils import with_config, get_dataset, get_data_tensor, get_data_generator
from diffusion import DDPM
from config import Config


def compute_entropy_derivative(
        data_generator: Generator[tuple[Tensor, ...], None, None],
        ddpm: DDPM,
        temp_range: Tensor,
        config: Config
) -> Tensor:
    d_entropy_d_log_temp = []
    for temp in tqdm(temp_range):
        errors = []
        log_temp = temp.log().cuda()[None]
        tau = ddpm.dynamic.noise_scheduler.get_tau(log_temp)
        for _ in range(config.empirical_stats.n_steps_per_temp):
            x0 = next(data_generator)[0]
            *_, xt = ddpm.dynamic(x0.cuda(), tau)
            x0_pred = ddpm.get_predictions(xt, log_temp).x0.cpu()
            errors.append((x0_pred - x0).square().sum(list(range(1, len(x0.shape)))))

        d_entropy_d_log_temp.append((0.5 * torch.cat(errors).mean() / temp).cpu())

    return torch.stack(d_entropy_d_log_temp)


@with_config(parse_args=(__name__ == "__main__"))
@torch.no_grad()
def main(config: Config) -> None:
    # dataset = get_dataset(config)
    dataset = TensorDataset(get_data_tensor(config))
    # dataset = TensorDataset(torch.from_numpy(np.load(config.samples_path)["x"]))
    data_generator = get_data_generator(dataset, batch_size=config.empirical_stats.batch_size)
    ddpm = DDPM.from_config(config, pretrained=True).cuda()

    temp_range = torch.logspace(
        np.log10(config.diffusion.min_temp),
        np.log10(config.diffusion.max_temp),
        config.empirical_stats.n_temps,
    ).flip(0)
    
    d_entropy_d_log_temp = compute_entropy_derivative(data_generator, ddpm, temp_range, config)
    np.savez(config.empirical_stats_path, temp=temp_range.numpy(), d_entropy_d_log_temp=d_entropy_d_log_temp.numpy())
    # np.savez("results/cifar10_empirical_stats.npz", temp=temp_range.numpy(), d_entropy_d_log_temp=torch.stack(d_entropy_d_log_temp).numpy())


if __name__ == "__main__":
    main()
