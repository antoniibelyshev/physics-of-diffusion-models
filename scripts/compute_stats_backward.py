import torch
from stats import compute_stats_traj_batch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from base_config import BaseConfig
from config import with_config
from utils import get_data_tensor
# from .sample import get_samples
from diffusion import get_ddpm, get_samples


@with_config()
def main(config: BaseConfig) -> None:
    ddpm = get_ddpm(config, pretrained = True)
    kwargs = {
        "shape": (config.backward_stats.n_samples, *config.data.obj_size),
        "step_type": config.backward_stats.step_type,
    }
    samples = get_samples(ddpm, kwargs, n_repeats = config.backward_stats.n_repeats)
    temp = ddpm.dynamic.temp.cpu()
    x_sample = torch.tensor(samples["states"])
    x_batched = x_sample.reshape(-1, config.backward_stats.batch_size, *x_sample.shape[1:])
    y = get_data_tensor(config)

    stats_lst: list[dict[str, Tensor]] = []
    for x in tqdm(x_batched):
        stats_lst.append(compute_stats_traj_batch(x * (1 + temp).sqrt().view(-1, *[1] * len(y.shape[1:])), y, temp))

    stats = {key: torch.stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}
    np.savez("results/mnist_backward_stats.npz", temp = temp, **stats) # type: ignore


if __name__ == "__main__":
    main()
