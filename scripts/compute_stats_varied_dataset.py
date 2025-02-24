from utils import get_data_tensor
import torch
from utils import compute_stats
import numpy as np
from math import log10

from config import Config, with_config


@with_config()
def main(config: Config) -> None:
    min_temp, max_temp = config.diffusion.temp_range
    temp = torch.logspace(log10(min_temp), log10(max_temp), config.varied_dataset_stats.n_temps)
    for dataset_name in config.varied_dataset_stats.dataset_names:
        config.data.dataset_name = dataset_name
        y = get_data_tensor(config)

        for data_fraction in config.varied_dataset_stats.sample_fractions:
            size = int(data_fraction * len(y))
            filename = f"results/{dataset_name}_{size}_forward_stats.npz"
            train_samples = y[np.random.choice(range(len(y)), size=size, replace=False)]
            stats = compute_stats(
                train_samples,
                temp,
                config.varied_dataset_stats.n_samples,
                config.varied_dataset_stats.batch_size,
                unbiased=False,
            )
            np.savez(filename, **stats) # type: ignore


if __name__ == "__main__":
    main()
