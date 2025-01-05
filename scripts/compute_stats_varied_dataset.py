from utils import get_data_tensor
import torch
from stats import compute_stats
import numpy as np
from base_config import BaseConfig
from config import with_config


@with_config()
def main(config: BaseConfig) -> None:
    temp = torch.logspace(
        config.varied_dataset_stats.min_temp,
        config.varied_dataset_stats.max_temp,
        config.varied_dataset_stats.n_temps,
    )
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
                config.varied_dataset_stats.n_repeats,
            )
            np.savez(filename, temp=temp, **stats) # type: ignore


if __name__ == "__main__":
    main()
