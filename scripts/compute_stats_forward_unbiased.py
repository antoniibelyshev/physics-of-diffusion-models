import torch
from stats import compute_stats_unbiased
import numpy as np
from utils import get_data_tensor
from config import Config, with_config
from math import log10


@with_config()
def main(config: Config) -> None:
    y = get_data_tensor(config)
    temp = torch.logspace(log10(config.forward_stats.min_temp), log10(config.forward_stats.max_temp), config.forward_stats.n_temps)
    stats = compute_stats_unbiased(y, temp, config.forward_stats.n_samples, config.forward_stats.n_repeats)
    np.savez(config.forward_unbiased_stats_path, temp = temp, **stats) # type: ignore


if __name__ == "__main__":
    main()
