import torch
from stats import compute_stats
import numpy as np
from utils import get_data_tensor
from base_config import BaseConfig
from config import with_config
from math import log10


@with_config()
def main(config: BaseConfig) -> None:
    y = get_data_tensor(config)
    temp = torch.logspace(log10(config.forward_stats.min_temp), log10(config.forward_stats.max_temp), config.forward_stats.n_temps)
    stats = compute_stats(y, temp, config.forward_stats.n_samples, config.forward_stats.n_repeats)
    np.savez(config.forward_stats_path, temp = temp, **stats) # type: ignore


if __name__ == "__main__":
    main()
