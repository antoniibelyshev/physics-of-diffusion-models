import torch
import numpy as np
import argparse
from stats import compute_stats, compute_stats_unbiased

from config import Config, with_config
from utils import get_data_tensor


@with_config()
def main(config: Config, *, unbiased: bool) -> None:
    y = get_data_tensor(config)
    config.diffusion.noise_schedule = f"entropy{'_u' if unbiased else''}"
    min_temp, max_temp = config.diffusion.temp_range
    temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), config.forward_stats.n_temps)
    compute_stats_func = compute_stats_unbiased if unbiased else compute_stats
    stats = compute_stats_func(y, temp, config.forward_stats.n_samples, config.forward_stats.n_repeats)
    np.savez(config.forward_stats_path, temp = temp, **stats) # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unbiased",
        type=bool,
        default=True,
        help="Whether to use an unbiased scheme. Default: True"
    )
    args = parser.parse_args()

    main(unbiased=args.unbiased)
