import torch
import numpy as np
import argparse
from utils import compute_stats

from config import Config, with_config
from utils import get_data_tensor


@with_config()
def main(config: Config, *, unbiased: bool) -> None:
    y = get_data_tensor(config)
    config.diffusion.noise_schedule_type = f"entropy{'_u' if unbiased else''}"
    min_temp, max_temp = config.diffusion.temp_range
    temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), config.forward_stats.n_temps)
    stats = compute_stats(y, temp, config.forward_stats.n_samples, config.forward_stats.batch_size, unbiased)
    path = config.forward_unbiased_stats_path if unbiased else config.forward_stats_path
    np.savez(path, **stats) # type: ignore


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
