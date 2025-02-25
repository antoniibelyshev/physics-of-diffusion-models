import torch
import numpy as np
from utils import compute_stats

from config import Config
from utils import get_data_tensor, with_config


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    y = get_data_tensor(config)
    min_temp, max_temp = config.diffusion.temp_range
    temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), config.forward_stats.n_temps)
    fwd_stats_cfg = config.forward_stats
    stats = compute_stats(y, temp, fwd_stats_cfg.n_samples, fwd_stats_cfg.batch_size, fwd_stats_cfg.unbiased)
    np.savez(config.forward_stats_path, **stats) # type: ignore


if __name__ == "__main__":
    main()
