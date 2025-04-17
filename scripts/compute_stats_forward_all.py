import torch
import numpy as np

from utils import compute_all_stats, get_names_of_image_datasets
from config import Config
from utils import get_data_tensor, with_config


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    config.forward_stats.unbiased = False
    for dataset_name in get_names_of_image_datasets():
        print(dataset_name)
        config.data.dataset_name = dataset_name
        data = get_data_tensor(config)
        min_temp, max_temp = config.forward_stats_temp_range
        temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), config.forward_stats.n_temps)
        if config.diffusion.min_temp < temp[-1]:
            temp = torch.cat((torch.full((1,), config.diffusion.min_temp), temp))
        fwd_stats_cfg = config.forward_stats
        stats = compute_all_stats(data, temp, fwd_stats_cfg.n_samples, fwd_stats_cfg.batch_size)
        np.savez(config.forward_stats_path, **stats) # type: ignore


if __name__ == "__main__":
    main()
