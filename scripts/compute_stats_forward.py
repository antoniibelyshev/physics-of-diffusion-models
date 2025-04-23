import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import compute_all_stats
from config import Config
from utils import get_dataset, get_data_generator, with_config, get_default_num_workers, dict_map


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    config.forward_stats.unbiased = False
    fwd_stats_cfg = config.forward_stats
    for dataset_name in config.available_datasets:
        print(dataset_name)
        config.dataset_name = dataset_name
        dataset = get_dataset(config)
        data_generator = get_data_generator(dataset, fwd_stats_cfg.batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=fwd_stats_cfg.dataloader_batch_size,
            shuffle=False,
            num_workers=get_default_num_workers()
        )
        min_temp, max_temp = config.dataset_config.temp_range
        temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), config.forward_stats.n_temps)
        if config.diffusion.min_temp < temp[-1]:
            temp = torch.cat((torch.full((1,), config.diffusion.min_temp), temp))
        stats = compute_all_stats(dataloader, data_generator, temp, fwd_stats_cfg.n_samples)
        np.savez(config.forward_stats_path, **stats)  # type: ignore


if __name__ == "__main__":
    main()
