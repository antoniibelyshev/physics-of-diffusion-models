import numpy as np
import torch
from typing import Any

from diffusion import get_samples, get_ddpm, DDPM
from base_config import BaseConfig
from config import with_config


def get_and_save_samples(config: BaseConfig) -> None:
    ddpm = get_ddpm(config, pretrained = True)
    kwargs = config.sample.kwargs
    kwargs["shape"] = (config.sample.n_samples, *config.data.obj_size)
    save_path = config.samples_path
    if (timestamp := config.sample.timestamp) is not None:
        kwargs["timestamp"] = timestamp
        samples = np.load(config.samples_path)
        kwargs["x_start"] =  torch.tensor(samples["states"][timestamp], dtype=torch.float32)
        if kwargs["track_ll"]:
            kwargs["init_ll"] = torch.tensor(samples["ll"][timestamp], dtype=torch.float32)
        save_path = config.samples_from_timestamp_path
    
    samples = get_samples(ddpm, kwargs, config.sample.n_repeats)
    temp = ddpm.dynamic.temp.cpu().detach()
    np.savez(save_path, temp = temp, **samples)  # pyright: ignore


@with_config()
def main(config: BaseConfig) -> None:
    get_and_save_samples(config)


if __name__ == "__main__":
    main()
