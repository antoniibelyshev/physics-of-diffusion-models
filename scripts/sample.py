import numpy as np
import torch

from diffusion import get_samples, get_ddpm, DDPM
from config import Config
from config import with_config


def get_and_save_samples(config: Config) -> None:
    ddpm = get_ddpm(config, pretrained = True)
    kwargs = config.sample.kwargs
    kwargs["n_steps"] = config.sample.n_steps
    kwargs["n_samples"] = config.sample.n_samples
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
def main(config: Config) -> None:
    get_and_save_samples(config)


if __name__ == "__main__":
    main()
