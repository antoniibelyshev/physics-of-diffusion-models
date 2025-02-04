import numpy as np
import torch

from diffusion import get_and_save_samples
from config import Config
from config import with_config


@with_config()
def main(config: Config) -> None:
    if isinstance(config.sample.n_steps, int):
        get_and_save_samples(config)
    else:
        for n_steps in config.sample.n_steps:
            config.sample.n_steps = n_steps
            get_and_save_samples(config)


if __name__ == "__main__":
    main()
