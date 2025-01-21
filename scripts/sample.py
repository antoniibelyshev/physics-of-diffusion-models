import numpy as np
import torch

from diffusion import get_and_save_samples
from config import Config
from config import with_config


@with_config()
def main(config: Config) -> None:
    get_and_save_samples(config)


if __name__ == "__main__":
    main()
