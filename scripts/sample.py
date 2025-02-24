from diffusion import get_samples
from config import Config
from config import with_config
import numpy as np


@with_config()
def main(config: Config) -> None:
    np.savez(config.samples_path, **get_samples(config)) # type: ignore


if __name__ == "__main__":
    main()
