from diffusion import get_samples
from config import Config
from config import with_config
import numpy as np


@with_config()
def main(config: Config) -> None:
    path = config.samples_path if config.sample.idx_start == -1 else config.samples_from_timestamp_path
    np.savez(path, **get_samples(config)) # type: ignore


if __name__ == "__main__":
    main()
