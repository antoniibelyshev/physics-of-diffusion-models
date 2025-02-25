from diffusion import get_samples
from config import Config
from utils import with_config
import numpy as np


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    np.savez(config.samples_path, **get_samples(config)) # type: ignore


if __name__ == "__main__":
    main()
