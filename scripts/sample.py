from diffusion import get_samples
from config import Config
from config import with_config


@with_config()
def main(config: Config) -> None:
    get_samples(config)


if __name__ == "__main__":
    main()
