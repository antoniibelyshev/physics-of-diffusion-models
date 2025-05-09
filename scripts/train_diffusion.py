from utils import get_dataset, get_data_generator
import torch
from diffusion import DDPM, DDPMTrainer
from config import Config
from utils import with_config


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    torch._dynamo.config.cache_size_limit = 64

    dataset = get_dataset(config)
    data_generator = get_data_generator(dataset, config.ddpm_training.batch_size)
    ddpm = DDPM.from_config(config)

    trainer = DDPMTrainer(config, ddpm)
    trainer.train(data_generator, total_iters=config.ddpm_training.total_iters)

    torch.save(ddpm.state_dict(), config.ddpm_checkpoint_path)


if __name__ == "__main__":
    main()
