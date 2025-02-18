from utils import get_dataset, get_data_generator
import torch
from diffusion import get_ddpm, DDPMTrainer
from config import Config
from config import with_config


@with_config()
def main(config: Config) -> None:
    dataset = get_dataset(config)
    data_generator = get_data_generator(dataset, config.ddpm_training.batch_size)
    ddpm = get_ddpm(config)

    trainer = DDPMTrainer(config, ddpm)
    trainer.train(data_generator, total_iters=config.ddpm_training.total_iters)

    torch.save(ddpm.state_dict(), config.ddpm_checkpoint_path)


if __name__ == "__main__":
    main()
