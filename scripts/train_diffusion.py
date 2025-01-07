from utils import get_data_tensor, get_data_generator
import torch
from diffusion import get_ddpm, DDPMTrainer
from config import load_config


if __name__ == "__main__":
    config = load_config("config/config.json")

    data = get_data_tensor(config)
    data_generator = get_data_generator(data, config.data.batch_size)
    ddpm = get_ddpm(config)

    trainer = DDPMTrainer(config, ddpm)
    trainer.train(data_generator, total_iters=config.ddpm_training.total_iters)

    torch.save(ddpm.state_dict(), config.ddpm_checkpoint_path)
