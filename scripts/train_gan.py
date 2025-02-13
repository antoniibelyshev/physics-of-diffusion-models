from utils import get_data_tensor, get_data_generator, get_compute_fid
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from gan import GANGenerator, GANDiscriminator, GANTrainer, add_noise
from config import Config
from config import with_config


@with_config()
def main(config: Config) -> None:
    generator = GANGenerator(config)
    discriminator = GANDiscriminator(config)
    trainer = GANTrainer(generator, discriminator, config, compute_fid=get_compute_fid(config))

    train_data = get_data_tensor(config)
    train_data_generator = get_data_generator(train_data, config.data.batch_size)

    test_data = get_data_tensor(config, train=False)
    diffusion_data = torch.from_numpy(np.load(config.samples_path)["states"][:, 1])

    eval_data_loaders = {
        "Test": DataLoader(TensorDataset(add_noise(test_data, config.gan_training.temp)), batch_size=500),
        "Diffusion": DataLoader(TensorDataset(diffusion_data), batch_size=500)
    }

    trainer.train(
        train_data_generator,
        total_iters = config.gan_training.total_iters,
        test_data = test_data,
        eval_data_loaders = eval_data_loaders
    )

    torch.save(generator.state_dict(), "checkpoints/generator.pth")


if __name__ == "__main__":
    main()
