from utils import get_dataset, get_data_tensor, get_data_generator, get_compute_fid
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from gan import GANGenerator, GANDiscriminator, GANTrainer, add_noise
from config import Config
from utils import with_config


class CustomDataset(Dataset[torch.Tensor]):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    generator = GANGenerator(config)
    discriminator = GANDiscriminator(config)
    trainer = GANTrainer(generator, discriminator, config, compute_fid=get_compute_fid(config))

    train_dataset = get_dataset(config)
    train_data_generator = get_data_generator(train_dataset, config.gan_training.batch_size)

    test_data = get_data_tensor(config, train=False)
    diffusion_data = torch.from_numpy(np.load(config.samples_path)["states"][:, 1])

    eval_data_loaders = {
        "Test": DataLoader(CustomDataset(add_noise(test_data, config.gan_training.temp)), batch_size=500),
        "Diffusion": DataLoader(CustomDataset(diffusion_data), batch_size=500)
    }

    trainer.train(
        train_data_generator,
        total_iters = config.gan_training.total_iters,
        test_data = test_data,
        eval_data_loaders = eval_data_loaders
    )

    torch.save(generator.state_dict(), f"checkpoints/generator_{config.gan_training.temp}_v1.pth")


if __name__ == "__main__":
    main()
