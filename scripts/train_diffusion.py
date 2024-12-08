from utils import MnistDataset, get_data_generator
import torch
from diffusion import DDPM, DDPMTrainer


if __name__ == "__main__":
    data = MnistDataset()
    data_generator = get_data_generator(data, 128)
    device = torch.device('cuda:0')

    ddpm_x0 = DDPM(parametrization="x0")

    trainer = DDPMTrainer(
        ddpm_x0,
        device,
    )

    trainer.train(data_generator)

    torch.save(ddpm_x0.state_dict(), "pretrained_diffusion.pth")
