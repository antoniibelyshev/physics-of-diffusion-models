from diffusion import sample, DDPM, DDPMUnet, DDPMTrue
from free_energy import compute_free_energy_direct
from utils import generate_dataset, MnistDataset
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Any


cfg = {
    "ddpm_type": "unet",
    "ddpm_checkpoint": "checkpoints/pretrained_diffusion.pth",
    "kwargs": {
        "shape": (1000, 1, 32, 32),
        "step_type": "sde",
    },
    "n_rep": 100,
}


if __name__ == "__main__":
    ddpm: DDPM
    if cfg["ddpm_type"] == "unet":
        ddpm = DDPMUnet()
        ddpm.load_state_dict(torch.load(cfg["ddpm_checkpoint"], map_location=torch.device('cpu'))) # type: ignore
    elif cfg["ddpm_type"] == "true":
        train_path = f"results/{cfg.get('train_data_filename') or 'train_data'}.npy"
        try:
            train_data = torch.tensor(np.load(train_path), dtype=torch.float32)
        except FileNotFoundError:
            train_data = torch.tensor(generate_dataset("hypersphere", 100), dtype=torch.float32)
            np.save(train_path, train_data.numpy())
        ddpm = DDPMTrue(train_data.unsqueeze(1).unsqueeze(2))
    else:
        raise ValueError(f"Invalid ddpm_type: {cfg['ddpm_type']}")
    
    kwargs = cfg["kwargs"]
    if (timestamp := kwargs.get("timestamp")) is not None: # type: ignore
        samples = np.load(f"results/{cfg['kwargs']['step_type']}_samples.npz") # type: ignore
        kwargs["x_start"] = torch.tensor(samples["states"][timestamp], dtype=torch.float32) # type: ignore

    mnist = MnistDataset()
    train_data = torch.stack([mnist[i]["images"].view(-1) for i in range(len(mnist))], 0).cuda()

    alpha_bar = ddpm.dynamic.alpha_bar
    temp = (1 - alpha_bar) / alpha_bar

    free_energy_lst: list[Tensor] = []
    for _ in range(cfg["n_rep"]):
        x = sample(ddpm, **cfg["kwargs"])["states"] / alpha_bar.sqrt()[:, None, None, None] # type: ignore
        x = x.transpose(0, 1).cuda()
        batch_free_energy_lst: list[Tensor] = []
        for i, T in enumerate(tqdm(temp)):
            batch_free_energy_lst.append(compute_free_energy_direct(x[i].view(x.shape[1], -1).cuda(), train_data.cuda(), T).mean())

        free_energy_lst.append(torch.stack(batch_free_energy_lst, 0))

    filename = cfg.get("filename") or f"{cfg['kwargs']['step_type']}_free_energy_samples_{cfg['ddpm_type']}" # type: ignore
    np.save(f"results/{filename}.npy", torch.stack(free_energy_lst, 0).mean(0))
