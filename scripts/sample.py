from diffusion import sample, DDPM, DDPMUnet, DDPMTrue
from utils import generate_dataset
import numpy as np
import torch
from typing import Any


cfg = {
    "ddpm_type": "true",
    # "ddpm_checkpoint": "checkpoints/pretrained_diffusion.pth",
    "kwargs": {
        "shape": (36, 1, 1, 100),
        "step_type": "ode",
        "track_ll": True,
    }
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
        kwargs["x_start"] =  torch.tensor(samples["states"][timestamp], dtype=torch.float32) # type: ignore
        if kwargs["track_ll"]: # type: ignore
            kwargs["init_ll"] = torch.tensor(samples["ll"][timestamp], dtype=torch.float32) # type: ignore
    
    results = sample(ddpm, **cfg["kwargs"]) # type: ignore
    filename = cfg.get("filename") or f"{cfg['kwargs']['step_type']}_samples_{cfg['ddpm_type']}" # type: ignore
    np.savez(f"results/{filename}.npz", **results)
