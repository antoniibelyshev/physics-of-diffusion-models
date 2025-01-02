from diffusion import sample, DDPM, DDPMUnet, DDPMTrue
from utils import generate_dataset
import numpy as np
import torch
from typing import Any


def get_cfg() -> dict[str, Any]:
    return {
        "ddpm_type": "unet",
        "ddpm_checkpoint": "checkpoints/pretrained_diffusion_100000.pth",
        "kwargs": {
            "shape": (1000, 1, 32, 32),
            "step_type": "sde",
            "track_ll": False,
        },
        "n_rep": 1,
    }


def main(cfg: dict[str, Any]):
    ddpm: DDPM
    if cfg["ddpm_type"] == "unet":
        ddpm = DDPMUnet()
        ddpm.load_state_dict(torch.load(cfg["ddpm_checkpoint"], map_location=torch.device('cpu'))) # type: ignore
    elif cfg["ddpm_type"] == "true":
        train_path = cfg.get("train_path") or f"results/{cfg.get('train_data_filename') or 'train_data'}.npy"
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
        if (samples_path := cfg.get("samples_path")) is not None:
            samples = np.load(samples_path)
        else:
            samples = np.load(f"results/{cfg['kwargs']['step_type']}_samples.npz")["states"][timestamp] # type: ignore
        kwargs["x_start"] =  torch.tensor(samples, dtype=torch.float32) # type: ignore
        if kwargs["track_ll"]: # type: ignore
            kwargs["init_ll"] = torch.tensor(samples["ll"][timestamp], dtype=torch.float32) # type: ignore

    results = sample(ddpm, **cfg["kwargs"]) # type: ignore
    for _ in range(cfg["n_rep"] - 1):
        for key, val in sample(ddpm, **cfg["kwargs"]).items(): # type: ignore
            results[key] = torch.cat([results[key], val], dim=0)

    abar = ddpm.dynamic.alpha_bar.cpu()
    temp = (1 - abar) / abar

    filename = cfg.get("filename") or f"{cfg['kwargs']['step_type']}_samples_{cfg['ddpm_type']}" # type: ignore
    np.savez(cfg.get("save_path") or f"results/{filename}.npz", temp = temp, **results)


if __name__ == "__main__":
    main(get_cfg())
