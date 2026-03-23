import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import Config
from utils import get_dataset, compute_metric_stats, get_data_generator, get_default_device
import os

def main():
    device = get_default_device()
    print(f"Using device: {device}")
    
    dataset_name = "cifar10"
    batch_size = 128
    
    # We want to check very low temperatures
    # T ranges from 1e-4 up to 10
    temps = torch.logspace(-4, 1, 30)
    
    config_dict = {
        "dataset_name": dataset_name,
        "diffusion": {"min_temp": 1e-4, "max_temp": 1e1},
        "entropy_schedule": {"extrapolate": True, "min_temp": 1e-4, "max_temp": 1e1},
        "ddpm": {"model_name": "true", "parametrization": "x0", "noise_schedule_type": "cosine"},
        "ddpm_training": {"batch_size": batch_size, "total_iters": 0, "learning_rate": 1e-4, "weight_decay": 0.0, "ema_decay": 0.999, "eval_steps": 1000, "warmup_steps": 0, "betas": [0.9, 0.999], "grad_clip": 1.0},
        "data_augmentation": {"use_augmentation": False, "horizontal_flip": False},
        "sample": {"n_steps": 10, "step_type": "ddim", "noise_schedule_type": "cosine", "n_samples": 100, "batch_size": batch_size, "precision": "full", "track_states": False},
        "forward_stats": {"n_samples": 1000, "batch_size": batch_size, "dataloader_batch_size": batch_size, "n_temps": 100},
        "empirical_stats": {"n_temps": 100, "n_steps_per_temp": 10, "batch_size": batch_size},
        "fid": {"n_steps": [10], "noise_schedule_type": ["cosine"], "min_temp": [1e-4], "train": True, "sample": True}
    }
    config = Config(**config_dict)
    
    print("Loading dataset...")
    dataset = get_dataset(config)
    data_gen = get_data_generator(dataset, batch_size=batch_size)
    
    print("Computing low-temperature metric stats...")
    # Use fewer samples for speed in diagnostic script
    stats = compute_metric_stats(DataLoader(dataset, batch_size=batch_size), data_gen, temps, n_samples=512)
    
    metric = stats["metric"].numpy()
    temp = stats["temp"].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.loglog(temp, metric, 'bo-', label='Empirical G(lambda)')
    
    # Let's try to fit an exponential decay for very small T?
    # Actually, let's just plot it and see.
    # Theoretical estimate: G(lambda) ~ (Delta^2 / 2T)^2 * e^(-Delta^2 / 2T)
    # where Delta^2 is the distance to the second nearest neighbor.
    
    # For a very rough visualization, let's pick a Delta^2
    delta_sq = 28.0 # Mean gap from previous script
    theoretical = (delta_sq / (2 * temp))**2 * np.exp(-delta_sq / (2 * temp))
    
    # Scale theoretical to match the magnitude for visualization
    # We find the peak or a point to match
    mask = temp < 1.0
    if mask.any():
        scale = metric[mask].max() / theoretical[mask].max() if theoretical[mask].max() > 0 else 1.0
        plt.loglog(temp, theoretical * scale, 'r--', label=f'Theoretical Asymptotic (Delta^2={delta_sq})')

    plt.xlabel("Temperature T")
    plt.ylabel("Metric G(lambda)")
    plt.title("Metric Tensor at Low Temperatures (CIFAR-10)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig("low_temp_metric.png")
    print("Saved plot to low_temp_metric.png")

if __name__ == "__main__":
    main()
