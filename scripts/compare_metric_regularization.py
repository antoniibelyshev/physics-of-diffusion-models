import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from config import Config
from utils import get_dataset, compute_metric_stats, get_data_generator, get_default_device

def main():
    device = get_default_device()
    print(f"Using device: {device}")
    
    dataset_name = "cifar10"
    batch_size = 128
    
    # Standard CIFAR-10 temperature range
    min_temp = 1e-4
    max_temp = 1e6
    temps = torch.logspace(np.log10(min_temp), np.log10(max_temp), 100)

    config_dict = {
        "dataset_name": dataset_name,
        "diffusion": {"min_temp": min_temp, "max_temp": max_temp},
        "entropy_schedule": {"extrapolate": True, "min_temp": min_temp, "max_temp": max_temp},
        "ddpm": {"model_name": "true", "parametrization": "x0", "noise_schedule_type": "cosine"},
        "ddpm_training": {"batch_size": batch_size, "total_iters": 0, "learning_rate": 1e-4, "weight_decay": 0.0, "ema_decay": 0.999, "eval_steps": 1000, "warmup_steps": 0, "betas": [0.9, 0.999], "grad_clip": 1.0},
        "data_augmentation": {"use_augmentation": False, "horizontal_flip": False},
        "sample": {"n_steps": 10, "step_type": "ddim", "noise_schedule_type": "cosine", "n_samples": 100, "batch_size": batch_size, "precision": "full", "track_states": False},
        "forward_stats": {"n_samples": 1000, "batch_size": batch_size, "dataloader_batch_size": batch_size, "n_temps": 100},
        "empirical_stats": {"n_temps": 100, "n_steps_per_temp": 10, "batch_size": batch_size},
        "fid": {"n_steps": [10], "noise_schedule_type": ["cosine"], "min_temp": [min_temp], "train": True, "sample": True}
    }
    config = Config(**config_dict)
    
    print("Loading dataset...")
    dataset = get_dataset(config)
    data_gen = get_data_generator(dataset, batch_size=batch_size)
    
    # 1. Compute empirical metric stats without regularization
    print("Computing empirical metric stats (No regularization)...")
    stats_emp = compute_metric_stats(DataLoader(dataset, batch_size=batch_size), data_gen, temps, n_samples=512, regularize=False)
    
    # 2. Compute empirical metric stats WITH regularization
    print("Computing empirical metric stats (With regularization)...")
    stats_reg = compute_metric_stats(DataLoader(dataset, batch_size=batch_size), data_gen, temps, n_samples=512, regularize=True)
    
    # Save the regularized stats as the new standard
    np.savez(config.metric_stats_path, **stats_reg)
    print(f"Saved regularized metric stats to {config.metric_stats_path}")
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Metric G(lambda) vs Temperature
    plt.subplot(1, 2, 1)
    plt.loglog(temps.numpy(), stats_emp["metric"].numpy(), 'r--', label='Empirical (Sparse)')
    plt.loglog(temps.numpy(), stats_reg["metric"].numpy(), 'b-', label='Regularized (Manifold-aware)')
    plt.xlabel("Temperature (1/SNR)")
    plt.ylabel("Metric G(lambda)")
    plt.title("CIFAR-10: Metric Tensor Comparison")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Plot 2: Schedule (Time vs Temperature)
    plt.subplot(1, 2, 2)
    def get_schedule(metric_vals):
        log_temp = temps.log()
        d_log_temp = log_temp[1:] - log_temp[:-1]
        sqrt_metric = torch.sqrt(torch.clamp(metric_vals, min=0))
        dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
        r = torch.cat([torch.zeros(1), torch.cumsum(dr, dim=0)])
        tau = r / r[-1]
        return tau.numpy(), temps.numpy()

    tau_emp, temp_emp = get_schedule(stats_emp["metric"])
    tau_reg, temp_reg = get_schedule(stats_reg["metric"])
    
    plt.semilogy(tau_emp, temp_emp, 'r--', label='Schedule (Sparse)')
    plt.semilogy(tau_reg, temp_reg, 'b-', label='Schedule (Regularized)')
    plt.xlabel("tau (Time)")
    plt.ylabel("Temperature (1/SNR)")
    plt.title("Schedule Comparison")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("metric_regularization_comparison.png")
    print("Saved comparison plot to metric_regularization_comparison.png")

if __name__ == "__main__":
    main()
