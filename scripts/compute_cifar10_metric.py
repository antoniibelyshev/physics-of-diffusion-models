import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from config import Config
from utils import get_dataset, compute_metric_stats, compute_model_metric_stats, get_data_generator, get_default_device
from diffusion import ddpm_from_config

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", action="store_true", help="Use trained DDPM to estimate metric stats")
    args = parser.parse_args()

    device = get_default_device()
    print(f"Using device: {device}")
    
    dataset_name = "cifar10"
    batch_size = 128
    
    # Standard CIFAR-10 temperature range
    min_temp = 1e-4
    max_temp = 1e8
    temp_range = torch.logspace(np.log10(min_temp), np.log10(max_temp), 100)

    config_dict = {
        "dataset_name": dataset_name,
        "diffusion": {
            "min_temp": min_temp,
            "max_temp": max_temp
        },
        "entropy_schedule": {
            "extrapolate": True,
            "min_temp": min_temp,
            "max_temp": max_temp
        },
        "ddpm": {
            "model_name": "diffusers",
            "parametrization": "eps",
            "noise_schedule_type": "cosine"
        },
        "ddpm_training": {
            "batch_size": batch_size,
            "total_iters": 0,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "ema_decay": 0.999,
            "eval_steps": 1000,
            "warmup_steps": 0,
            "betas": [0.9, 0.999],
            "grad_clip": 1.0
        },
        "data_augmentation": {
            "use_augmentation": False,
            "horizontal_flip": False
        },
        "sample": {
            "n_steps": 10,
            "step_type": "ddim",
            "noise_schedule_type": "cosine",
            "n_samples": 100,
            "batch_size": batch_size,
            "precision": "full",
            "track_states": False
        },
        "forward_stats": {
            "n_samples": 1000,
            "batch_size": batch_size,
            "dataloader_batch_size": batch_size,
            "n_temps": 100
        },
        "empirical_stats": {
            "n_temps": 100,
            "n_steps_per_temp": 10,
            "batch_size": batch_size
        },
        "fid": {
            "n_steps": [10],
            "noise_schedule_type": ["cosine"],
            "min_temp": [min_temp],
            "train": True,
            "sample": True
        }
    }
    config = Config(**config_dict)
    
    print("Loading CIFAR-10 dataset...")
    dataset = get_dataset(config)
    data_gen_iterator = get_data_generator(dataset, batch_size=batch_size)
    
    os.makedirs("stats", exist_ok=True)
    metric_stats_path = config.metric_stats_path
    
    if args.use_model:
        print("Using pre-trained DDPM model for metric estimation...")
        ddpm = ddpm_from_config(config, pretrained=True).to(device).eval()
        metric_stats = compute_model_metric_stats(
            DataLoader(dataset, batch_size=batch_size),
            data_gen_iterator,
            ddpm,
            temp_range,
            n_samples=2048
        )
        # Suffix path to distinguish model-based stats
        metric_stats_path = metric_stats_path.replace(".npz", "_model.npz")
    else:
        print("Computing empirical (prior-based) metric stats for CIFAR-10...")
        # Enable adaptive k-NN regularization to avoid premature collapse in high-D
        # Tuned sigma_reg_scale to make the distance d(0, sigma) small near T=1e-2.
        # A smaller scale means it collapses faster.
        metric_stats = compute_metric_stats(
            DataLoader(dataset, batch_size=batch_size), 
            data_gen_iterator, 
            temp_range, 
            n_samples=2000,
            regularize=True,
            adaptive_knn=True,
            knn_k=5,
            sigma_reg_scale=0.0001,
        )
    
    # Save the stats
    np.savez(metric_stats_path, **metric_stats)
    print(f"Saved metric stats to {metric_stats_path}")
    
    # Visualization
    log_temp = metric_stats["log_temp"]
    metric = metric_stats["metric"]
    temp = metric_stats["temp"]
    
    # Sort for integration and plotting
    sort_idx = np.argsort(log_temp)
    log_temp_sorted = log_temp[sort_idx]
    metric_sorted = metric[sort_idx]
    temp_sorted = temp[sort_idx]
    
    # Compute distance r(lambda)
    d_log_temp = log_temp_sorted[1:] - log_temp_sorted[:-1]
    sqrt_metric = np.sqrt(np.maximum(metric_sorted, 0))
    dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
    r_vals = np.concatenate([[0], np.cumsum(dr)])
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Metric Tensor G(lambda) vs Temperature
    plt.subplot(1, 2, 1)
    plt.loglog(temp_sorted, metric_sorted, 'b-')
    plt.xlabel("Temperature (1/SNR)")
    plt.ylabel("Metric G(lambda)")
    plt.title("CIFAR-10: Metric Tensor vs Temperature")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Plot 2: Distance r(0, sigma) vs Temperature
    plt.subplot(1, 2, 2)
    plt.semilogx(temp_sorted, r_vals, 'r-')
    plt.xlabel("Temperature (1/SNR)")
    plt.ylabel("Distance r(0, sigma)")
    plt.title("CIFAR-10: Cumulative Distance vs Temperature")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plot_path = "cifar10_metric_plots.png"
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

if __name__ == "__main__":
    main()
