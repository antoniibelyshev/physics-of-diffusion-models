import torch
import numpy as np
from torch.utils.data import DataLoader
from config import Config
from utils import get_dataset, compute_pw_dist_sqr, get_default_device
from tqdm import tqdm

def main():
    device = get_default_device()
    print(f"Using device: {device}")
    
    # 1. Load CIFAR-10 training set
    dataset_name = "cifar10"
    config_dict = {
        "dataset_name": dataset_name,
        "diffusion": {"min_temp": 1e-4, "max_temp": 1e8},
        "entropy_schedule": {"extrapolate": True, "min_temp": 1e-4, "max_temp": 1e8},
        "ddpm": {"model_name": "true", "parametrization": "x0", "noise_schedule_type": "cosine"},
        "ddpm_training": {"batch_size": 128, "total_iters": 0, "learning_rate": 1e-4, "weight_decay": 0.0, "ema_decay": 0.999, "eval_steps": 1000, "warmup_steps": 0, "betas": [0.9, 0.999], "grad_clip": 1.0},
        "data_augmentation": {"use_augmentation": False, "horizontal_flip": False},
        "sample": {"n_steps": 10, "step_type": "ddim", "noise_schedule_type": "cosine", "n_samples": 100, "batch_size": 128, "precision": "full", "track_states": False},
        "forward_stats": {"n_samples": 1000, "batch_size": 128, "dataloader_batch_size": 128, "n_temps": 100},
        "empirical_stats": {"n_temps": 100, "n_steps_per_temp": 10, "batch_size": 128},
        "fid": {"n_steps": [10], "noise_schedule_type": ["cosine"], "min_temp": [1e-4], "train": True, "sample": True}
    }
    config = Config(**config_dict)
    dataset = get_dataset(config)
    
    # 2. Analyze nearest-neighbor distances in CIFAR-10
    # We'll use a subset to avoid OOM or long computation (5000 samples)
    n_analyze = 5000
    dataloader = DataLoader(dataset, batch_size=n_analyze, shuffle=True)
    x0_batch = next(iter(dataloader))[0].to(device) # (N, 3, 32, 32)
    
    print(f"Analyzing distances for {n_analyze} CIFAR-10 samples...")
    # dist shape: (N, N)
    dist_sq = compute_pw_dist_sqr(x0_batch)
    
    # Fill diagonal with large values to ignore self-distance
    dist_sq.fill_diagonal_(1e10)
    
    # Nearest neighbor squared distances
    nn_dist_sq, _ = dist_sq.min(dim=1)
    
    # Second nearest neighbor squared distances
    dist_sq.scatter_(1, _.unsqueeze(1), 1e10)
    nn2_dist_sq, _ = dist_sq.min(dim=1)
    
    gap_sq = nn2_dist_sq - nn_dist_sq
    
    print(f"NN dist_sq - Mean: {nn_dist_sq.mean().item():.4f}, Min: {nn_dist_sq.min().item():.4f}, Max: {nn_dist_sq.max().item():.4f}")
    print(f"NN2 dist_sq - Mean: {nn2_dist_sq.mean().item():.4f}")
    print(f"Gap dist_sq - Mean: {gap_sq.mean().item():.4f}")
    
    print(f"Estimated Critical Temperature: {nn_dist_sq.mean().item():.4f}")

if __name__ == "__main__":
    main()
