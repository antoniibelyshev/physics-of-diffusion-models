import torch
import matplotlib.pyplot as plt
from diffusion.ddpm import DDPMTrue
from diffusion.ddpm_sampling import DDPMSampler
from config import Config
from config.dataset_configs import BaseDatasetConfig, DatasetRegistry
import numpy as np

# 1. Define a custom dataset config for our 1D GMM
@DatasetRegistry.register
class GMM1DConfig(BaseDatasetConfig):
    name: str = "gmm1d"
    channels: int = 1
    image_size: tuple[int, int] = (1, 1)
    min_temp: float = 1e-4
    max_temp: float = 1e1
    fid_samples: int = 100

def generate_gmm_data(n_samples=1_000_000):
    # 4 distinct Gaussians
    means = torch.tensor([-1.1, -0.9, 0.9, 1.1])
    stds = torch.tensor([0.01, 0.01, 0.01, 0.01])
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    # Sample component indices
    comp_indices = torch.multinomial(probs, n_samples, replacement=True)
    
    # Sample from the chosen Gaussians
    samples = torch.randn(n_samples) * stds[comp_indices] + means[comp_indices]
    
    # Reshape to (N, C, H, W) as expected by the framework
    return samples.view(n_samples, 1, 1, 1)

def compute_mmd(x, y, sigma=0.1):
    """
    Compute Maximum Mean Discrepancy (MMD) with an RBF kernel.
    MMD^2(P, Q) = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    """
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    
    def rbf_kernel(a, b, sigma):
        dist_sq = torch.cdist(a, b).pow(2)
        return torch.exp(-dist_sq / (2 * sigma**2))

    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    
    # Use biased estimator for simplicity as n is relatively small for samples
    mmd_sq = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd_sq.item()

def main():
    from utils import get_default_device
    device = torch.device(get_default_device())
    print(f"Using device: {device}")

    # 2. Generate the dataset
    print("Generating GMM dataset (1e6 samples)...")
    train_data = generate_gmm_data(1_000_000).to(device)

    # 3. Create a mock Config object
    # We'll use pydantic to construct it as defined in config.py
    from config import Config
    from config.config import DiffusionConfig, EntropyScheduleConfig, DDPMConfig, SampleConfig, DDPMTrainingConfig, DataAugmentationConfig, ForwardStatsConfig, EmpiricalStatsConfig, FIDConfig

    config_dict = {
        "dataset_name": "gmm1d",
        "diffusion": {
            "min_temp": 1e-4,
            "max_temp": 1e1
        },
        "entropy_schedule": {
            "extrapolate": False,
            "min_temp": 1e-4,
            "max_temp": 1e1
        },
        "ddpm": {
            "model_name": "true",
            "parametrization": "x0",
            "noise_schedule_type": "log_snr"
        },
        "ddpm_training": {
            "batch_size": 128,
            "total_iters": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0,
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
            "step_type": "ddpm",
            "noise_schedule_type": "log_snr",
            "n_samples": 100,
            "batch_size": 100,
            "precision": "full",
            "track_states": True
        },
        "forward_stats": {
            "n_samples": 100,
            "batch_size": 100,
            "dataloader_batch_size": 100,
            "n_temps": 100
        },
        "empirical_stats": {
            "n_temps": 100,
            "n_steps_per_temp": 10,
            "batch_size": 100
        },
        "fid": {
            "n_steps": [10],
            "noise_schedule_type": ["log_snr"],
            "min_temp": [1e-4],
            "train": True,
            "sample": True
        }
    }
    
    config = Config(**config_dict)

    # 4. Initialize DDPMTrue with our data
    print("Initializing DDPMTrue...")
    from diffusion.scheduler import Scheduler, LogSNRScheduler
    scheduler = LogSNRScheduler(min_temp=1e-4, max_temp=1e1)
    ddpm = DDPMTrue(
        scheduler=scheduler,
        parametrization="x0",
        train_data=train_data
    )
    ddpm.to(device)

    # 5. Initialize Sampler and Sample with Initial Schedule
    print("Sampling with Initial Schedule (linear log-SNR)...")
    initial_sampler = DDPMSampler(
        ddpm=ddpm,
        scheduler=LogSNRScheduler(min_temp=1e-4, max_temp=1e1),
        n_steps=10,
        batch_size=100,
        n_samples=100,
        obj_size=(1, 1, 1),
        step_type="ddpm",
        track_states=True,
        device=device
    )
    initial_results = initial_sampler.sample()
    initial_samples = initial_results["x"].view(-1).cpu().numpy()
    
    # 6. Initialize Sampler and Sample with Optimized Schedule
    print("Sampling with Optimized Schedule...")
    import os
    if os.path.exists("optimized_log_temp.pt"):
        optimized_log_temp = torch.load("optimized_log_temp.pt")
        optimized_sampler = DDPMSampler(
            ddpm=ddpm,
            scheduler=LogSNRScheduler(min_temp=1e-4, max_temp=1e1), # Not used if log_temp is provided
            n_steps=10,
            batch_size=100,
            n_samples=100,
            obj_size=(1, 1, 1),
            step_type="ddpm",
            track_states=True,
            device=device,
            log_temp=optimized_log_temp
        )
        optimized_results = optimized_sampler.sample()
        optimized_samples = optimized_results["x"].view(-1).cpu().numpy()
    else:
        print("Optimized schedule not found. Skipping.")
        optimized_samples = None

    # 7. Plot results
    plt.figure(figsize=(12, 8))
    
    true_samples_subset = train_data[:10000].view(-1).cpu()
    
    # Compute MMDs
    initial_mmd = compute_mmd(true_samples_subset, torch.from_numpy(initial_samples), sigma=0.1)
    print(f"Initial Schedule MMD (sigma=0.1): {initial_mmd:.6f}")
    
    if optimized_samples is not None:
        optimized_mmd = compute_mmd(true_samples_subset, torch.from_numpy(optimized_samples), sigma=0.1)
        print(f"Optimized Schedule MMD (sigma=0.1): {optimized_mmd:.6f}")
    
    # Plot histogram of original data
    plt.hist(true_samples_subset.numpy(), bins=500, density=True, alpha=0.3, label="True Distribution (subset)")
    
    # Plot histogram of initial samples
    plt.hist(initial_samples, bins=100, density=True, alpha=0.5, label=f"Initial Samples (MMD: {initial_mmd:.4f})")
    
    # Plot histogram of optimized samples
    if optimized_samples is not None:
        plt.hist(optimized_samples, bins=100, density=True, alpha=0.5, label=f"Optimized Samples (MMD: {optimized_mmd:.4f})")
    
    plt.title("GMM Sampling: Initial vs Optimized Schedule")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("gmm_comparison_optimized.png")
    print("Saved comparison plot to gmm_comparison_optimized.png")
    
    # If track_states was True, we can see the reverse process
    if optimized_samples is not None and "states" in optimized_results:
        states = optimized_results["states"].view(config.sample.n_steps, -1).cpu().numpy()
        plt.figure(figsize=(10, 6))
        for i in range(min(5, states.shape[1])):
            plt.plot(states[:, i], label=f"Sample {i}")
        plt.title("Reverse Diffusion Trajectories")
        plt.xlabel("Step")
        plt.ylabel("x")
        plt.legend()
        plt.savefig("gmm_trajectories.png")
        print("Saved trajectories plot to gmm_trajectories.png")

if __name__ == "__main__":
    main()
