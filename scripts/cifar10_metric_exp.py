import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from diffusion.ddpm import ddpm_from_config
from diffusion.ddpm_sampling import DDPMSampler
from diffusion.scheduler import (
    LinearBetaScheduler,
    CosineScheduler,
    MetricScheduler,
)
from config import Config
from utils import (
    get_dataset, 
    get_data_generator, 
    get_compute_fid,
    compute_metric_stats,
    get_default_device
)

def main():
    # 1. Configuration for CIFAR-10 experiment
    device = get_default_device()
    print(f"Using device: {device}")
    
    dataset_name = "cifar10"
    n_samples = 50000
    batch_size = 128
    n_steps = 10

    # Standard CIFAR-10 temperature range from configs
    min_temp = 1e-4
    max_temp = 2.478e4

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
            "n_steps": n_steps,
            "step_type": "ddim",
            "noise_schedule_type": "cosine",
            "n_samples": n_samples,
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
            "n_steps": [n_steps],
            "noise_schedule_type": ["cosine"],
            "min_temp": [min_temp],
            "train": True,
            "sample": True
        }
    }
    config = Config(**config_dict)
    
    # 2. Dataset and Metric Stats Computation
    print("Loading CIFAR-10 dataset...")
    dataset = get_dataset(config)
    
    data_gen_iterator = get_data_generator(dataset, batch_size=batch_size)
    
    # CIFAR-10 metric stats path
    os.makedirs("stats", exist_ok=True)
    metric_stats_path = config.metric_stats_path
    
    if not os.path.exists(metric_stats_path):
        print("Computing empirical metric stats for CIFAR-10...")
        # Use a log-spaced temperature range for metric computation
        temp_range = torch.logspace(np.log10(min_temp), np.log10(max_temp), 100)
        # Use more samples for better estimation if resources allow
        metric_stats = compute_metric_stats(DataLoader(dataset, batch_size=batch_size), data_gen_iterator, temp_range, n_samples=2000)
        np.savez(metric_stats_path, **metric_stats)
        print(f"Saved metric stats to {metric_stats_path}")
    else:
        print(f"Loading existing metric stats from {metric_stats_path}")

    # 3. Initialize pre-trained CIFAR-10 model
    print("Loading pre-trained CIFAR-10 model...")
    # Using ddpm_from_config which handles model loading, processor settings and compilation
    model = ddpm_from_config(config, pretrained=True).to(device).eval()

    # 4. Define schedules
    print("Defining schedules...")
    linear_sch = LinearBetaScheduler(min_temp=min_temp, max_temp=max_temp)
    cosine_sch = CosineScheduler(min_temp=min_temp, max_temp=max_temp)
    metric_sch = MetricScheduler(metric_stats_path=metric_stats_path)
    
    schedules = {
        "Linear Beta": linear_sch,
        "Cosine": cosine_sch,
        "Metric": metric_sch
    }

    # 5. Plot schedules (Temperature vs Time)
    print("Generating schedule plots...")
    plt.figure(figsize=(10, 6))
    tau = torch.linspace(0, 1, 100)
    for name, sch in schedules.items():
        log_temp = sch.log_temp_from_tau(tau)
        plt.plot(tau.numpy(), np.exp(log_temp.numpy()), label=name)
    
    plt.yscale('log')
    plt.xlabel("tau (Time)")
    plt.ylabel("Temperature (1/SNR)")
    plt.title("CIFAR-10: Temperature vs Time for different schedules")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig("cifar10_schedules.png")
    print("Saved schedule comparison to cifar10_schedules.png")

    # 6. Evaluation (FID)
    print("Initializing FID computation (using InceptionV3)...")
    compute_fid_fn = get_compute_fid(config, device=device)
    
    results = {}
    
    for name, sch in schedules.items():
        print(f"\n--- Sampling with {name} schedule ({n_steps} steps, {n_samples} samples) ---")
        model.scheduler = sch
        sampler = DDPMSampler(
            ddpm=model,
            scheduler=sch,
            n_steps=n_steps,
            batch_size=batch_size,
            n_samples=n_samples,
            obj_size=config.dataset_config.obj_size,
            step_type="ddim",
            device=device
        )
        
        # Sampling
        samples_dict = sampler.sample()
        samples = samples_dict["x"] # (N, C, H, W)
        
        # FID
        print(f"Computing FID for {name} schedule...")
        fid_score = compute_fid_fn(samples)
        results[name] = fid_score
        print(f"FID ({name}): {fid_score:.4f}")
        
        # Save some samples for visual inspection
        from torchvision.utils import save_image
        os.makedirs("samples", exist_ok=True)
        grid_samples = samples[:64]
        # Normalize from [-1, 1] to [0, 1]
        grid_samples = (grid_samples + 1) / 2
        save_image(grid_samples, f"samples/cifar10_{name.lower().replace(' ', '_')}.png", nrow=8)

    # 7. Final Report
    print("\n" + "="*40)
    print(f"{'Schedule':<20} | {'FID (50k)':<10}")
    print("-" * 40)
    for name, fid in results.items():
        print(f"{name:<20} | {fid:<10.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
