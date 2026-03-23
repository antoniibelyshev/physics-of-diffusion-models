import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from utils import (
    get_dataset,
    get_data_generator,
    get_default_device,
    compute_model_metric_stats,
)
from diffusion import ddpm_from_config


from config.dataset_configs import DatasetRegistry


def build_config(dataset_name: str, min_temp: float, max_temp: float, batch_size: int) -> Config:
    # Use default temps from DatasetRegistry if not explicitly provided
    ds_config = DatasetRegistry.get(dataset_name)
    min_temp = min_temp if min_temp is not None else ds_config.min_temp
    max_temp = max_temp if max_temp is not None else ds_config.max_temp

    # Minimal config sufficient for dataset/model loading
    cfg = {
        "dataset_name": dataset_name,
        "diffusion": {"min_temp": min_temp, "max_temp": max_temp},
        "entropy_schedule": {"extrapolate": True, "min_temp": min_temp, "max_temp": max_temp},
        # Use diffusers UNet wrapper by default; change here if you want a different model
        "ddpm": {"model_name": "diffusers", "parametrization": "eps", "noise_schedule_type": "cosine"},
        "ddpm_training": {
            "batch_size": batch_size,
            "total_iters": 0,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "ema_decay": 0.999,
            "eval_steps": 1000,
            "warmup_steps": 0,
            "betas": [0.9, 0.999],
            "grad_clip": 1.0,
        },
        "data_augmentation": {"use_augmentation": False, "horizontal_flip": False},
        "sample": {
            "n_steps": 10,
            "step_type": "ddim",
            "noise_schedule_type": "cosine",
            "n_samples": 100,
            "batch_size": batch_size,
            "precision": "full",
            "track_states": False,
        },
        "forward_stats": {
            "n_samples": 1000,
            "batch_size": batch_size,
            "dataloader_batch_size": batch_size,
            "n_temps": 100,
        },
        "empirical_stats": {"n_temps": 100, "n_steps_per_temp": 10, "batch_size": batch_size},
        "fid": {
            "n_steps": [10],
            "noise_schedule_type": ["cosine"],
            "min_temp": [min_temp],
            "train": True,
            "sample": True,
        },
    }
    return Config(**cfg)


def compute_and_save_model_metric(
    dataset_name: str,
    min_temp: float,
    max_temp: float,
    n_temps: int,
    n_samples: int,
    batch_size: int,
    out_dir: str,
) -> tuple[str, str]:
    device = get_default_device()
    print(f"Using device: {device}")

    # 1) Build config and load dataset
    config = build_config(dataset_name, min_temp, max_temp, batch_size)

    print(f"Loading dataset: {dataset_name} ...")
    dataset = get_dataset(config)
    data_gen = get_data_generator(dataset, batch_size=batch_size)

    # 2) Load pretrained model
    print("Loading pretrained DDPM model ...")
    ddpm = ddpm_from_config(config, pretrained=True).to(device).eval()

    # 3) Temperature grid
    temp = torch.logspace(np.log10(min_temp), np.log10(max_temp), n_temps)

    # 4) Compute model-based metric stats
    print("Computing model-based metric stats ...")
    stats = compute_model_metric_stats(
        DataLoader(dataset, batch_size=batch_size), data_gen, ddpm, temp, n_samples=n_samples
    )

    os.makedirs(out_dir, exist_ok=True)
    metric_stats_path = os.path.join(out_dir, f"{dataset_name}_metric_model.npz")
    np.savez(metric_stats_path, **{k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in stats.items()})
    print(f"Saved model-based metric stats to {metric_stats_path}")

    # 5) Build schedule from metric: r(λ) = ∫ sqrt(G(λ')) dλ' (trapezoidal), τ = r / r[-1]
    log_temp = stats["log_temp"] if isinstance(stats["log_temp"], np.ndarray) else stats["log_temp"].cpu().numpy()
    metric = stats["metric"] if isinstance(stats["metric"], np.ndarray) else stats["metric"].cpu().numpy()

    sort_idx = np.argsort(log_temp)
    log_temp = log_temp[sort_idx]
    metric = metric[sort_idx]

    d_log_temp = np.diff(log_temp)
    sqrt_metric = np.sqrt(np.maximum(metric, 0))
    dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
    r_vals = np.concatenate([[0.0], np.cumsum(dr)])

    if r_vals[-1] <= 0:
        raise RuntimeError("Integrated distance r is non-positive; check metric values.")

    timestamps = r_vals / r_vals[-1]

    # Save a schedule file compatible with CustomScheduler (timestamps, log_temp)
    schedule_path = os.path.join(out_dir, f"{dataset_name}_metric_model_schedule.npz")
    np.savez(schedule_path, timestamps=timestamps, log_temp=log_temp)
    print(f"Saved metric schedule (timestamps/log_temp) to {schedule_path}")

    # 6) Visualization
    plt.figure(figsize=(14, 5))

    # Plot 1: Metric vs Temperature
    plt.subplot(1, 2, 1)
    temp_sorted = np.exp(log_temp)
    plt.loglog(temp_sorted, np.maximum(metric, 0), 'b-')
    plt.xlabel("Temperature (T = 1/SNR)")
    plt.ylabel("Metric G(λ)")
    plt.title(f"{dataset_name}: Model-based Metric vs Temperature")
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Plot 2: Distance r(0, σ) vs Temperature
    plt.subplot(1, 2, 2)
    plt.semilogx(temp_sorted, r_vals, 'r-')
    plt.axvline(1e-2, color='k', linestyle='--', label='T=1e-2')
    plt.xlabel("Temperature (T = 1/SNR)")
    plt.ylabel("Distance r(0, σ)")
    plt.title(f"{dataset_name}: Cumulative Distance vs Temperature")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plot_path = os.path.join(out_dir, f"{dataset_name}_model_metric_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

    return metric_stats_path, schedule_path


def main():
    parser = argparse.ArgumentParser(description="Compute model-based metric tensor and schedule using a trained DDPM")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (registered in DatasetRegistry)")
    parser.add_argument("--min_temp", type=float, default=None, help="Minimum temperature (T). Defaults to dataset config.")
    parser.add_argument("--max_temp", type=float, default=None, help="Maximum temperature (T). Defaults to dataset config.")
    parser.add_argument("--n_temps", type=int, default=100, help="Number of temperature points")
    parser.add_argument("--n_samples", type=int, default=2048, help="Number of x0 samples for metric estimate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loader")
    parser.add_argument("--out_dir", type=str, default="stats", help="Output directory for stats and schedule")

    args = parser.parse_args()

    compute_and_save_model_metric(
        dataset_name=args.dataset,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        n_temps=args.n_temps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
