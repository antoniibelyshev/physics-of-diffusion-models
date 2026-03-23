import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--stats_path", type=str, default="stats/cifar10_metric.npz")
args = parser.parse_args()

stats = np.load(args.stats_path)
temp = stats["temp"]
metric = stats["metric"]
log_temp = stats["log_temp"]

# Sort
sort_idx = np.argsort(log_temp)
log_temp = log_temp[sort_idx]
metric = metric[sort_idx]
temp = temp[sort_idx]

# Compute distance r(lambda)
d_log_temp = log_temp[1:] - log_temp[:-1]
sqrt_metric = np.sqrt(np.maximum(metric, 0))
dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
r_vals = np.concatenate([[0], np.cumsum(dr)])

plt.figure(figsize=(10, 6))
plt.semilogx(temp, r_vals, 'r-')
plt.axvline(1e-2, color='k', linestyle='--', label='T=1e-2')
plt.xlabel("Temperature T")
plt.ylabel("Distance r(0, sigma)")
plt.title("CIFAR-10 Cumulative Distance (Regularized Prior)")
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.savefig("debug_cifar_distance.png")

# Check values
idx_1e2 = np.abs(temp - 1e-2).argmin()
print(f"At T={temp[idx_1e2]:.2e}, r={r_vals[idx_1e2]:.4f}")
print(f"Max r={r_vals[-1]:.4f}")
print(f"Ratio r(1e-2)/r_max = {r_vals[idx_1e2]/r_vals[-1]:.4f}")
