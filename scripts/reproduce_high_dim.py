import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from diffusion.ddpm import DDPMTrue
from diffusion.ddpm_sampling import DDPMSampler
from diffusion.scheduler import LogSNRScheduler, EntropyScheduler, CosineScheduler, MetricScheduler
from config import Config
from config.dataset_configs import BaseDatasetConfig, DatasetRegistry
from utils.stats import compute_stats, compute_metric_stats
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Optional
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from torch import from_numpy

def generate_anisotropic_gmm_data(dim=100, n_components=5, n_samples=100000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Sample 5 cluster centers m1, ..., m5 from N(0, I)
    means = torch.randn(n_components, dim)
    
    # 2. Generate anisotropic covariances
    covs = []
    for i in range(n_components):
        A = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(A)
        s = torch.exp(-torch.linspace(0, 5, dim)) * 0.01
        Sigma = Q @ torch.diag(s) @ Q.T
        covs.append(Sigma)
        
    # 3. Sample from the mixture
    comp_indices = torch.multinomial(torch.ones(n_components)/n_components, n_samples, replacement=True)
    samples = torch.zeros(n_samples, dim)
    
    for i in range(n_components):
        mask = (comp_indices == i)
        count = mask.sum().item()
        if count > 0:
            L = torch.linalg.cholesky(covs[i] + 1e-8 * torch.eye(dim))
            z = torch.randn(count, dim)
            samples[mask] = means[i] + (L @ z.T).T
            
    return samples.view(n_samples, 1, dim, 1), means, covs

def compute_mmd_custom(x, y, sigma=1.0):
    # Use a subset for MMD to avoid OOM with 20k samples
    n_mmd = 5000
    x = x[:n_mmd].view(n_mmd, -1)
    y = y[:n_mmd].view(n_mmd, -1)
    def rbf_kernel(a, b, s):
        dist_sq = torch.cdist(a, b).pow(2)
        return torch.exp(-dist_sq / (2 * s**2))
    k_xx = rbf_kernel(x, x, sigma).mean()
    k_yy = rbf_kernel(y, y, sigma).mean()
    k_xy = rbf_kernel(x, y, sigma).mean()
    return (k_xx + k_yy - 2 * k_xy).item()

@DatasetRegistry.register
class AnisotropicGMMConfig(BaseDatasetConfig):
    name: str = "anisotropic_gmm_repro"
    channels: int = 1
    image_size: tuple[int, int] = (100, 1)
    min_temp: float = 1e-4
    max_temp: float = 1e2
    fid_samples: int = 100

def compute_kl_gmm(samples_induced, true_means, true_covs):
    """
    Compute KL(P_induced || P_true) using MC integration.
    P_induced is estimated by fitting a GMM to the generated samples.
    P_true is the ground truth mixture.
    
    KL = E_{x ~ P_induced} [log P_induced(x) - log P_true(x)]
    """
    dim = samples_induced.shape[1]
    n_components = len(true_means)
    
    # samples_induced is (N, dim)
    X = samples_induced.numpy()
    
    # Robust GMM fitting
    # We use init_params='random' to avoid unstable sklearn kmeans in high dimensions.
    # We use multiple initializations (n_init) to improve mode capture.
    # Increased reg_covar to 1e-4 for better numerical stability in 100D.
    gmm_induced = GaussianMixture(
        n_components=n_components, 
        covariance_type='full', 
        init_params='random', 
        n_init=3,
        random_state=42, 
        reg_covar=1e-4,
        max_iter=200
    )
    gmm_induced.fit(X)
    
    # Ensure weights are perfectly normalized for sampling
    weights = gmm_induced.weights_.astype(np.float64)
    weights /= weights.sum()
    weights[0] += 1.0 - weights.sum()
    gmm_induced.weights_ = weights
    
    # MC samples from the induced GMM
    x_mc_np, _ = gmm_induced.sample(50000)
    x_mc = torch.from_numpy(x_mc_np).float()
    
    # log P_induced(x_mc)
    log_p_induced = torch.from_numpy(gmm_induced.score_samples(x_mc_np))
    
    # log P_true(x_mc)
    log_probs_true_components = []
    for k in range(n_components):
        # Multivariate normal log-density
        cov = true_covs[k] + 1e-8 * torch.eye(dim)
        m = torch.distributions.MultivariateNormal(true_means[k], cov)
        log_probs_true_components.append(m.log_prob(x_mc))
    
    log_p_true = torch.logsumexp(torch.stack(log_probs_true_components), dim=0) - np.log(n_components)
    
    kl = torch.mean(log_p_induced - log_p_true)
    return kl.item()

def main():
    config_path = "config/high_dim_exp.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    dim = config.dataset_config.image_size[0]
    n_components = 5
    n_train_samples = 50000
    n_gen_samples = config.sample.n_samples
    device = "cpu"
    
    print(f"Generating data (dim={dim}, components={n_components}, n_samples={n_train_samples})...")
    train_data, true_means, true_covs = generate_anisotropic_gmm_data(dim=dim, n_components=n_components, n_samples=n_train_samples)
    train_data = train_data.to(device)

    os.makedirs("stats", exist_ok=True)
    dataloader = DataLoader(TensorDataset(train_data), batch_size=100)
    
    def data_gen():
        while True:
            for b in dataloader:
                yield b

    print("Computing forward stats for entropic schedule...")
    temp_range = torch.logspace(np.log10(config.diffusion.min_temp), np.log10(config.diffusion.max_temp), 200)
    stats = compute_stats(dataloader, data_gen(), temp_range, n_samples=100)
    np.savez(config.forward_stats_path, **stats)

    print("Computing metric stats for metric schedule...")
    metric_stats = compute_metric_stats(dataloader, data_gen(), temp_range, n_samples=1000)
    np.savez(config.metric_stats_path, **metric_stats)

    results_table = []

    # 1. Plot schedules
    print("Generating plots...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Temperature vs Time (Log-scale)
    plt.subplot(2, 2, 1)
    def get_temp_vals(sch):
        tau = torch.linspace(0, 1, 100)
        log_temp = sch.log_temp_from_tau(tau)
        return tau.numpy(), np.exp(log_temp.numpy())

    # We need all schedulers for plotting
    linear_scheduler = LogSNRScheduler(min_temp=config.diffusion.min_temp, max_temp=config.diffusion.max_temp)
    cosine_scheduler = CosineScheduler(min_temp=config.diffusion.min_temp, max_temp=config.diffusion.max_temp)
    entropy_scheduler_noext = EntropyScheduler(
        forward_stats_path=config.forward_stats_path,
        extrapolate=False,
        min_temp=config.entropy_schedule.min_temp,
        max_temp=config.entropy_schedule.max_temp
    )
    metric_scheduler = MetricScheduler(metric_stats_path=config.metric_stats_path)

    tau_lin, temp_lin = get_temp_vals(linear_scheduler)
    tau_cos, temp_cos = get_temp_vals(cosine_scheduler)
    tau_ent_no, temp_ent_no = get_temp_vals(entropy_scheduler_noext)
    tau_met, temp_met = get_temp_vals(metric_scheduler)

    plt.plot(tau_lin, temp_lin, label="Linear log-SNR", alpha=0.3)
    plt.plot(tau_cos, temp_cos, label="Cosine", linewidth=2)
    plt.plot(tau_ent_no, temp_ent_no, label="Entropic", alpha=0.3)
    plt.plot(tau_met, temp_met, label="Metric", linewidth=2)
    
    plt.yscale('log')
    plt.xlabel("tau (Time)")
    plt.ylabel("Temperature (1/SNR)")
    plt.title("Temperature vs Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Distance r(0, sigma) vs Temperature
    plt.subplot(2, 2, 2)
    metric_stats_data = np.load(config.metric_stats_path)
    log_temp_metric = torch.from_numpy(metric_stats_data["log_temp"])
    metric_vals = torch.from_numpy(metric_stats_data["metric"])
    
    # Sort for integration
    sort_idx = torch.argsort(log_temp_metric)
    log_temp_metric = log_temp_metric[sort_idx]
    metric_vals = metric_vals[sort_idx]
    
    d_log_temp = log_temp_metric[1:] - log_temp_metric[:-1]
    sqrt_metric = torch.sqrt(torch.clamp(metric_vals, min=0))
    dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
    r_vals_plot = torch.cat([torch.zeros(1), torch.cumsum(dr, dim=0)])
    
    plt.plot(np.exp(log_temp_metric.numpy()), r_vals_plot.numpy(), 'b-')
    plt.xscale('log')
    plt.xlabel("Temperature (1/SNR)")
    plt.ylabel("r(0, sigma)")
    plt.title("Distance r(0, sigma) vs Temperature")
    plt.grid(True, alpha=0.3)

    # Plot 3: Entropy vs Temperature (Log-scale)
    plt.subplot(2, 2, 3)
    stats_data = np.load(config.forward_stats_path)
    plt.plot(stats_data["temp"], stats_data["entropy"], 'k--', label="Forward Stats")
    plt.xscale('log')
    plt.xlabel("Temperature (1/SNR)")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Metric Tensor tail (G vs sigma^2)
    plt.subplot(2, 2, 4)
    # Highlight the decay
    plt.loglog(np.exp(log_temp_metric.numpy()), metric_vals.numpy(), 'b-', label='Empirical G(lambda)')
    # Add asymptotic line
    temp_tail = np.exp(log_temp_metric.numpy())
    data_centered = train_data.view(len(train_data), dim) - train_data.view(len(train_data), dim).mean(0)
    cov_trace = torch.trace(torch.cov(data_centered.T)).item()
    asymptotic = cov_trace / temp_tail
    plt.loglog(temp_tail, asymptotic, 'r--', label='Theoretical Tail (Tr(Sigma0)/sigma^2)')
    plt.xlabel("Temperature (sigma^2)")
    plt.ylabel("Metric G(lambda)")
    plt.title("Metric Asymptotic Behavior")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig("high_dim_plots.png")
    print("\nSaved comparison plots to high_dim_plots.png")

    # 1. Cosine sampling
    print("Sampling with Cosine schedule...")
    ddpm_cosine = DDPMTrue(scheduler=cosine_scheduler, parametrization="x0", train_data=train_data)
    sampler_cosine = DDPMSampler(
        ddpm=ddpm_cosine,
        scheduler=cosine_scheduler,
        n_steps=config.sample.n_steps,
        batch_size=config.sample.batch_size,
        n_samples=n_gen_samples,
        obj_size=(1, dim, 1),
        step_type="ddpm",
        device=device
    )
    samples_cosine = sampler_cosine.sample()["x"].view(n_gen_samples, dim)

    # 2. Metric sampling
    print("Sampling with Metric schedule...")
    ddpm_metric = DDPMTrue(scheduler=metric_scheduler, parametrization="x0", train_data=train_data)
    sampler_metric = DDPMSampler(
        ddpm=ddpm_metric,
        scheduler=metric_scheduler,
        n_steps=config.sample.n_steps,
        batch_size=config.sample.batch_size,
        n_samples=n_gen_samples,
        obj_size=(1, dim, 1),
        step_type="ddpm",
        device=device
    )
    samples_metric = sampler_metric.sample()["x"].view(n_gen_samples, dim)

    # Evaluation
    print("\nEvaluating Results...")
    ref_data = train_data[torch.randint(0, n_train_samples, (n_gen_samples,))].view(n_gen_samples, dim)
    
    # Baseline: Sample from initial distribution
    print("Evaluating Baseline (sample from initial distribution)...")
    baseline_samples = train_data[torch.randint(0, n_train_samples, (n_gen_samples,))].view(n_gen_samples, dim)

    def evaluate(samples, name):
        mmd = compute_mmd_custom(ref_data, samples, sigma=np.sqrt(dim))
        kl = compute_kl_gmm(samples, true_means, true_covs)
        
        dists = torch.cdist(samples, true_means)
        assignments = dists.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=n_components).float() / len(samples)
        
        mse_list = []
        for i in range(n_components):
            comp_samples = samples[assignments == i]
            if len(comp_samples) > 0:
                mse = torch.mean(torch.sum((comp_samples - true_means[i])**2, dim=1))
                mse_list.append(mse.item())
            else:
                mse_list.append(float('nan'))
        
        return {
            "name": name,
            "mmd": mmd,
            "kl": kl,
            "mse": np.nanmean(mse_list),
            "counts": counts.tolist()
        }

    results = [
        evaluate(baseline_samples, "Baseline (True)"),
        evaluate(samples_cosine, "Cosine"),
        evaluate(samples_metric, "Metric")
    ]

    print("-" * 85)
    print(f"{'Schedule':<20} | {'MMD':<10} | {'KL':<10} | {'Avg MSE':<10} | {'Component Dist.'}")
    print("-" * 85)
    for r in results:
        dist_str = ", ".join([f"{x:.2f}" for x in r["counts"]])
        print(f"{r['name']:<20} | {r['mmd']:<10.6f} | {r['kl']:<10.4f} | {r['mse']:<10.4f} | [{dist_str}]")
    print("-" * 85)

if __name__ == "__main__":
    main()
