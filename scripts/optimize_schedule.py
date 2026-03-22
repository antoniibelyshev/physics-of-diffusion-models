import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from diffusion.ddpm import DDPMTrue, DDPMPredictions
from diffusion.ddpm_sampling import DDPMSampler
from diffusion.scheduler import LogSNRScheduler, alpha_bar_from_log_temp, cast_log_temp
from config.dataset_configs import BaseDatasetConfig, DatasetRegistry
import numpy as np
from tqdm import tqdm

# 1. Define a custom dataset config
@DatasetRegistry.register
class GMM1DOptConfig(BaseDatasetConfig):
    name: str = "gmm1d_opt"
    channels: int = 1
    image_size: tuple[int, int] = (1, 1)
    min_temp: float = 1e-4
    max_temp: float = 1e1
    fid_samples: int = 100

def generate_gmm_data(n_samples=1_000_000):
    means = torch.tensor([-1.1, -0.9, 0.9, 1.1])
    stds = torch.tensor([0.01, 0.01, 0.01, 0.01])
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    comp_indices = torch.multinomial(probs, n_samples, replacement=True)
    samples = torch.randn(n_samples) * stds[comp_indices] + means[comp_indices]
    return samples.view(n_samples, 1, 1, 1)

def compute_mmd(x, y, sigmas=[0.01, 0.05, 0.1, 0.5]):
    """
    Multi-scale RBF kernel for more robust MMD.
    """
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    
    dist_xx = torch.cdist(x, x).pow(2)
    dist_yy = torch.cdist(y, y).pow(2)
    dist_xy = torch.cdist(x, y).pow(2)
    
    loss = 0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma**2 + 1e-8)
        k_xx = torch.exp(-dist_xx * gamma).mean()
        k_yy = torch.exp(-dist_yy * gamma).mean()
        k_xy = torch.exp(-dist_xy * gamma).mean()
        loss += (k_xx + k_yy - 2 * k_xy)
        
    return loss / len(sigmas)

class DifferentiableSampler:
    def __init__(self, ddpm, step_type="ddpm"):
        self.ddpm = ddpm
        self.step_type = step_type
        self.clean_log_temp = torch.tensor([-20.0], device=ddpm.train_data.device)

    def step(self, xt, log_temp, prev_log_temp):
        tau = self.ddpm.scheduler.tau_from_log_temp(log_temp).clip(0, 1)
        alpha_bar = cast_log_temp(self.ddpm.scheduler.alpha_bar_from_tau(tau), xt)
        
        x0_pred = self.ddpm(xt, tau)
        predictions = DDPMPredictions(x0_pred, xt, alpha_bar, self.ddpm.parametrization)
        
        prev_tau = self.ddpm.scheduler.tau_from_log_temp(prev_log_temp).clip(0, 1)
        prev_alpha_bar = cast_log_temp(self.ddpm.scheduler.alpha_bar_from_tau(prev_tau), xt)

        if self.step_type == "ddpm":
            eps = 1e-10
            alpha = (alpha_bar + eps) / (prev_alpha_bar + eps)
            beta = 1 - alpha
            
            x0_coef = (prev_alpha_bar.sqrt() * beta) / (1 - alpha_bar + eps)
            xt_coef = (alpha.sqrt() * (1 - prev_alpha_bar)) / (1 - alpha_bar + eps)
            noise_coef = ((1 - prev_alpha_bar) / (1 - alpha_bar + eps) * beta).clamp(min=0).sqrt()
            
            noise = torch.randn_like(xt) if prev_log_temp > -15.0 else 0
            return predictions.x0 * x0_coef + xt * xt_coef + noise * noise_coef
        
        elif self.step_type == "ddim":
            return prev_alpha_bar.sqrt() * predictions.x0 + (1 - prev_alpha_bar).sqrt() * predictions.eps

        raise ValueError(f"unknown step type: {self.step_type}")

    def sample(self, log_temp, batch_size, obj_size):
        device = log_temp.device
        xt = torch.randn(batch_size, *obj_size, device=device)
        for idx in range(len(log_temp) - 1, -1, -1):
            curr_t = log_temp[idx]
            prev_t = log_temp[idx - 1] if idx > 0 else self.clean_log_temp
            xt = self.step(xt, curr_t, prev_t)
        return xt

def main():
    from utils import get_default_device
    device = torch.device(get_default_device())
    print(f"Using device: {device}")

    n_steps = 10
    batch_size = 2 ** 10 # Larger batch for better MMD
    n_iters = 1000
    lr = 0.001
    
    print("Generating GMM dataset...")
    train_data = generate_gmm_data(100000).to(device)

    # Initialize DDPMTrue
    obj_size = (1, 1, 1)
    scheduler = LogSNRScheduler(min_temp=1e-4, max_temp=1e1)
    ddpm = DDPMTrue(
        scheduler=scheduler,
        parametrization="x0",
        train_data=train_data
    )
    ddpm.to(device)
    
    scheduler = LogSNRScheduler(min_temp=1e-4, max_temp=1e1).to(device)
    
    # Optimize log_temp at uniform tau steps instead of tau directly
    uniform_tau = torch.linspace(0.0, 1.0, n_steps + 1, device=device)[1:]
    # Initial log_temp from LogSNR scheduler for those uniform tau
    initial_log_temp = scheduler.log_temp_from_tau(uniform_tau).detach()
    log_temp_param = nn.Parameter(initial_log_temp)
    optimizer = optim.Adam([log_temp_param], lr=lr)
    
    diff_sampler = DifferentiableSampler(ddpm, step_type="ddim")
    
    history_mmd = []
    
    print(f"Optimizing schedule for {n_iters} iterations...")
    pbar = tqdm(range(n_iters))
    for i in pbar:
        optimizer.zero_grad()
        
        # Enforce monotonicity for log_temp (log_temp increases with tau)
        with torch.no_grad():
            log_temp_param.data = log_temp_param.data.sort().values
            # Also clamp to reasonable range based on scheduler
            log_temp_param.clamp_(scheduler.min_log_temp, scheduler.max_log_temp)

        generated_samples = diff_sampler.sample(log_temp_param, batch_size, (1, 1, 1))
        
        indices = torch.randint(0, len(train_data), (batch_size,))
        true_samples_batch = train_data[indices]
        
        loss = compute_mmd(true_samples_batch, generated_samples)
        
        if torch.isnan(loss):
            print("NaN loss detected!")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_temp_param], max_norm=1.0)
        optimizer.step()
        
        history_mmd.append(loss.item())
        pbar.set_postfix(mmd=f"{loss.item():.6f}")

    # Plot optimization history
    plt.figure(figsize=(10, 5))
    plt.plot(history_mmd)
    plt.title("MMD Optimization History (Multi-scale RBF)")
    plt.xlabel("Iteration")
    plt.ylabel("MMD")
    plt.savefig("optimization_history.png")
    
    # Save optimized schedule
    optimized_log_temp = log_temp_param.detach().cpu()
    torch.save(optimized_log_temp, "optimized_log_temp.pt")
    print(f"Optimized log_temp: {optimized_log_temp}")
    
    # Compare schedules
    plt.figure(figsize=(10, 5))
    plt.plot(initial_log_temp.cpu().numpy(), label="Initial (Linear Log-SNR)")
    plt.plot(optimized_log_temp.numpy(), label="Optimized")
    plt.title("Sampling Schedule (log_temp)")
    plt.xlabel("Step index")
    plt.ylabel("log_temp")
    plt.legend()
    plt.savefig("schedule_comparison.png")

if __name__ == "__main__":
    main()
