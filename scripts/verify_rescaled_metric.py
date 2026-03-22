import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import compute_rescaled_metric_matrix

def get_analytical_rescaled_metric(sigma_sq, sigma0_sq=1.0):
    """
    Computes analytical rescaled G_tilde(sigma^2) for p(x) = N(0, sigma0^2).
    G_tilde(sigma^2) = 2 * sigma0^2 / (sigma0^2 + sigma^2)^2
    """
    return 2 * sigma0_sq / (sigma0_sq + sigma_sq)**2

def main():
    sigma0_sq = 1.0
    sigma_sqs = np.logspace(-2, 2, 20)
    
    # Prior samples p(x) = N(0, sigma0^2)
    K = 10000
    D = 1
    x_samples = torch.randn((K, D)) * np.sqrt(sigma0_sq)
    
    analytical_results = [get_analytical_rescaled_metric(s, sigma0_sq) for s in sigma_sqs]
    
    mc_results = []
    print("Computing Monte Carlo estimates for rescaled metric...")
    for s in sigma_sqs:
        # compute_rescaled_metric_matrix expects Sigma as a tensor
        sigma_sq_tensor = torch.tensor([s], dtype=torch.float32)
        mc_val = compute_rescaled_metric_matrix(sigma_sq_tensor, x_samples, n_y_samples=10000)
        mc_results.append(mc_val.item())
        print(f"sigma_sq: {s:.4f}, Analytical: {get_analytical_rescaled_metric(s, sigma0_sq):.4f}, MC: {mc_val.item():.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.loglog(sigma_sqs, analytical_results, 'b-', label='Analytical')
    plt.loglog(sigma_sqs, mc_results, 'ro', label='Monte Carlo Estimation')
    plt.xlabel('$\sigma^2$')
    plt.ylabel('Rescaled Metric $\widetilde{\mathcal{G}}(\sigma^2)$')
    plt.title('Comparison of Analytical and MC Estimated Rescaled Metric Tensor')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_file = 'rescaled_metric_comparison.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
