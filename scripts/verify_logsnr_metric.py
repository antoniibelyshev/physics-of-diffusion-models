import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import compute_metric_scalar

def get_analytical_metric_lambda(lambda_val, sigma0=1.0):
    """
    Computes analytical G(lambda) for p(x) = N(0, sigma0^2).
    sigma^2 = exp(lambda)
    G(sigma) = 2*sigma0^2*(sigma0^2 + 2*sigma^2) / (sigma^2 * (sigma0^2 + sigma^2)^2)
    G(lambda) = G(sigma) * (d sigma / d lambda)^2
    d sigma / d lambda = sigma / 2
    G(lambda) = G(sigma) * sigma^2 / 4
              = [2*sigma0^2*(sigma0^2 + 2*sigma^2) / (sigma^2 * (sigma0^2 + sigma^2)^2)] * sigma^2 / 4
              = 0.5 * sigma0^2 * (sigma0^2 + 2*sigma^2) / (sigma0^2 + sigma^2)^2
    """
    sigma_sq = np.exp(lambda_val)
    numerator = 0.5 * sigma0**2 * (sigma0**2 + 2 * sigma_sq)
    denominator = (sigma0**2 + sigma_sq)**2
    return numerator / denominator

def main():
    sigma0 = 1.0
    lambdas = np.linspace(-4, 4, 20)
    
    # Prior samples p(x) = N(0, sigma0^2)
    K = 10000
    D = 1
    x_samples = torch.randn((K, D)) * sigma0
    
    analytical_results = [get_analytical_metric_lambda(l, sigma0) for l in lambdas]
    
    mc_results = []
    print("Computing Monte Carlo estimates for G(lambda)...")
    for l in lambdas:
        mc_val = compute_metric_scalar(l, x_samples, n_y_samples=10000)
        mc_results.append(mc_val.item())
        print(f"lambda: {l:.4f}, Analytical: {get_analytical_metric_lambda(l, sigma0):.4f}, MC: {mc_val.item():.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, analytical_results, 'b-', label='Analytical')
    plt.plot(lambdas, mc_results, 'ro', label='Monte Carlo Estimation')
    plt.xlabel('$\lambda = \ln \sigma^2$')
    plt.ylabel('Metric $\mathcal{G}(\lambda)$')
    plt.title('Comparison of Analytical and MC Estimated Metric Tensor (log-SNR parameterization)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_file = 'metric_comparison_lambda.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
