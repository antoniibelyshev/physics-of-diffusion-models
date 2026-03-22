import numpy as np
import matplotlib.pyplot as plt

def get_analytical_metric(sigma, sigma0=1.0):
    """
    Computes the analytical metric G(sigma) derived in derivation.tex.
    G(sigma) = 2 * sigma0^2 * (sigma0^2 + 2 * sigma^2) / (sigma^2 * (sigma0^2 + sigma^2)^2)
    """
    numerator = 2 * sigma0**2 * (sigma0**2 + 2 * sigma**2)
    denominator = sigma**2 * (sigma0**2 + sigma**2)**2
    return numerator / denominator

def get_mc_metric(sigma, x_samples, n_y_samples=1000, sigma0=1.0):
    """
    Estimates the global metric G(sigma) using Monte Carlo samples from p(x).
    1. Sample y_i from the joint distribution p(y, sigma).
    2. Estimate the marginal score s(y_i, sigma) using Importance Sampling.
    3. G(sigma) = I(sigma) - Var_y[s(y_i, sigma)].
    """
    # 1. Sample y from the joint p(y, sigma) = N(y | 0, sigma0^2 + sigma^2)
    y_samples = np.random.normal(0, np.sqrt(sigma0**2 + sigma**2), size=n_y_samples)
    
    # Pre-calculate x_samples and sigma powers
    # x_samples: (K,)
    # y_samples: (N,)
    K = len(x_samples)
    N = n_y_samples
    
    # 2. Compute marginal scores for each y_i
    # s(y_i, sigma) = sum_k w_{ik} * (-(1/sigma) + (y_i - x_k)^2 / sigma^3)
    # w_{ik} = p(y_i | x_k, sigma) / sum_m p(y_i | x_m, sigma)
    
    # For numerical stability, use log-sum-exp trick for weights
    # log p(y_i | x_k, sigma) = -0.5 * log(2*pi*sigma^2) - 0.5 * (y_i - x_k)^2 / sigma^2
    # We can omit common terms in weights calculation
    
    # (N, K)
    y_grid, x_grid = np.meshgrid(y_samples, x_samples, indexing='ij')
    
    # (N, K)
    sq_diff = (y_grid - x_grid)**2
    log_weights = -0.5 * sq_diff / (sigma**2)
    
    # (N, 1)
    max_log_weights = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - max_log_weights)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # (N, K)
    individual_scores = -1.0/sigma + sq_diff / (sigma**3)
    
    # (N,)
    marginal_scores = np.sum(weights * individual_scores, axis=1)
    
    # 3. Global metric
    # I(sigma) = 2 / sigma^2
    fisher_noise = 2.0 / (sigma**2)
    var_marginal_score = np.var(marginal_scores)
    
    return fisher_noise - var_marginal_score

def main():
    sigma0 = 1.0
    sigmas = np.logspace(-1, 1, 20)
    
    # Prior samples p(x) = N(0, sigma0^2)
    K = 10000
    x_samples = np.random.normal(0, sigma0, size=K)
    
    analytical_results = get_analytical_metric(sigmas, sigma0)
    
    mc_results = []
    print("Computing Monte Carlo estimates...")
    for sigma in sigmas:
        mc_val = get_mc_metric(sigma, x_samples, n_y_samples=10000, sigma0=sigma0)
        mc_results.append(mc_val)
        print(f"sigma: {sigma:.4f}, Analytical: {get_analytical_metric(sigma, sigma0):.4f}, MC: {mc_val:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.loglog(sigmas, analytical_results, 'b-', label='Analytical')
    plt.loglog(sigmas, mc_results, 'ro', label='Monte Carlo Estimation')
    plt.xlabel('$\sigma$')
    plt.ylabel('Metric $\mathcal{G}(\sigma)$')
    plt.title('Comparison of Analytical and MC Estimated Metric Tensor ($p(x) = \mathcal{N}(0, 1)$)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_file = 'metric_comparison.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
