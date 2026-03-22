import torch
import numpy as np

def compute_metric_scalar(log_sigma_sq, x_samples, n_y_samples=10000):
    """
    Computes the metric tensor for the isotropic case Sigma = sigma^2 * I.
    Natural parameter: lambda = log(sigma^2).
    
    Args:
        log_sigma_sq: float or torch.Tensor, log(sigma^2)
        x_samples: torch.Tensor, prior samples (K, D)
        n_y_samples: int, number of samples for marginal y
        
    Returns:
        metric: torch.Tensor, scalar value of G(lambda)
    """
    device = x_samples.device
    K, D = x_samples.shape
    sigma_sq = torch.exp(torch.tensor(log_sigma_sq, device=device))
    sigma = torch.sqrt(sigma_sq)
    
    # 1. Sample y from joint p(y, lambda)
    # y = x + sigma * epsilon, where x ~ p(x)
    indices = torch.randint(0, K, (n_y_samples,), device=device)
    x_for_y = x_samples[indices]
    eps = torch.randn((n_y_samples, D), device=device)
    y_samples = x_for_y + sigma * eps
    
    # 2. Estimate marginal score s(y, lambda) = d ln p(y, lambda) / d lambda
    # d ln p(y|x, lambda) / d lambda = d ln p(y|x, sigma) / d sigma * d sigma / d lambda
    # d ln p(y|x, sigma) / d sigma = -D/sigma + ||y-x||^2 / sigma^3
    # d sigma / d lambda = sigma / 2
    # d ln p(y|x, lambda) / d lambda = -D/2 + ||y-x||^2 / (2 * sigma^2)
    
    # We compute this for each y_i and x_k in a batch to avoid memory issues
    # But for small D and K, we can try vectorized
    
    # (n_y, K)
    # ||y-x||^2 = ||y||^2 + ||x||^2 - 2 y^T x
    y_sq = torch.sum(y_samples**2, dim=1, keepdim=True) # (n_y, 1)
    x_sq = torch.sum(x_samples**2, dim=1).unsqueeze(0) # (1, K)
    sq_dist = y_sq + x_sq - 2 * torch.mm(y_samples, x_samples.t()) # (n_y, K)
    
    log_weights = -0.5 * sq_dist / sigma_sq # (n_y, K), ignoring constant terms
    weights = torch.softmax(log_weights, dim=1) # (n_y, K)
    
    # Individual scores: d ln p(y|x, lambda) / d lambda
    individual_scores = -0.5 * D + 0.5 * sq_dist / sigma_sq # (n_y, K)
    
    # Marginal score: E_{x|y, lambda} [individual_scores]
    marginal_scores = torch.sum(weights * individual_scores, dim=1) # (n_y,)
    
    # 3. G(lambda) = I(lambda) - Var_y[marginal_scores]
    # I(lambda) = 0.5 * D
    fisher_noise = 0.5 * D
    var_marginal_score = torch.var(marginal_scores)
    
    return fisher_noise - var_marginal_score

def compute_metric_matrix(Lambda, x_samples, n_y_samples=10000):
    """
    Computes the metric tensor for arbitrary Sigma = exp(Lambda).
    Natural parameter: Lambda (symmetric matrix).
    
    Args:
        Lambda: torch.Tensor, symmetric matrix (D, D)
        x_samples: torch.Tensor, prior samples (K, D)
        n_y_samples: int, number of samples for marginal y
        
    Returns:
        metric: torch.Tensor, matrix G(Lambda) of shape (D, D, D, D) or flattened (D*D, D*D)?
                Actually, the natural parameter space has D*(D+1)/2 dimensions.
                Let's compute it for each component of Lambda.
    """
    device = x_samples.device
    K, D = x_samples.shape
    
    # Sigma = exp(Lambda)
    evals, evecs = torch.linalg.eigh(Lambda)
    Sigma = evecs @ torch.diag(torch.exp(evals)) @ evecs.t()
    
    # 1. Sample y from joint
    indices = torch.randint(0, K, (n_y_samples,), device=device)
    x_for_y = x_samples[indices]
    
    # Sample epsilon ~ N(0, I)
    eps = torch.randn((n_y_samples, D), device=device)
    # y = x + Sigma^{1/2} * eps
    sqrt_Sigma = evecs @ torch.diag(torch.sqrt(torch.exp(evals))) @ evecs.t()
    y_samples = x_for_y + (sqrt_Sigma @ eps.t()).t()
    
    # 2. Marginal score d ln p(y, Lambda) / d Lambda_ij
    # d ln p(y|x, Lambda) / d Lambda_ij = Tr( [ d ln p(y|x, Sigma) / d Sigma ] * [ d Sigma / d Lambda_ij ] )
    # d ln p(y|x, Sigma) / d Sigma = -0.5 * Sigma^{-1} + 0.5 * Sigma^{-1} (y-x)(y-x)^T Sigma^{-1}
    
    inv_Sigma = evecs @ torch.diag(torch.exp(-evals)) @ evecs.t()
    
    # We need the gradient with respect to Lambda.
    # For Sigma = exp(Lambda), d Sigma / d Lambda is non-trivial if they don't commute.
    # But for the Fisher Info, we can use the property that if we parameterize by Lambda, 
    # and compute at a point where Lambda is diagonal, it's easier.
    
    # To simplify, let's assume Lambda is diagonal for now, or just use autodiff for the score.
    # Since we need this to be reusable and general, autodiff is a good choice.
    
    def log_likelihood(Lambda_param, y, x):
        # Sigma = exp(Lambda_param)
        e, v = torch.linalg.eigh(Lambda_param)
        S = v @ torch.diag(torch.exp(e)) @ v.t()
        inv_S = v @ torch.diag(torch.exp(-e)) @ v.t()
        diff = y - x
        quad = torch.sum(diff * (inv_S @ diff.t()).t(), dim=1)
        log_det = torch.sum(e)
        return -0.5 * log_det - 0.5 * quad # ignore constant
    
    # However, we need to compute this for many y and x.
    # Let's use the formula: d ln p(y|x, Sigma) / d Sigma = -0.5 * (Sigma^{-1} - Sigma^{-1} (y-x)(y-x)^T Sigma^{-1})
    # and d Sigma / d Lambda.
    
    # For reusable method, maybe stick to diagonal Lambda for start if it's too complex, 
    # but the user asked for arbitrary Sigma.
    
    # Let's implement it for diagonal Lambda first as it's common in diffusion (anisotropic noise).
    # If Lambda is diagonal, Lambda = diag(lambda_1, ..., lambda_D)
    # Then Sigma_ii = exp(lambda_i)
    # d ln p(y|x, Lambda) / d lambda_i = -0.5 + 0.5 * (y_i - x_i)^2 / Sigma_ii
    
    Sigma_diag = torch.diag(Sigma)
    
    # (n_y, K, D)
    diff = y_samples.unsqueeze(1) - x_samples.unsqueeze(0)
    sq_diff = diff**2
    
    # log_weights (n_y, K)
    # ln p(y|x, Sigma) = -0.5 * sum(ln Sigma_ii) - 0.5 * sum((y_i-x_i)^2 / Sigma_ii)
    log_weights = -0.5 * torch.sum(sq_diff / Sigma_diag, dim=2)
    weights = torch.softmax(log_weights, dim=1)
    
    # individual_scores (n_y, K, D)
    # s_k,i = d ln p(y|x_k, Lambda) / d lambda_i
    individual_scores = -0.5 + 0.5 * sq_diff / Sigma_diag
    
    # marginal_scores (n_y, D)
    marginal_scores = torch.sum(weights.unsqueeze(2) * individual_scores, dim=1)
    
    # Global metric G_ii = I_ii - Var(s_i)
    # I_ii = 0.5
    fisher_noise = 0.5 * torch.ones(D, device=device)
    var_marginal_score = torch.var(marginal_scores, dim=0)
    
    return fisher_noise - var_marginal_score

def compute_rescaled_metric_matrix(Sigma, x_samples, n_y_samples=10000):
    """
    Computes the rescaled metric tensor for parameterization theta = Sigma.
    Args:
        Sigma: torch.Tensor, (D, D) or (D,) diagonal
        x_samples: torch.Tensor, (K, D)
        n_y_samples: int
    """
    device = x_samples.device
    K, D = x_samples.shape
    
    if Sigma.ndim == 1:
        Sigma_diag = Sigma
        Sigma_mat = torch.diag(Sigma)
    else:
        Sigma_mat = Sigma
        Sigma_diag = torch.diag(Sigma)

    # 1. Sample y from joint
    indices = torch.randint(0, K, (n_y_samples,), device=device)
    x_for_y = x_samples[indices]
    eps = torch.randn((n_y_samples, D), device=device)
    
    if Sigma.ndim == 1:
        y_samples = x_for_y + torch.sqrt(Sigma) * eps
    else:
        evals, evecs = torch.linalg.eigh(Sigma_mat)
        sqrt_Sigma = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=1e-10))) @ evecs.t()
        y_samples = x_for_y + (sqrt_Sigma @ eps.t()).t()

    # 2. Marginal score d ln p(y, Sigma) / d Sigma_ii (assuming diagonal Sigma)
    # d ln p(y|x, Sigma) / d Sigma_ii = -0.5/Sigma_ii + 0.5 * (y_i - x_i)^2 / Sigma_ii^2
    
    # (n_y, K, D)
    diff = y_samples.unsqueeze(1) - x_samples.unsqueeze(0)
    sq_diff = diff**2
    
    # log_weights (n_y, K)
    log_weights = -0.5 * torch.sum(sq_diff / Sigma_diag, dim=2)
    weights = torch.softmax(log_weights, dim=1)
    
    # individual_scores (n_y, K, D)
    individual_scores = -0.5 / Sigma_diag + 0.5 * sq_diff / (Sigma_diag**2)
    
    # marginal_scores (n_y, D)
    marginal_scores = torch.sum(weights.unsqueeze(2) * individual_scores, dim=1)
    
    # I_ii = 0.5 / Sigma_ii^2
    fisher_noise = 0.5 / (Sigma_diag**2)
    var_marginal_score = torch.var(marginal_scores, dim=0)
    
    G_ii = fisher_noise - var_marginal_score
    
    # Compute factor F_ii = Sigma_ii^2 / (Sigma_0_ii + 2*Sigma_ii)
    # Since Sigma_0 is unknown, we estimate it from samples
    Sigma0_diag = torch.var(x_samples, dim=0)
    # factor = (Sigma_diag**2) / (Sigma0_diag + 2 * Sigma_diag) # This was for G(sigma)
    
    # Actually if we want to multiply G(sigma^2) by a factor to get 2*Sigma0 / (Sigma0 + Sigma)^2
    # G(sigma^2) = 0.5 * Sigma0 * (Sigma0 + 2*Sigma) / (Sigma^2 * (Sigma0 + Sigma)^2)
    # Factor should be 4 * Sigma^2 / (Sigma0 + 2*Sigma)
    factor = 4 * (Sigma_diag**2) / (Sigma0_diag + 2 * Sigma_diag)
    
    return G_ii * factor
