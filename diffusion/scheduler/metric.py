import torch
import numpy as np
from .interpolated import InterpolatedDiscreteTimeScheduler


class MetricScheduler(InterpolatedDiscreteTimeScheduler):
    """
    Scheduler based on the empirical MC estimate of the metric tensor integrated along sigma.
    """
    def __init__(self, metric_stats_path: str):
        stats = np.load(metric_stats_path)
        log_temp = torch.from_numpy(stats["log_temp"])
        metric = torch.from_numpy(stats["metric"])
        
        # Sort for integration
        sort_idx = torch.argsort(log_temp)
        log_temp = log_temp[sort_idx]
        metric = metric[sort_idx]
        
        # Compute distance r(lambda) = integral sqrt(G(lambda')) d lambda'
        # lambda = log(sigma^2) = log_temp
        
        d_log_temp = log_temp[1:] - log_temp[:-1]
        sqrt_metric = torch.sqrt(torch.clamp(metric, min=0))
        
        # trapezoidal rule for integration
        # integral_a^b f(x) dx \approx (f(a) + f(b)) / 2 * (b - a)
        dr = 0.5 * (sqrt_metric[1:] + sqrt_metric[:-1]) * d_log_temp
        
        r = torch.cat([torch.zeros(1), torch.cumsum(dr, dim=0)])
        
        # Normalize r to [0, 1] for timestamps
        timestamps = r / r[-1]
        
        super().__init__(timestamps, log_temp)
