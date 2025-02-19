import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm # type: ignore
from typing import Callable

from config import Config
from .fid_models import get_feature_extractor
from .data import get_data_tensor
from .lenet import LeNet


ArrayT = NDArray[np.float32]


def extract_features_statistics(
        dataset: Tensor,
        feature_extractor: nn.Module,
        batch_size: int = 500,
        device: str = 'cuda'
) -> tuple[ArrayT, ArrayT]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # type: ignore
    all_features = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            features = feature_extractor(data)
            all_features.append(features)
    features = torch.cat(all_features, dim=0).cpu().numpy()
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def compute_fid(mu1: ArrayT, sigma1: ArrayT, mu2: ArrayT, sigma2: ArrayT) -> float:
    mean_diff_term = ((mu1 - mu2) ** 2).sum()
    cov_sqrt = sqrtm(sigma1 @ sigma2 + 1e-7)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    cov_diff_term = np.trace(sigma1 + sigma2 - 2 * cov_sqrt).mean()
    return mean_diff_term + cov_diff_term # type: ignore


def get_compute_fid(config: Config) -> Callable[[Tensor], float]:
    reference = get_data_tensor(config, train=config.fid.train)
    model = get_feature_extractor(config).cuda()
    mu_train, sigma_train = extract_features_statistics(reference, model)

    def _compute_fid(data: Tensor) -> float:
        mu_eval, sigma_eval = extract_features_statistics(data, model)
        return compute_fid(mu_train, sigma_train, mu_eval, sigma_eval)

    return _compute_fid
