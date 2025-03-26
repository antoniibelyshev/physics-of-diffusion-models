import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm # type: ignore
from typing import Callable
from tqdm import tqdm

from config import Config
from .data import get_data_tensor
from .lenet import LeNet
from .data import to_uint8


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fid = FrechetInceptionDistance(feature=2048).cuda()

    def forward(self, x: Tensor) -> Tensor:
        return self.fid.inception(to_uint8(x)) # type: ignore


class LeNetFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lenet = LeNet(1024, 10)
        self.lenet.load_state_dict(torch.load("checkpoints/lenet_mnist.pth"))

    def forward(self, x: Tensor) -> Tensor:
        return self.lenet.features(x)


def get_feature_extractor(config: Config) -> nn.Module:
    match config.data.dataset_name:
        case "mnist":
            return LeNetFeatureExtractor()
        case _:
            return InceptionV3FeatureExtractor()


ArrayT = NDArray[np.float32]


def extract_features_statistics(
        dataset: Tensor,
        feature_extractor: nn.Module,
        batch_size: int = 100,
        device: str = 'cuda'
) -> tuple[ArrayT, ArrayT]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # type: ignore
    all_features = []
    with torch.no_grad():
        for data in tqdm(dataloader):
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
    cov_diff_term = np.trace(sigma1 + sigma2 - 2 * cov_sqrt)
    return mean_diff_term + cov_diff_term # type: ignore


def get_compute_fid(config: Config) -> Callable[[Tensor], float]:
    reference = get_data_tensor(config, train=config.fid.train)
    model = get_feature_extractor(config).cuda()
    mu_train, sigma_train = extract_features_statistics(reference, model)

    def _compute_fid(data: Tensor) -> float:
        mu_eval, sigma_eval = extract_features_statistics(data, model)
        return compute_fid(mu_train, sigma_train, mu_eval, sigma_eval)

    return _compute_fid
