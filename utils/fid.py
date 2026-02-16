import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Callable
from tqdm import tqdm
from config import Config
from .data import get_dataset
from .lenet import LeNet
from .data import to_uint8, get_default_num_workers
from .utils import get_default_device

EPS = 1e-10


def sqrtm_torch(matrix: Tensor) -> Tensor:
    u, s, v = torch.svd(matrix + EPS * torch.eye(matrix.shape[0], device=matrix.device))
    return (u @ torch.diag(torch.sqrt(s)) @ v.T).real


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, device: str = get_default_device()) -> None:
        super().__init__()
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.fid.inception(to_uint8(x).to(self.device))  # type: ignore


class LeNetFeatureExtractor(nn.Module):
    def __init__(self, device: str = get_default_device()) -> None:
        super().__init__()
        self.device = device
        self.lenet = LeNet(1024, 10).to(device)
        self.lenet.load_state_dict(torch.load("checkpoints/lenet_mnist.pth", map_location=device))
        self.lenet.eval()

    def forward(self, x: Tensor) -> Tensor:
        return self.lenet.features(x.to(self.device))


def get_feature_extractor(config: Config, device: str = get_default_device()) -> nn.Module:
    match config.dataset_name:
        case "mnist":
            return LeNetFeatureExtractor(device=device)
        case _:
            return InceptionV3FeatureExtractor(device=device)


def extract_features_statistics(
        dataset: Dataset[tuple[Tensor, ...]],
        feature_extractor: nn.Module,
        batch_size: int = 100,
        device: str = get_default_device()
) -> tuple[Tensor, Tensor]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=get_default_num_workers())
    all_features = []
    with torch.no_grad():
        for data, *_ in tqdm(dataloader):
            data = data.to(device, non_blocking=True)
            features = feature_extractor(data)
            all_features.append(features)
    features = torch.cat(all_features, dim=0)
    mu = features.mean(dim=0)
    sigma = torch.cov(features.T)
    return mu, sigma


def compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> float:
    mean_diff_term = torch.sum((mu1 - mu2) ** 2)
    cov_sqrt = sqrtm_torch(sigma1 @ sigma2 + 1e-7 * torch.eye(sigma1.shape[0], device=sigma1.device))
    cov_diff_term = torch.trace(sigma1 + sigma2 - 2 * cov_sqrt)
    return (mean_diff_term + cov_diff_term).item()


def get_compute_fid(config: Config, device: str = get_default_device()) -> Callable[[Tensor], float]:
    reference = get_dataset(config, train=config.fid.train)
    model = get_feature_extractor(config, device=device).eval()
    mu_train, sigma_train = extract_features_statistics(reference, model, device=device)

    def _compute_fid(data: Tensor) -> float:
        mu_eval, sigma_eval = extract_features_statistics(TensorDataset(data), model, device=device)
        return compute_fid(mu_train, sigma_train, mu_eval, sigma_eval)

    return _compute_fid
