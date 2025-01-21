from torchvision.transforms import ToPILImage
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch
import os
from scipy.linalg import sqrtm
import numpy as np
from numpy.typing import NDArray


def save_tensors_as_images(tensor_dataset: Tensor, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(file_path)

    to_pil = ToPILImage()
    for i, img_tensor in enumerate(tensor_dataset):
        img = to_pil(img_tensor)
        img.save(os.path.join(output_dir, f"image_{i:05d}.png"))


def extract_features_statistics(dataset: Tensor, model: nn.Module, batch_size: int = 50, device: str = 'cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            features = model.features(data)
            all_features.append(features)
    features = torch.cat(all_features, dim=0).cpu().numpy()
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def compute_fid(
    mu1: NDArray[np.float32],
    sigma1: NDArray[np.float32],
    mu2: NDArray[np.float32],
    sigma2: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    mean_diff_term = ((mu1 - mu2) ** 2).sum()
    cov_sqrt = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    cov_diff_term = np.trace(sigma1 + sigma2 - 2 * cov_sqrt).mean()
    return mean_diff_term, cov_diff_term, mean_diff_term + cov_diff_term
