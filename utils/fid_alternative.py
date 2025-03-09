import os
import torchvision.utils as vutils # type: ignore
from pytorch_fid import fid_score # type: ignore
import shutil
from torch import Tensor
from typing import Callable

from config import Config
from .data import get_data_tensor


def normalize(imgs: Tensor) -> Tensor:
    return (imgs + 1) / 2


def save_images(tensor: Tensor, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for i in range(tensor.shape[0]):
        vutils.save_image(tensor[i], os.path.join(directory, f"{i}.png"))


def compute_fid(real_dir: str, fake_dir: str, batch_size: int = 50, device: str = "cuda") -> float:
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size, device, 2048) # type: ignore


def get_compute_fid(config: Config) -> Callable[[Tensor], float]:
    reference = normalize(get_data_tensor(config, train=config.fid.train))

    shutil.rmtree("./real_images", ignore_errors=True)
    save_images(reference, "./real_images")

    def _compute_fid(data: Tensor) -> float:
        save_images(normalize(data), "./fake_images")
        fid = compute_fid("./real_images", "./fake_images")
        shutil.rmtree("./fake_images", ignore_errors=True)
        return fid

    return _compute_fid
