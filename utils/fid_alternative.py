import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_fid import fid_score
import shutil
from torch import Tensor
from typing import Callable

from config import Config
from .data import get_data_tensor


def normalize(imgs):
    return (imgs + 1) / 2


def save_images(tensor, directory):
    os.makedirs(directory, exist_ok=True)
    for i in range(tensor.shape[0]):
        vutils.save_image(tensor[i], os.path.join(directory, f"{i}.png"))


def compute_fid(real_dir, fake_dir, batch_size=50, device="cuda"):
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size, device, 2048)


def get_compute_fid(config: Config) -> Callable[[Tensor], float]:
    reference = get_data_tensor(config, train=config.fid.train)

    shutil.rmtree("./real_images", ignore_errors=True)
    save_images(reference, "./real_images")

    def _compute_fid(data: Tensor) -> float:
        save_images(data, "./fake_images")
        fid = compute_fid("./real_images", "./fake_images")
        shutil.rmtree("./fake_images", ignore_errors=True)
        return fid

    return _compute_fid
