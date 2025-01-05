import torch
import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST # type: ignore
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Generator
from torch import Tensor
from PIL.Image import Image
from base_config import BaseConfig


def preprocess_image(image: Image, size: tuple[int, int]) -> NDArray[np.float32]:
    return np.array(image.resize(size), dtype = np.float32) / 127.5 - 1


def postprocess_image(image: Tensor) -> Tensor:
    image = image.cpu().detach()
    image = (image + 1) * 127.5
    image = torch.clip(image, 0, 255)
    return image


class SizedDataset(Dataset[dict[str, Tensor]]):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        raise NotImplementedError
    

class ImageDataset(SizedDataset):
    DATASET_CLASSES = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "fashion_mnist": FashionMNIST,
    }

    DEFAULR_IMAGE_SIZES = {
        "mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "fashion_mnist": (1, 28, 28),
    }

    def __init__(
        self,
        dataset_name: str,
        image_size: tuple[int, int, int] | None = None,
        train: bool = True,
    ):
        super().__init__()

        self.dataset = self.DATASET_CLASSES[dataset_name](
            'data',
            train=train,
            download=True
        )

        self.image_size = image_size or self.DEFAULR_IMAGE_SIZES[dataset_name]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, target = self.dataset[index]
        image = preprocess_image(image, self.image_size[1:])
        target = torch.tensor(target).long()
        return {
            "images": Tensor(image).permute(2, 0, 1),
            "targets": target,
        }


def get_data_generator(
    dataset: Tensor,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Generator[Tensor, None, None]:
    loader = DataLoader(
        TensorDataset(dataset),
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        drop_last = drop_last
    )

    while True:
        yield from loader


def get_data_tensor(config: BaseConfig) -> Tensor:
    if config.data.dataset_name in ImageDataset.DATASET_CLASSES:
        assert len(config.data.obj_size) == 3
        dataset = ImageDataset(config.data.dataset_name, config.data.obj_size)
    else:
        raise ValueError(f"Unknown dataset name: {config.data.dataset_name}")
    return torch.stack([dataset[i]["images"] for i in range(len(dataset))], 0)


def get_obj_size(dataset_name: str) -> tuple[int, int, int]:
    if dataset_name in ImageDataset.DATASET_CLASSES:
        return ImageDataset.DEFAULR_IMAGE_SIZES[dataset_name]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
