import torch
import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST  # type: ignore
from torch.utils.data import DataLoader, Dataset
from typing import Generator
from torch import Tensor
from PIL.Image import Image


def preprocess_image(image: Image) -> NDArray[np.float32]:
    return np.array(image.resize((32, 32)), dtype=np.float32) / 127.5 - 1
    # return np.array(image, dtype=np.float32) / 127.5 - 1


def postprocess_image(image: Tensor) -> Tensor:
    image = image.cpu().detach()
    image = (image + 1) * 127.5
    image = torch.clip(image, 0, 255)
    return image


class MnistDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, train: bool = True):
        super().__init__()
        self.mnist = MNIST(
            'data',
            train=train,
            download=True
        )

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, target = self.mnist[index]
        image = preprocess_image(image)
        target = torch.tensor(target).long()
        return {
            "images": Tensor(image)[None],
            "targets": target,
        }


class Cifar10Dataset(Dataset[dict[str, Tensor]]):
    def __init__(self, train: bool = True):
        super().__init__()
        self.cifar10 = CIFAR10(
            'data',
            train=train,
            download=True
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, target = self.cifar10[index]
        image = preprocess_image(image)
        target = torch.tensor(target).long()
        return {
            "images": Tensor(image).permute(2, 0, 1),  # Convert HWC to CHW
            "targets": target,
        }


class Cifar100Dataset(Dataset[dict[str, Tensor]]):
    def __init__(self, train: bool = True):
        super().__init__()
        self.cifar100 = CIFAR100(
            'data',
            train=train,
            download=True
        )

    def __len__(self) -> int:
        return len(self.cifar100)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, target = self.cifar100[index]
        image = preprocess_image(image)
        target = torch.tensor(target).long()
        return {
            "images": Tensor(image).permute(2, 0, 1),  # Convert HWC to CHW
            "targets": target,
        }


class FashionMnistDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, train: bool = True):
        super().__init__()
        self.fashion_mnist = FashionMNIST(
            'data',
            train=train,
            download=True
        )

    def __len__(self) -> int:
        return len(self.fashion_mnist)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, target = self.fashion_mnist[index]
        image = preprocess_image(image)
        target = torch.tensor(target).long()
        return {
            "images": Tensor(image)[None],
            "targets": target,
        }


def get_data_generator(
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True
) -> Generator[dict[str, Tensor], None, None]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )

    while True:
        yield from loader


def get_samples_tensor(dataset_name: str) -> Tensor:
    match dataset_name:
        case "mnist":
            dataset = MnistDataset()
        case "cifar10":
            dataset = Cifar10Dataset()
        case "cifar100":
            dataset = Cifar100Dataset()
        case "fashion_mnist":
            dataset = FashionMnistDataset()
        case _:
            raise ValueError(f"Dataset {dataset_name} not supported.")
    return torch.stack([dataset[i]["images"] for i in range(len(dataset))], 0)
