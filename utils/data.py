import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST # type: ignore
from torchvision.transforms import Compose, Resize, ToTensor # type: ignore
from typing import Generator

from config import Config
    

class ImageDataset(TensorDataset):
    DATASET_CLASSES = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "fashion_mnist": FashionMNIST,
    }

    def __init__(self, dataset_name: str, image_size: tuple[int, int, int], *, train: bool = True):
        super().__init__()

        transform = Compose([
            Resize(image_size[1:]),
            ToTensor(),
        ])
        self.dataset = self.DATASET_CLASSES[dataset_name](
            "data",
            train=train,
            download=True,
            transform=transform
        )


def get_dataset(config: Config, *, train: bool = True) -> TensorDataset:
    dataset_name = config.data.dataset_name
    if dataset_name in ImageDataset.DATASET_CLASSES:
        obj_size = config.data.obj_size
        assert len(obj_size) == 3
        return ImageDataset(dataset_name, obj_size, train=train)
    raise ValueError(f"Unknown dataset {dataset_name}")


def get_data_generator(
    dataset: TensorDataset,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Generator[tuple[Tensor, ...], None, None]:
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        drop_last = drop_last
    )

    while True:
        yield from loader


def get_data_tensor(config: Config, train: bool = True) -> Tensor:
    dataset = get_dataset(config, train=train)
    return torch.stack([dataset[i][0] for i in range(len(dataset))], 0)
