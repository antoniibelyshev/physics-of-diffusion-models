import torch
import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST # type: ignore
from torchvision.transforms import ToTensor # type: ignore
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Generator
from torch import Tensor
from PIL.Image import Image
from config import Config


def preprocess_image(image: Image, size: tuple[int, int]) -> NDArray[np.float32]:
    return np.array(image.resize(size), dtype = np.float32) / 127.5 - 1.


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

    def __init__(
        self,
        dataset_name: str,
        image_size: tuple[int, int, int],
        train: bool = True,
    ):
        super().__init__()

        self.dataset = self.DATASET_CLASSES[dataset_name](
            'data',
            train=train,
            download=True
        )

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pil_image, target = self.dataset[index]
        image = self.preprocess_image(pil_image)
        target = torch.tensor(target).long()
        return {
            "images": image,
            "targets": target,
        }

    def preprocess_image(self, pil_image: Image) -> Tensor:
        return ToTensor()(pil_image.resize(self.image_size[1:])) # type: ignore


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
        for batch in loader:
            yield batch[0]


def get_data_tensor(config: Config) -> Tensor:
    if config.data.dataset_name in ImageDataset.DATASET_CLASSES:
        assert len(config.data.obj_size) == 3
        dataset = ImageDataset(config.data.dataset_name, config.data.obj_size)
    else:
        raise ValueError(f"Unknown dataset name: {config.data.dataset_name}")
    return torch.stack([dataset[i]["images"] for i in range(len(dataset))], 0)
