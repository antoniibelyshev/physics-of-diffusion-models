import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST, ImageNet, VisionDataset, ImageFolder  # type: ignore
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Normalize # type: ignore
from tqdm import tqdm
from typing import Generator, Any

from config import Config


class CelebA:
    def __init__(self, root: str, *, train: bool, download: bool, **kwargs: Any) -> None:
        super().__init__()

        self.dataset = ImageFolder(root=root + "/celeba", **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.dataset[index]  # type: ignore


class ImageDataset(Dataset[tuple[Tensor, Tensor]]):
    DATASET_CLASSES = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "fashion_mnist": FashionMNIST,
        "image_net": ImageNet,
        "celeba": CelebA,
    }

    def __init__(self, dataset_name: str, image_size: tuple[int, ...], *, train: bool = True, labeled: bool = False):
        super().__init__()
        
        assert len(image_size) == 3
        transform = Compose([
            Resize(image_size[1:]),
            ToTensor(),
            # Lambda(lambda x: x.half()),
            *(() if dataset_name == "mnist" else (Normalize(mean=0.5, std=0.5),)),
        ])
        self.dataset = self.DATASET_CLASSES[dataset_name](
            "data",
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[idx]
        return image, torch.tensor(label)


def get_dataset(config: Config, train: bool = True, labeled: bool = False) -> Dataset[tuple[Tensor, Tensor]]:
    dataset_name = config.data.dataset_name
    if dataset_name in ImageDataset.DATASET_CLASSES:
        obj_size = config.data.obj_size
        return ImageDataset(dataset_name, obj_size, train=train, labeled=labeled)
    raise ValueError(f"Unknown dataset {dataset_name}")


def get_data_generator(
    dataset: Dataset[tuple[Tensor, ...]],
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Generator[tuple[Tensor, ...], None, None]:
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers=8,
    )

    while True:
        yield from loader


def get_data_tensor(config: Config, train: bool = True) -> Tensor:
    dataset = get_dataset(config, train=train)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=8)
    return torch.cat([x[0] for x in tqdm(dataloader)])


def to_uint8(images: Tensor, values_range: tuple[float, float] = (-1, 1)) -> Tensor:
    a, b = values_range
    return ((torch.clip(images, a, b) - a) / (b - a) * 255).to(dtype=torch.uint8)
