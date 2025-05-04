import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from datasets import load_dataset  # type: ignore
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip  # type: ignore
from tqdm import tqdm
import os
from typing import Generator, Iterable

from config import Config
from .synthetic_datasets import generate_dataset


def get_default_num_workers() -> int:
    """
    Determines the default number of workers based on CPU cores.
    Returns half of the available CPU cores, but at least 1.
    """
    cpu_count = os.cpu_count()
    return 1 if cpu_count is None else max(1, cpu_count // 2)


class HFDataset(Dataset[tuple[Tensor, ...]]):
    """
    A generic class to wrap Hugging Face datasets into a PyTorch Dataset.
    """
    def __init__(
            self,
            hf_dataset_name: str,
            image_size: tuple[int, int],
            image_key: str,
            *,
            split: str = "train",
            config: Config,
    ) -> None:
        super().__init__()

        # Load dataset using Hugging Face's `load_dataset`
        self.dataset = load_dataset(hf_dataset_name, split=split)

        # Define transformations
        transforms = [
            Resize(image_size),
            ToTensor(),
        ]

        # Add data augmentation if enabled
        if config.data_augmentation.use_augmentation:
            if config.data_augmentation.horizontal_flip:
                transforms.insert(1, RandomHorizontalFlip())

        # Add normalization for non-MNIST datasets
        if hf_dataset_name != "mnist":
            transforms.append(Normalize(mean=0.5, std=0.5))

        self.transform = Compose(transforms)
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        data = self.dataset[idx]
        image = self.transform(data[self.image_key])

        if "label" in data:
            label = data["label"]
            return image, torch.tensor(label)
        else:
            return (image,)


def get_dataset(config: Config, train: bool = True) -> Dataset[tuple[Tensor, ...]]:
    dataset_config = config.dataset_config
    if (hf_dataset_name := dataset_config.hf_dataset_name) is not None:
        return HFDataset(
            hf_dataset_name, 
            config.dataset_config.image_size, 
            config.dataset_config.image_key,
            config=config
        )
    else:
        return TensorDataset(generate_dataset(config.dataset_name))


def get_data_generator(
        dataset: Dataset[tuple[Tensor, ...]],
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
) -> Generator[tuple[Tensor, ...], None, None]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=get_default_num_workers(),
    )

    while True:
        yield from loader


def get_data_tensor(config: Config, train: bool = True) -> Tensor:
    dataset = get_dataset(config, train=train)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=get_default_num_workers())
    return torch.cat([x[0] for x in tqdm(dataloader, desc="Loading data ...")])


def to_uint8(images: Tensor, values_range: tuple[float, float] = (-1, 1)) -> Tensor:
    a, b = values_range
    return ((torch.clip(images, a, b) - a) / (b - a) * 255).to(dtype=torch.uint8)
