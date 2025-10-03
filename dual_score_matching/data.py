""" Datasets and dataloaders. """

from pathlib import Path
from typing import *

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms

from .memoize import disk_memoize, short_name
from .tensor_ops import rand_log_uniform
from .trackers import BatchCovarianceTracker


class DatasetInfo:
    """ Named-tuple-like class holding general dataset statistics. Note: this class assumes square images. """
    def __init__(self, spatial_size: int, num_channels: int, mean: float, variance: float) -> None:
        self.spatial_size: int = spatial_size
        self.num_channels: int = num_channels
        self.mean: float = mean  # Dataset mean (over pixels and channels)
        self.variance: float = variance  # Dataset variance (over pixels and channels)

    @property
    def dimension(self) -> int:
        """ Returns the number of pixels times the number of channels. """
        return self.spatial_size ** 2 * self.num_channels

    @property
    def stddev(self) -> float:
        """ Returns the standard deviation over pixels and channels. """
        return np.sqrt(self.variance)


def get_dataset_info(dataloader: DataLoader, num_samples: int = 1000) -> DatasetInfo:
    """ Returns DatasetInfo for the given dataloader. """
    tracker = BatchCovarianceTracker()
    for x in dataloader:  # (B, C, H, W)
        # Drop class/other information if provided in dataset.
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.cuda()
        num_channels, spatial_size = x.shape[1], x.shape[-1]
        tracker.update(x.reshape((-1, 1)))  # (BCHW, 1)
        if tracker.num_samples() >= num_samples * spatial_size ** 2 * num_channels:  # num_samples should refer to images as opposed to pixels or channels
            break

    # Assumes H = W and convert to float.
    info = DatasetInfo(spatial_size=spatial_size, num_channels=num_channels,
                       mean=tracker.mean().item(), variance=tracker.covariance().item())
    return info


def load_data(dataset: str, spatial_size: Optional[int] = None, grayscale: bool = True, horizontal_flip: bool = True, data_subset: Optional[slice | Iterable[int]] = None,
              train_batch_size: int = 256, test_batch_size: int = 256, num_workers: int = 0, seed: int = None, load_only_info: bool = False) -> tuple[DataLoader, DataLoader, DatasetInfo]:
    """ Returns infinite dataloaders for both train and test sets, and compute (memoize) dataset statistics.
    Yields images with pixel values in [0, 1].

    Args:
        path: path to a folder of images (the dataset info will be saved in the same folder)
        spatial_size: optionally, extract patches of this given size
        data_subset: optionally, take a subset of the training data
        batch_size:
        num_workers:

    Returns:
        train_dataloader, test_dataloader, dataset_info
    """
    transform_kwargs = dict(spatial_size=spatial_size, grayscale=grayscale, horizontal_flip=horizontal_flip)
    transform = get_transform(**transform_kwargs)

    train_dataset, path, name = get_dataset(dataset=dataset, transform=transform, train=True, load_dataset=not load_only_info)
    if not load_only_info:
        train_dataset = take_subset(train_dataset, data_subset)  # Optionally take a subset of the data.
        train_dataloader = get_dataloader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers, seed=seed)

        test_dataset, _, _ = get_dataset(dataset=dataset, transform=transform, train=False)
        test_dataloader = get_dataloader(dataset=test_dataset, batch_size=test_batch_size, num_workers=num_workers, seed=seed)
    else:
        train_dataloader, test_dataloader = None, None

    def get_info():
        if load_only_info:
            raise ValueError("Dataset info not found")  # This will not be executed because of memoization.
        return get_dataset_info(train_dataloader)

    # Note: we do not recompute dataset info when taking a subset of the data.
    info_path = path / short_name(name, **transform_kwargs)
    info: DatasetInfo = disk_memoize(info_path, get_info)

    return train_dataloader, test_dataloader, info


def get_dataset(dataset: str, transform: transforms.Transform, train: bool, load_dataset: bool = True) -> Tuple[Dataset, Path]:
    """ Returns a dataset with the given transform, possibly taking a subset of the data.
    :param dataset: name of common dataset or path to an image folder
    :transform: transform to apply to the images
    :train: whether to use the training or testing set (for image folders, the last 10% of images are used for testing)
    :load_dataset: if False, do not load the dataset, simply return the path to the dataset folder and name for info file
    :return: dataset, path to the dataset folder, and name for info file (for memoization)
    """
    if dataset in dataset_class:
        root = "data"
        path = Path(root) / dataset_class[dataset].base_folder
        name = "info"

        if load_dataset:
            dataset = dataset_class[dataset](root=root, train=train, transform=transform, download=False)  # No download to avoid double check_integrity and "Files downloaded and verified" print.
        else:
            dataset = None
    elif dataset == "disks":
        images = disk_dataset(patch_size=64, num_patches=1_000_000 if train else 10_000, translate=True, scale=True, foreground=True, background=True, wrap=False, continuous=True)
        dataset = torch.utils.data.TensorDataset(images)  # Ignores transforms...
        path = Path("data")
        name = "info_disks"
    else:
        path: Path = Path(dataset)
        if path.is_dir():
            if load_dataset:
                # Assumes dataset is the path to a folder of images.
                files: list[Path] = sorted(path.glob("*.png"))

                # Use 10% of the images for testing by default.
                num_files = len(files)
                num_test = num_files // 10
                files = files[:-num_test] if train else files[-num_test:]

                dataset = ImageListDataset(files, transform)
            else:
                dataset = None

            name = "info"
        else:
            if load_dataset:
                # Assumes dataset is a torch file containing images.
                filename = str(path)

                images = torch.load(filename)  # (N, C, H, W), should be float32 with values in [0, 1]

                # Use 10% of the images for testing by default.
                num_imgs = len(images)
                num_test = num_imgs // 10
                images = images[:-num_test] if train else images[-num_test:]

                dataset = torch.utils.data.TensorDataset(images)  # Ignores transforms...
            else:
                dataset = None

            name = f"{path.stem}_info"
            path = path.parent

    return dataset, path, name


def take_subset(dataset: Dataset, data_subset: Optional[slice | Iterable[int]]) -> Dataset:
    """ Takes a subset of the dataset (does nothing if data_subset is None). data_subset can be a slice or an iterable of indices. """
    if data_subset is not None:
        if isinstance(data_subset, slice):
            data_subset = np.arange(len(dataset))[data_subset]
        dataset = torch.utils.data.Subset(dataset, data_subset)
    return dataset


def get_transform(spatial_size: Optional[int], grayscale=False, horizontal_flip=True) -> transforms.Transform:
    """ Returns a transform that extracts a patch in the image at a random location and scale, then randomly flips it horizontally.
    Aspect ratio is maintained. Rotations are disabled to preserve the vertical orientation of the images.
    """
    return transforms.Compose(list(filter(None, [
        transforms.Grayscale() if grayscale else None,
        RandomResizedCrop(size=spatial_size) if spatial_size is not None else None,
        transforms.RandomHorizontalFlip() if horizontal_flip else None,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),  # converts to float [0, 1]
    ])))


def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int, seed=0, num_samples: Optional[int] = None, num_epochs: Optional[int] = None) -> DataLoader:
    sampler = InfiniteSampler(dataset, seed=seed, num_samples=num_samples, num_epochs=num_epochs)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


class ImageListDataset(Dataset):
    """ Simple Dataset for an image folder. """
    def __init__(self, files: list[Path], transform: transforms.Transform):
        """
        Args:
            path: path to a folder of images
            num_repeats: artificially repeats the dataset this many times to avoid small batches
            transform: _description_
        """
        super().__init__()
        self.files: list[Path] = files
        self.num_images: int = len(self.files)
        self.transform: transforms.Transform = transform

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int) -> np.ndarray:
        """ Returns (C, H, W) uint8 tensor with values in [0, 255]. """
        return self.transform(torchvision.io.read_image(str(self.files[index])))  # (C, H, W)


class ImageNet32(torchvision.datasets.CIFAR10):
    """ ImageNet32 dataset. """

    base_folder = "imagenet32"
    url = None
    filename = None
    tgz_md5 = None
    train_list = [
        ["train_data_batch_1",  "6c4495e65cd24a8c3019857ef9b758ee"],
        ["train_data_batch_2",  "3dd727de4155836807dfc19f98c9fe70"],
        ["train_data_batch_3",  "03d3dc4e850e23e1d526f268a0196296"],
        ["train_data_batch_4",  "876f7e6d6ddb1f52ecb654ffdc8293f6"],
        ["train_data_batch_5",  "c789bcdd1feed34a9bc58598a1a1bf7d"],
        ["train_data_batch_6",  "8ce5344cb1e11f31bc507cae4c856083"],
        ["train_data_batch_7",  "32ecc8ad6c55b1c9cb6cf79a2cc46094"],
        ["train_data_batch_8",  "bdeb6da3d05771121992b48c59c69439"],
        ["train_data_batch_9",  "58417149b5ce31688f805341e7f57b4f"],
        ["train_data_batch_10", "46ad60a1144aaf97a143914453b0dabb"],
    ]

    test_list = [
        ["val_data", "06c02b8b4c8de8b3a36b07859a49de6f"],
    ]

    meta = {}

    def _load_meta(self):
        pass

    def _check_integrity(self) -> bool:
        # Disabled because slow
        return True


class ImageNet64(torchvision.datasets.CIFAR10):
    """ ImageNet64 dataset. """

    base_folder = "imagenet64"
    url = None
    filename = None
    tgz_md5 = None
    train_list = [
        ["train_data_batch_1",  ""],
        ["train_data_batch_2",  ""],
        ["train_data_batch_3",  ""],
        ["train_data_batch_4",  ""],
        ["train_data_batch_5",  ""],
        ["train_data_batch_6",  ""],
        ["train_data_batch_7",  ""],
        ["train_data_batch_8",  ""],
        ["train_data_batch_9",  ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [
        ["val_data", ""]
    ]

    meta = {}

    def _load_meta(self):
        # Fix shape of dataset because CIFAR10 assumes 32x32 resolution. Need to invert transpose...
        self.data = self.data.transpose((0, 3, 1, 2))  # convert back to CHW
        self.data = self.data.reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def _check_integrity(self) -> bool:
        # Disabled because slow
        return True


dataset_class = dict(
    ImageNet32=ImageNet32, ImageNet64=ImageNet64, CIFAR10=torchvision.datasets.CIFAR10,
    # CIFAR100=datasets.CIFAR100, MNIST=datasets.MNIST, ImageNet=datasets.ImageFolder,
)


class InfiniteSampler(torch.utils.data.Sampler):
    """ Infinite sampler (or not!) which cycles over random permutations of the dataset. """
    def __init__(self, dataset: Dataset, seed: Optional[int] = 0, num_samples: Optional[int] = None, num_epochs: Optional[int] = None) -> None:
        """ If seed is None, then the sampler will cycle over the dataset in order (no shuffling). """
        super().__init__()
        self.dataset: Dataset = dataset
        self.dataset_size: int = len(self.dataset)
        self.generator = np.random.default_rng(seed) if seed is not None else None
        self.num_samples: int = num_samples or np.inf
        self.num_epochs: int = num_epochs or np.inf

    def permutation(self):
        """ Return a random permutation of the dataset (or the ordered list of indices if randomness has been disabled). """
        return self.generator.permutation(self.dataset_size) if self.generator is not None else range(self.dataset_size)

    def __iter__(self) -> Iterator[int]:
        num_samples = 0
        num_epochs = 0
        while True:
            if num_epochs == self.num_epochs:
                return
            for idx in self.permutation():
                if num_samples == self.num_samples:
                    return
                yield idx
                num_samples += 1
            num_epochs += 1

    def __len__(self) -> int:
        return min(self.num_samples, self.num_epochs * self.dataset_size)  # Can be np.infty


class RandomResizedCrop(torch.nn.Module):
    """ Custom random resized crop to have more control. """
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Extracts a random patch of x of size between self.size and x.size, and rescales to self.size. """
        # Samples a scale log-uniformly distributed in the allowed range.
        H, W = x.shape[-2:]
        max_scale = min(H, W) / self.size
        scale = rand_log_uniform(shape=(), low=1, high=max_scale).item()

        # Patch parameters.
        patch_size = int(round(self.size * scale))
        top = np.random.randint(0, H - patch_size + 1)
        left = np.random.randint(0, W - patch_size + 1)

        return transforms.functional.resized_crop(x, top=top, left=left, height=patch_size, width=patch_size, size=(self.size, self.size),
                                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)


def disk_dataset(patch_size, num_patches, translate=False, scale=False, foreground=False, background=False,
                   wrap=False, continuous=True):
    """ Returns (N, 1, H, W) images of disks.
    :param patch_size: (H, W) tuple
    :param num_patches: number of samples N
    :param translate: whether to translate the disks
    :param scale: whether to scale the disks
    :param foreground: whether to randomize the foreground color
    :param background: whether to randomize the background color
    :param wrap: whether to generate wrapping disks at the boundaries
    :param continuous: whether consider continuous positions and sizes as opposed to integers
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    else:
        assert patch_size[0] == patch_size[1]  # XXX too lazy to implement rectangular images
    size = patch_size[0]  # Generate large images by the aliasing factor

    # Use numpy for randint because pytorch doesn't support low and high arguments as tensors
    rand_np = np.random.uniform if continuous else np.random.randint
    rand = lambda *args, **kwargs: torch.from_numpy(rand_np(*args, **kwargs)).to(torch.float32 if continuous else torch.int64)

    min_r = 3
    max_r = size / 2 - min_r if continuous else size // 2 - min_r
    radius = rand(size=(num_patches,), low=min_r, high=max_r) if scale \
        else torch.full(size=(num_patches,), fill_value=size // 4)  # (N,)

    r = 0 if wrap else radius[:, None]
    border = 1
    center = rand(size=(num_patches, 2), low=r + border, high=size - r - border) if translate \
        else torch.full(size=(num_patches, 2), fill_value=size // 2, dtype=torch.int32)  # (N, 2)

    foregrounds = torch.rand((num_patches,)) if foreground else torch.ones((num_patches,))  # (N,)
    backgrounds = torch.rand((num_patches,)) if background else torch.zeros((num_patches,))  # (N,)

    y, x = torch.meshgrid(*(torch.arange(s) for s in (size, size)), indexing="ij")  # (H, W) both
    u = torch.stack((x, y), dim=-1)  # (H, W, 2)

    # Compute modulo in [-(size // 2), (size - 1) // 2]
    diff_mod = lambda diff: (diff + size / 2) % size - size / 2
    signed_dist = torch.sqrt(torch.sum(diff_mod(u[None] - center[:, None, None]) ** 2, dim=-1)) - radius[:, None, None].float()  # (N, H, W)
    # Build disk with cosine boundary
    disk = (signed_dist < 0).float()  # (N, H, W)
    I = (signed_dist > 0) & (signed_dist < border)
    disk[I] = torch.cos(np.pi * signed_dist[I] / (2 * border))  # (N, H, W)

    data = backgrounds[:, None, None] + (foregrounds - backgrounds)[:, None, None] * disk.float()  # (N, H, W)
    data = data[:, None]  # (N, 1, H, W)
    return data
