from pydantic import BaseModel, ConfigDict
from typing import Optional


class BaseDatasetConfig(BaseModel):
    """Base configuration class for datasets"""
    model_config = ConfigDict(frozen=True)  # Makes instances immutable

    # Class variables for dataset-specific constants
    name: str
    channels: int
    image_size: tuple[int, int]
    image_key: str = "image"
    min_temp: float
    max_temp: float
    fid_samples: int = 50000
    diffusers_model_id: Optional[str] = None
    hf_dataset_name: Optional[str] = None
 
    def __init__(self) -> None:
        super().__init__()

    @property
    def obj_size(self) -> tuple[int, ...]:
        """Returns the full object size including channels"""
        return self.channels, *self.image_size

    @property
    def temp_range(self) -> tuple[float, float]:
        """Returns the temperature range for forward stats"""
        return self.min_temp, self.max_temp


class DatasetRegistry:
    _configs: dict[str, BaseDatasetConfig] = {}

    @classmethod
    def register(cls, config_class: type[BaseDatasetConfig]) -> type[BaseDatasetConfig]:
        """Register a dataset config class"""
        config = config_class()
        cls._configs[config.name] = config
        return config_class

    @classmethod
    def get(cls, name: str) -> BaseDatasetConfig:
        """Get the dataset config class by name"""
        if name not in cls._configs:
            raise KeyError(f"Dataset config '{name}' not found. Available configs: {list(cls._configs.keys())}")
        return cls._configs[name]

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        """Get the names of all registered dataset configs"""
        return list(cls._configs.keys())


@DatasetRegistry.register
class MNISTConfig(BaseDatasetConfig):
    name: str = "mnist"
    channels: int = 1
    image_size: tuple[int, int] = (32, 32)
    min_temp: float = 1e-2
    max_temp: float = 1e4
    hf_dataset_name: Optional[str] = "mnist"


@DatasetRegistry.register
class CIFAR10Config(BaseDatasetConfig):
    name: str = "cifar10"
    channels: int = 3
    image_size: tuple[int, int] = (32, 32)
    image_key: str = "img"
    min_temp: float = 1e0
    max_temp: float = 1e6
    diffusers_model_id: Optional[str] = "./checkpoints/ddpm_ema_cifar10"
    hf_dataset_name: Optional[str] = "cifar10"


@DatasetRegistry.register
class CIFAR100Config(BaseDatasetConfig):
    name: str = "cifar100"
    channels: int = 3
    image_size: tuple[int, int] = (32, 32)
    image_key: str = "img"
    min_temp: float = 1e-1
    max_temp: float = 1e4
    hf_dataset_name: Optional[str] = "cifar100"


@DatasetRegistry.register
class FashionMNISTConfig(BaseDatasetConfig):
    name: str = "fashion_mnist"
    channels: int = 1
    image_size: tuple[int, int] = (32, 32)
    min_temp: float = 1e-1
    max_temp: float = 1e4
    hf_dataset_name: Optional[str] = "fashion_mnist"


@DatasetRegistry.register
class ImageNetConfig(BaseDatasetConfig):
    name: str = "image-net"
    channels: int = 3
    image_size: tuple[int, int] = (64, 64)
    min_temp: float = 1e-1
    max_temp: float = 1e4
    # diffusers_model_id: Optional[str] = "pcuenq/ddpm-ema-cifar"
    hf_dataset_name: Optional[str] = "benjamin-paine/imagenet-1k-64x64"


@DatasetRegistry.register
class CelebAConfig(BaseDatasetConfig):
    name: str = "celeba-hq"
    channels: int = 3
    image_size: tuple[int, int] = (256, 256)
    min_temp: float = 1e1
    max_temp: float = 1e6
    diffusers_model_id: Optional[str] = "google/ddpm-celebahq-256"
    hf_dataset_name: Optional[str] = "student/celebA"


@DatasetRegistry.register
class CelebaHQConfig(BaseDatasetConfig):
    name: str = "celeba-hq-256-30k"
    channels: int = 3
    image_size: tuple[int, int] = (256, 256)
    min_temp: float = 1e2
    max_temp: float = 1e7
    diffusers_model_id: Optional[str] = "google/ddpm-ema-celebahq-256"
    hf_dataset_name: Optional[str] = "korexyz/celeba-hq-256x256"


@DatasetRegistry.register
class LSUNBedroomsConfig(BaseDatasetConfig):
    name: str = "lsun-bedrooms"
    channels: int = 3
    image_size: tuple[int, int] = (256, 256)
    min_temp: float = 1e2
    max_temp: float = 1e7
    diffusers_model_id: Optional[str] = "google/ddpm-ema-bedroom-256"
    hf_dataset_name: Optional[str] = "pcuenq/lsun-bedrooms"


@DatasetRegistry.register
class GaussianConfig(BaseDatasetConfig):
    name: str = "gaussian"
    channels: int = 100
    image_size: tuple[int, int] = (1, 1)
    min_temp: float = 1e-1
    max_temp: float = 1e4