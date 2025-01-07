from matplotlib.cbook import STEP_LOOKUP_MAP
from pydantic import BaseModel, Field
from typing import Any


class BaseDataConfig(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    obj_size: tuple[int, ...] = Field(..., description="Size of the object in the dataset")
    batch_size: int = Field(128, description="Batch size for training")


class BaseDDPMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model architecture")
    parametrization: str = Field(..., description="Parametrization of the model")
    beta_min: float = Field(1e-4, description="Minimum value of beta")
    beta_max: float = Field(2e-2, description="Maximum value of beta")
    T: int = Field(1000, description="Number of time steps")


class BaseDDPMTrainingConfig(BaseModel):
    total_iters: int = Field(100000, description="Total number of iterations for training")
    learning_rate: float = Field(1e-2, description="Learning rate for training")
    weight_decay: float = Field(1e-2, description="Weight decay for training")


class BaseSampleConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_repeats: int = Field(1, description="Number of repeats")
    timestamp: int | None = Field(None, description="Starting timestamp")
    kwargs: dict[str, Any] = Field({}, description="Additional arguments for sampling")


class BaseForwardStatsConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_repeats: int = Field(1, description="Number of repeats")
    min_temp: float = Field(1e-2, description="Minimum value of log10(temp)")
    max_temp: float = Field(1e+2, description="Maximum value of log10(temp)")
    n_temps: int = Field(500, description="Number of temperatures")


class BaseBackwardStatsConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_repeats: int = Field(1, description="Number of repeats")
    batch_size: int = Field(20, description="Batch size for stats computation")
    step_type: str = Field("sde", description="Type of step")


class BaseVariedDatasetStatsConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_repeats: int = Field(1, description="Number of repeats")
    dataset_names: list[str]  = Field(["mnist", "cifar10", "cifar100", "fashion_mnist"], description="Names of the datasets")
    sample_fractions: list[float] = Field([1.0, 0.1, 0.01], description="Sample fractions")
    min_temp: float = Field(1e-2, description="Minimum value of log10(temp)")
    max_temp: float = Field(1e+2, description="Maximum value of log10(temp)")
    n_temps: int = Field(500, description="Number of temperatures")


class BaseConfig(BaseModel):
    ddpm: BaseDDPMConfig = Field(..., description="Diffusion model configuration")
    ddpm_training: BaseDDPMTrainingConfig = Field(..., description="DDPM training configuration")
    data: BaseDataConfig = Field(..., description="Data configuration")
    sample: BaseSampleConfig = Field(..., description="Sample configuration")
    forward_stats: BaseForwardStatsConfig = Field(..., description="Forward statistics configuration")
    backward_stats: BaseBackwardStatsConfig = Field(..., description="Backward statistics configuration")
    varied_dataset_stats: BaseVariedDatasetStatsConfig = Field(..., description="Varied dataset statistics configuration")

    # paths
    ddpm_checkpoint_path: str = Field("", description="Path to the checkpoint")
    samples_path: str = Field("", description="Path to the samples")
    samples_from_timestamp_path: str = Field("", description="Path to the samples from timestamp")
    forward_stats_path: str = Field("", description="Path to the forward stats")
    backward_stats_path: str = Field("", description="Path to the backward stats")
    varied_dataset_stats_path_suffix: str = Field("", description="Path to the varied dataset stats")

    # wandb
    project_name: str = Field("physics-of-diffusion-models", description="Name of the project")
    experiment_name: str = Field("", description="Name of the experiment")
