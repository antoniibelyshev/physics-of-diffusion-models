from pydantic import BaseModel, Field, model_validator
from typing import Any
from typing_extensions import Self


class DataConfig(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    batch_size: int = Field(..., description="Batch size for training")
    obj_size: tuple[int, ...] = Field((), init=False, description="Size of an object")

    @model_validator(mode="after")
    def _set_obj_size(self) -> Self:
        match self.dataset_name:
            case "mnist":
                self.obj_size = (1, 32, 32)
            case "cifar10":
                self.obj_size = (3, 32, 32)
            case "cifar100":
                self.obj_size = (3, 32, 32)
            case "fashion_mnist":
                self.obj_size = (1, 32, 32)
            case _:
                raise ValueError
        return self


class DDPMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model architecture")
    parametrization: str = Field(..., description="Parametrization of the model")
    beta_min: float = Field(..., description="Minimum value of beta")
    beta_max: float = Field(..., description="Maximum value of beta")
    total_time: int = Field(..., description="Number of time steps")
    schedule_type: str = Field(..., description="Type of the temperature schedule")


class DDPMTrainingConfig(BaseModel):
    total_iters: int = Field(..., description="Total number of iterations for training")
    learning_rate: float = Field(..., description="Learning rate for training")
    weight_decay: float = Field(..., description="Weight decay for training")


class SampleConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_repeats: int = Field(1, description="Number of repeats")
    timestamp: int | None = Field(None, description="Starting timestamp")
    kwargs: dict[str, Any] = Field(..., description="Additional arguments for sampling")


class ForwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    min_temp: float = Field(..., description="Minimum value of log10(temp)")
    max_temp: float = Field(..., description="Maximum value of log10(temp)")
    n_temps: int = Field(..., description="Number of temperatures")


class BackwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    batch_size: int = Field(..., description="Batch size for stats computation")
    step_type: str = Field(..., description="Type of step")


class VariedDatasetStatsConfig(ForwardStatsConfig):
    dataset_names: tuple[str, ...]  = Field(
        ("mnist", "cifar10", "cifar100", "fashion_mnist"), description="Names of the datasets"
    )
    sample_fractions: tuple[float, ...] = Field((1.0, 0.1, 0.01), description="Sample fractions")


class Config(BaseModel):
    ddpm: DDPMConfig = Field(..., description="Diffusion model configuration")
    ddpm_training: DDPMTrainingConfig = Field(..., description="DDPM training configuration")
    data: DataConfig = Field(..., description="Data configuration")
    sample: SampleConfig = Field(..., description="Sample configuration")
    forward_stats: ForwardStatsConfig = Field(..., description="Forward statistics configuration")
    backward_stats: BackwardStatsConfig = Field(..., description="Backward statistics configuration")
    varied_dataset_stats: VariedDatasetStatsConfig = Field(
        ..., description="Varied dataset statistics configuration"
    )

    # wandb setup
    @property
    def experiment_name(self) -> str:
        return "_".join([
            self.data.dataset_name,
            self.ddpm.model_name,
            self.ddpm.parametrization,
            str(self.ddpm_training.total_iters),
            "iter",
            self.ddpm.schedule_type,
            "schedule",
        ])

    @property
    def project_name(self) -> str:
        return "physics-of-diffusion-models"

    @property
    def ddpm_checkpoint_path(self) -> str:
        return f"checkpoints/{self.experiment_name}.pth"

    @property
    def samples_prefix(self) -> str:
        return f"results/{self.experiment_name}_{self.sample.kwargs['step_type']}"

    @property
    def samples_path(self) -> str:
        return f"{self.samples_prefix}_samples.npz"

    @property
    def samples_from_timestamp_path(self) -> str:
        return f"{self.samples_prefix}_samples_from_timestamp_{self.sample.timestamp}.npz"

    @property
    def forward_stats_path(self) -> str:
        return f"results/{self.data.dataset_name}_forward_stats.npz"

    @property
    def forward_unbiased_stats_path(self) -> str:
        return f"results/{self.data.dataset_name}_forward_unbiased_stats.npz"

    @property
    def backward_stats_path(self) -> str:
        return f"results/{self.experiment_name}_backward_stats.npz"

    @property
    def flattening_temp_stats_path(self) -> str:
        match self.ddpm.schedule_type:
            case "flattening_temp":
                return self.forward_stats_path
            case "flattening_temp_unbiased":
                return self.forward_unbiased_stats_path
            case _:
                raise ValueError
