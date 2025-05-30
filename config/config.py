from pydantic import BaseModel, Field
from typing import Optional, Any

from .dataset_configs import BaseDatasetConfig, DatasetRegistry


class DiffusionConfig(BaseModel):
    min_temp: float = Field(..., description="Minimum temperature")
    max_temp: float = Field(..., description="Maximum temperature")

    @property
    def temp_range(self) -> tuple[float, float]:
        return self.min_temp, self.max_temp


class EntropyScheduleConfig(BaseModel):
    extrapolate: bool = Field(..., description="Extrapolate entropy schedule")
    min_temp: float = Field(..., description="Minimum temperature")
    max_temp: float = Field(..., description="Maximum temperature")


class DDPMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model architecture")
    parametrization: str = Field(..., description="Parametrization of the model")
    noise_schedule_type: str = Field(..., description="Type of noise schedule")
    unet_config: Optional[dict[str, Any]] = Field(None, description="Configuration for the UNet model")


class DDPMTrainingConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    total_iters: int = Field(..., description="Total number of iterations for training")
    learning_rate: float = Field(..., description="Learning rate for training")
    weight_decay: float = Field(..., description="Weight decay for training")
    ema_decay: float = Field(..., description="Decay rate for exponential moving average of model parameters")
    eval_steps: int = Field(..., description="Training steps between the evaluation phases")
    warmup_steps: int = Field(..., description="Number of warmup steps for learning rate")
    betas: tuple[float, float] = Field(..., description="Beta values for Adam optimizer")
    grad_clip: float = Field(..., description="Gradient clipping value (upper)")


class SampleConfig(BaseModel):
    n_steps: int = Field(..., description="Number of steps for sampling")
    step_type: str = Field(..., description="Type of step")
    noise_schedule_type: str = Field(..., description="Type of the noise schedule for sampling")
    n_samples: int = Field(..., description="Number of samples to generate")
    batch_size: int = Field(..., description="Batch size for sampling")
    precision: str = Field(..., description="Precision of the computations")


class ForwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of trajectory starts for stats estimate")
    batch_size: int = Field(..., description="Size of the batched trajectories")
    dataloader_batch_size: int = Field(..., description="SSize of the batches in the dataloader")
    n_temps: int = Field(..., description="Number of temperatures")


class EmpiricalStatsConfig(BaseModel):
    n_temps: int = Field(..., description="Number of temperatures")
    n_steps_per_temp: int = Field(..., description="Number of loss accumulation steps per temperature level")
    batch_size: int = Field(..., description="Batch size")


class DataAugmentationConfig(BaseModel):
    use_augmentation: bool = Field(False, description="Whether to use data augmentation")
    horizontal_flip: bool = Field(False, description="Whether to use random horizontal flips")


class FIDConfig(BaseModel):
    n_steps: list[int] = Field(..., description="Number of steps for sampling")
    noise_schedule_type: list[str] = Field(..., description="Type of noise schedules for sampling")
    min_temp: list[float] = Field(..., description="Minimum temperature for sampling")
    train: bool = Field(..., description="Whether to use train sample for reference")
    sample: bool = Field(..., description="Whether to sample images or use sampled")


class Config(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    diffusion: DiffusionConfig = Field(..., description="Diffusion configuration")
    entropy_schedule: EntropyScheduleConfig = Field(..., description="Entropy schedule configuration")
    ddpm: DDPMConfig = Field(..., description="DDPM configuration")
    ddpm_training: DDPMTrainingConfig = Field(..., description="DDPM training configuration")
    data_augmentation: DataAugmentationConfig = Field(..., description="Data augmentation configuration")
    sample: SampleConfig = Field(..., description="Sample configuration")
    forward_stats: ForwardStatsConfig = Field(..., description="Forward statistics configuration")
    empirical_stats: EmpiricalStatsConfig = Field(..., description="Empirical statistics configuration")
    fid: FIDConfig = Field(..., description="FID configuration")

    dataset_registry: type[DatasetRegistry] = Field(DatasetRegistry, description="Dataset registry")

    @property
    def available_datasets(self) -> list[str]:
        if self.dataset_name == "all":
            return list(self.dataset_registry.get_dataset_names())
        return [self.dataset_name]

    @property
    def dataset_config(self) -> BaseDatasetConfig:
        return self.dataset_registry.get(self.dataset_name)

    @property
    def ddpm_config_name(self) -> str:
        if self.ddpm.model_name == "unet":
            return f"unet_{self.ddpm.parametrization}_{self.ddpm.noise_schedule_type}_schedule"
        return self.ddpm.model_name

    @property
    def experiment_name(self) -> str:
        return "_".join([
            self.dataset_name,
            self.ddpm_config_name
        ])

    @property
    def project_name(self) -> str:
        return "physics-of-diffusion-models"

    @property
    def ddpm_checkpoint_path(self) -> str:
        return f"checkpoints/{self.experiment_name}.pth"

    @property
    def samples_path(self) -> str:
        return "_".join([
            f"samples/{self.experiment_name}",
            str(self.sample.n_steps),
            self.sample.step_type,
            "steps",
        ])

    @property
    def forward_stats_path(self) -> str:
        return f"stats/{self.dataset_name}_forward.npz"

    @property
    def empirical_stats_path(self) -> str:
        return f"stats/{self.experiment_name}_empirical.npz"

    @property
    def fid_results_path(self) -> str:
        return f"fid/{self.experiment_name}.csv"
