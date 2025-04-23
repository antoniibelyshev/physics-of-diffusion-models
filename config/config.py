from pydantic import BaseModel, Field
from typing import Any

from .dataset_configs import BaseDatasetConfig, DatasetRegistry


class DiffusionConfig(BaseModel):
    min_temp: float = Field(..., description="Minimum temperature")
    max_temp: float = Field(..., description="Maximum temperature")
    entropy_key: str = Field(..., description="Entropy key in the stats archive for the entropy schedule")

    @property
    def temp_range(self) -> tuple[float, float]:
        return self.min_temp, self.max_temp


class DDPMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model architecture")
    parametrization: str = Field(..., description="Parametrization of the model")
    noise_schedule_type: str = Field(..., description="Type of noise schedule")
    dim: int = Field(..., description="Base number of channels in the Unet")
    dim_mults: list[int] = Field(..., description="Base channel multipliers in the Unet")
    use_lrelu: bool = Field(..., description="Whether to use LeakyReLU instead of ReLU")


class DDPMTrainingConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    total_iters: int = Field(..., description="Total number of iterations for training")
    learning_rate: float = Field(..., description="Learning rate for training")
    weight_decay: float = Field(..., description="Weight decay for training")


class SampleConfig(BaseModel):
    n_steps: int = Field(..., description="Number of steps for sampling")
    step_type: str = Field(..., description="Type of step")
    ddpm_noise_schedule_type: str = Field(..., description="Type of noise schedule in the model")
    sample_noise_schedule_type: str = Field(..., description="Type of the noise schedule for sampling")
    n_samples: int = Field(..., description="Number of samples to generate")
    batch_size: int = Field(..., description="Batch size for sampling")
    track_ll: bool = Field(..., description="Whether to track log likelihood")
    track_states: bool = Field(..., description="Whether to track states")
    min_temp: float = Field(1e-4, description="Minimal temperature for sampling")
    precision: str = Field(..., description="Precision of the computations")


class ForwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    batch_size: int = Field(..., description="Number of repeats")
    n_temps: int = Field(..., description="Number of temperatures")
    unbiased: bool = Field(..., description="Whether to use unbiased estimation")


class BackwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    batch_size: int = Field(..., description="Batch size for stats computation")


class EmpiricalStatsConfig(BaseModel):
    n_temps: int = Field(..., description="Number of temperatures")
    n_steps_per_temp: int = Field(..., description="Number of loss accumulation steps per temperature level")
    batch_size: int = Field(..., description="Batch size")


class VariedDatasetStatsConfig(ForwardStatsConfig):
    dataset_names: tuple[str, ...]  = Field(
        ("mnist", "cifar10", "cifar100", "fashion_mnist"), description="Names of the datasets"
    )
    sample_fractions: tuple[float, ...] = Field((1.0, 0.1, 0.01), description="Sample fractions")


class FIDConfig(BaseModel):
    varied_parameters: dict[str, list[Any]] = Field(..., description="List of parameters to vary")
    # n_steps: list[int] = Field(..., description="Number of steps for sampling")
    # ddpm_noise_schedule_type: list[str] = Field(..., description="Type of noise schedules in the model")
    # sample_noise_schedule_type: list[str] = Field(..., description="Type of noise schedules for sampling")
    # step_type: list[str] = Field(..., description="Step types for sampling")
    train: bool = Field(..., description="Whether to use train sample for reference")
    sample: bool = Field(..., description="Whether to sample images or use sampled")
    save_imgs: bool = Field(..., description="Whether to save generated images")


class Config(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    diffusion: DiffusionConfig = Field(..., description="Diffusion configuration")
    ddpm: DDPMConfig = Field(..., description="DDPM configuration")
    ddpm_training: DDPMTrainingConfig = Field(..., description="DDPM training configuration")
    sample: SampleConfig = Field(..., description="Sample configuration")
    forward_stats: ForwardStatsConfig = Field(..., description="Forward statistics configuration")
    backward_stats: BackwardStatsConfig = Field(..., description="Backward statistics configuration")
    empirical_stats: EmpiricalStatsConfig = Field(..., description="Empirical statistics configuration")
    varied_dataset_stats: VariedDatasetStatsConfig = Field(
        ..., description="Varied dataset statistics configuration"
    )
    fid: FIDConfig = Field(..., description="FID configuration")

    dataset_registry: type[DatasetRegistry] = Field(DatasetRegistry, description="Dataset registry")

    @property
    def available_datasets(self) -> list[str]:
        return ["mnist", "cifar10"]
        return list(self.dataset_registry.get_dataset_names())

    @property
    def dataset_config(self) -> BaseDatasetConfig:
        return self.dataset_registry.get(self.dataset_name)

    @property
    def experiment_name(self) -> str:
        return "_".join([
            self.dataset_name,
            self.ddpm.model_name,
            self.ddpm.noise_schedule_type,
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
        return f"results/{self.experiment_name}_{self.sample.n_steps}_{self.sample.step_type}_steps"

    @property
    def samples_path(self) -> str:
        return f"{self.samples_prefix}_samples.npz"

    @property
    def forward_stats_path(self) -> str:
        return f"results/_{self.dataset_name}_forward{'_unbiased' if self.forward_stats.unbiased else ''}_stats.npz"

    @property
    def backward_stats_path(self) -> str:
        return f"results/{self.experiment_name}_backward_stats.npz"

    @property
    def empirical_stats_path(self) -> str:
        return f"results/{self.experiment_name}_empirical_stats.npz"

    def get_noise_schedule_stats_path(self, noise_schedule_type: str) -> str:
        assert noise_schedule_type.startswith("entropy")
        prev_unbiased = self.forward_stats.unbiased
        self.forward_stats.unbiased = "_u" in noise_schedule_type
        stats_path = self.forward_stats_path
        self.forward_stats.unbiased = prev_unbiased
        return stats_path

    @property
    def fid_results_path(self) -> str:
        return f"results/{self.dataset_name}_{'train' if self.fid.train else 'test'}_fid.csv"
