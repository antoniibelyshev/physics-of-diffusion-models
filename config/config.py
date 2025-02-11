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
    beta0: float = Field(..., description="Minimum value of beta")
    beta1: float = Field(..., description="Maximum value of beta")
    schedule_type: str = Field(..., description="Type of the temperature schedule")
    dim: int = Field(..., description="Base number of channels in the Unet")
    dim_mults: list[int] = Field(..., description="Base channel multipliers in the Unet")
    use_lrelu: bool = Field(..., description="Whether to use LeakyReLU instead of ReLU")

    @property
    def min_t(self) -> float:
        match self.schedule_type:
            case "linear_beta":
                return 1e-3
            case "cosine":
                return 0
            case _:
                return 1e-3


class DDPMTrainingConfig(BaseModel):
    total_iters: int = Field(..., description="Total number of iterations for training")
    learning_rate: float = Field(..., description="Learning rate for training")
    weight_decay: float = Field(..., description="Weight decay for training")
    continuous_time: bool = Field(..., description="Whether to use continuous time sampling during training")


class GANConfig(BaseModel):
    dim_mults_g: list[int] = Field(..., description="Base channel multipliers in the generator")
    base_channels_g: int = Field(..., description="Base number of channels in the generator")
    dim_mults_d: list[int] = Field(..., description="Base channel multipliers in the discriminator")
    base_channels_d: int = Field(..., description="Base number of channels in the discriminator")


class GANTrainingConfig(BaseModel):
    lr_g: float = Field(..., description="Learning rate for generator")
    weight_decay_g: float = Field(..., description="Weight decay for generator")
    n_iter_g: int = Field(..., description="Number of iterations for generator")
    lr_d: float = Field(..., description="Learning rate for discriminator")
    weight_decay_d: float = Field(..., description="Weight decay for discriminator")
    n_iter_d: int = Field(..., description="Number of iterations for discriminator")
    real_p: float = Field(..., description="Smoothing parameter for real samples")
    fake_p: float = Field(..., description="Smoothing parameter for fake samples")
    temp: float = Field(..., description="Temperature for noise")
    real_temp: float = Field(..., description="Temperature for noise in real samples")
    eval_steps: int = Field(..., description="Number of steps between evaluation")
    project_name: str = Field(..., description="Name of the project")
    total_iters: int = Field(..., description="Total number of iterations for training")


class SampleConfig(BaseModel):
    n_steps: int | list[int] = Field(..., description="Number of steps for sampling")
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    idx_start: int | None = Field(None, description="Starting index")
    step_type: str = Field("sde", description="Type of step")
    track_ll: bool = Field(False, description="Whether to track log likelihood")


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
    gan: GANConfig = Field(..., description="GAN configuration")
    gan_training: GANTrainingConfig = Field(..., description="GAN training configuration")
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
            # self.ddpm.model_name,
            # self.ddpm.parametrization,
            # str(self.ddpm_training.total_iters),
            # "iter",
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
        step_type = "_ode" if self.sample.step_type == "ode" else ""
        return f"results/{self.experiment_name}_{self.sample.n_steps}{step_type}_steps"

    @property
    def samples_path(self) -> str:
        return f"{self.samples_prefix}_samples.npz"

    @property
    def samples_from_timestamp_path(self) -> str:
        return f"{self.samples_prefix}_samples_from_timestamp_{self.sample.idx_start}.npz"

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
            case "entropy":
                return self.forward_stats_path
            case "entropy_u":
                return self.forward_unbiased_stats_path
            case _:
                raise ValueError
