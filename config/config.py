from pydantic import BaseModel, Field


class DiffusionConfig(BaseModel):
    noise_schedule: str = Field(..., description="Type of the noise schedule")

    @property
    def temp_range(self) -> tuple[float, float]:
        if self.noise_schedule in ["linear_beta", "cosine"]:
            return 1e-4, 1e4
        if self.noise_schedule.startswith("entropy"):
            return 1e-4, 1e4
        raise ValueError(f"Unknown noise schedule {self.noise_schedule}")


class DataConfig(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")

    @property
    def obj_size(self) -> tuple[int, ...]:
        match self.dataset_name:
            case "mnist":
                return 1, 32, 32
            case "cifar10":
                return 3, 32, 32
            case "cifar100":
                return 3, 32, 32
            case "fashion_mnist":
                return 1, 32, 32
            case _:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")


class DDPMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model architecture")
    parametrization: str = Field(..., description="Parametrization of the model")
    dim: int = Field(..., description="Base number of channels in the Unet")
    dim_mults: list[int] = Field(..., description="Base channel multipliers in the Unet")
    use_lrelu: bool = Field(..., description="Whether to use LeakyReLU instead of ReLU")


class DDPMTrainingConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    total_iters: int = Field(..., description="Total number of iterations for training")
    learning_rate: float = Field(..., description="Learning rate for training")
    weight_decay: float = Field(..., description="Weight decay for training")


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
    batch_size: int = Field(..., description="Batch size for gan training")
    temp: float = Field(..., description="Temperature for noise")
    real_temp: float = Field(..., description="Temperature for noise in real samples")
    n_images: int = Field(..., description="Number of images to log")
    show_images_steps: int = Field(..., description="Number of steps between logging resulting images")
    eval_steps: int = Field(..., description="Number of steps between evaluation")
    project_name: str = Field(..., description="Name of the project")
    total_iters: int = Field(..., description="Total number of iterations for training")


class SampleConfig(BaseModel):
    n_steps: int = Field(..., description="Number of steps for sampling")
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    step_type: str = Field(..., description="Type of step")
    track_ll: bool = Field(..., description="Whether to track log likelihood")
    track_states: bool = Field(..., description="Whether to track states")
    idx_start: int = Field(..., description="Starting index")


class ForwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    n_temps: int = Field(..., description="Number of temperatures")


class BackwardStatsConfig(BaseModel):
    n_samples: int = Field(..., description="Number of samples to generate")
    n_repeats: int = Field(..., description="Number of repeats")
    batch_size: int = Field(..., description="Batch size for stats computation")


class VariedDatasetStatsConfig(ForwardStatsConfig):
    dataset_names: tuple[str, ...]  = Field(
        ("mnist", "cifar10", "cifar100", "fashion_mnist"), description="Names of the datasets"
    )
    sample_fractions: tuple[float, ...] = Field((1.0, 0.1, 0.01), description="Sample fractions")


class FIDConfig(BaseModel):
    n_steps: list[int] = Field(..., description="Number of steps for sampling")
    noise_schedules: list[str] = Field(..., description="Noise schedules for diffusion")
    step_types: list[str] = Field(..., description="Step types for sampling")
    train: bool = Field(..., description="Whether to use train sample for reference")
    sample: bool = Field(..., description="Whether to sample images or use sampled")
    save_imgs: bool = Field(..., description="Whether to save generated images")

    @property
    def n_samples(self) -> int:
        return 10000 if self.train else 60000


class Config(BaseModel):
    diffusion: DiffusionConfig = Field(..., description="Diffusion configuration")
    data: DataConfig = Field(..., description="Data configuration")
    ddpm: DDPMConfig = Field(..., description="DDPM configuration")
    ddpm_training: DDPMTrainingConfig = Field(..., description="DDPM training configuration")
    gan: GANConfig = Field(..., description="GAN configuration")
    gan_training: GANTrainingConfig = Field(..., description="GAN training configuration")
    sample: SampleConfig = Field(..., description="Sample configuration")
    forward_stats: ForwardStatsConfig = Field(..., description="Forward statistics configuration")
    backward_stats: BackwardStatsConfig = Field(..., description="Backward statistics configuration")
    varied_dataset_stats: VariedDatasetStatsConfig = Field(
        ..., description="Varied dataset statistics configuration"
    )
    fid: FIDConfig = Field(..., description="FID configuration")

    @property
    def experiment_name(self) -> str:
        return "_".join([
            self.data.dataset_name,
            self.ddpm.model_name,
            self.diffusion.noise_schedule,
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
    def schedule_stats_path(self) -> str:
        match self.diffusion.noise_schedule:
            case "entropy":
                return self.forward_stats_path
            case "entropy_u":
                return self.forward_unbiased_stats_path
            case _:
                raise ValueError

    @property
    def fid_results_path(self) -> str:
        return f"results/{self.data.dataset_name}_{'train' if self.fid.train else 'test'}_fid.csv"

