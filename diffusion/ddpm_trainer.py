import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
from torch_ema import ExponentialMovingAverage # type: ignore
from typing import Generator, Optional
from tqdm import trange
import wandb
import copy

from config import Config
from .ddpm import DDPM
from .ddpm_sampling import DDPMSampler
from utils import get_default_device, get_compute_fid


class DDPMTrainer:
    def __init__(self, config: Config, ddpm: DDPM, device: str = get_default_device()) -> None:
        self.config = config
        self.ddpm = ddpm
        self.ddpm.to(device)
        ema_decay = config.ddpm_training.ema_decay
        self.ema = ExponentialMovingAverage(ddpm.parameters(), decay=ema_decay)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.ddpm.parameters(),
            lr = config.ddpm_training.learning_rate,
            weight_decay = config.ddpm_training.weight_decay,
            betas = config.ddpm_training.betas,
        )

        warmup_steps = config.ddpm_training.warmup_steps
        total_steps = config.ddpm_training.total_iters
        self.scheduler: Optional[LambdaLR] = None
        if warmup_steps > 0:
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
                )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.project_name = config.project_name
        self.experiment_name = config.experiment_name

        self.compute_fid = get_compute_fid(config)

    def switch_to_ema(self) -> None:
        self.ema.store(self.ddpm.parameters())
        self.ema.copy_to(self.ddpm.parameters())

    def switch_back_from_ema(self) -> None:
        self.ema.restore(self.ddpm.parameters())

    def calc_loss(self, x0: Tensor) -> Tensor:
        tau, eps, xt = self.ddpm.dynamic(x0)

        pred = self.ddpm(xt, tau)
        target = eps if self.ddpm.parametrization == "eps" else x0
        return mse_loss(target, pred)

    def optimizer_logic(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward() # type: ignore
        self.optimizer.step()
        self.ema.update(self.ddpm.parameters())

        # Update learning rate scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()

    def evaluate(self, step: int) -> None:
        """
        Evaluate the model:
        1. Switch to EMA checkpoint
        2. Sample 25 images for visualization
        3. Upload the images to wandb
        4. Sample 50k images for FID computation
        5. Compute the FID score
        6. Upload the FID score to wandb
        7. Switch back from the EMA checkpoint
        """
        # Store training state
        training_mode = self.ddpm.training
        self.ddpm.eval()
        self.switch_to_ema()

        # Save checkpoint
        torch.save(self.ddpm.state_dict(), self.config.ddpm_checkpoint_path)
        torch.save(self.ddpm.state_dict(), f"{self.config.ddpm_checkpoint_path[:-4]}_step_{step}.pth")

        # Create a temporary config for sampling
        eval_config = copy.deepcopy(self.config)
        eval_config.sample.track_states = False
        eval_config.sample.track_ll = False
        eval_config.sample.step_type = "ddim"
        eval_config.sample.n_steps = 10

        # Sample 25 images for visualization
        eval_config.sample.n_samples = 25
        sampler = DDPMSampler(eval_config)
        samples = sampler.sample()

        # Convert samples to uint8 and log to wandb
        images = samples["x"]  # Shape: [25, channels, height, width]
        images_grid = images.view(5, 5, *images.shape[1:]).permute(0, 3, 1, 4, 2).reshape(5 * images.shape[2], 5 * images.shape[3], -1).numpy()
        images_wandb = wandb.Image(images_grid)
        wandb.log({"samples": images_wandb})

        # Sample 50k images for FID computation
        eval_config.sample.n_samples = self.config.dataset_config.fid_samples
        eval_config.sample.batch_size = 1000  # Adjust batch size for efficiency
        sampler = DDPMSampler(eval_config)
        samples = sampler.sample()

        # Compute FID
        fid_score = self.compute_fid(samples["x"])

        # Log FID to wandb
        wandb.log({"fid": fid_score})

        # Restore training state
        self.switch_back_from_ema()
        if training_mode:
            self.ddpm.train()
        else:
            self.ddpm.eval()

    def train(self, train_generator: Generator[tuple[Tensor, ...], None, None], total_iters: int) -> None:
        wandb.init(project = self.project_name, name = self.experiment_name)

        self.ddpm.train()

        with trange(1, 1 + total_iters) as pbar:
            for iter_idx in pbar:
                batch = next(train_generator)[0].to(self.device)
                loss = self.calc_loss(batch)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log metrics to wandb
                wandb.log({
                    'iteration': iter_idx, 
                    'loss': loss.item(),
                    'learning_rate': current_lr
                })

                self.optimizer_logic(loss)

                pbar.set_postfix(loss=loss.item(), lr=current_lr)

                # Evaluate every 10k steps
                if iter_idx % self.config.ddpm_training.eval_steps == 0:
                    self.evaluate(iter_idx)

        self.ddpm.eval()
        self.switch_to_ema()

        wandb.finish()
