import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
from torch_ema import ExponentialMovingAverage # type: ignore
from typing import Callable, Generator, Optional
from tqdm import trange
import wandb
import copy
import os

from config import Config
from .ddpm import DDPM
from .ddpm_sampling import DDPMSampler
from utils import get_default_device, get_compute_fid, to_uint8


class DDPMTrainer:
    def __init__(
        self,
        ddpm: DDPM,
        ema_decay: float,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        warmup_steps: int,
        total_iters: int,
        grad_clip: float,
        project_name: str,
        experiment_name: str,
        compute_fid_fn: Callable[[Tensor], float],
        device: str = get_default_device(),
    ) -> None:
        self.ddpm = ddpm.to(device)
        self.ema = ExponentialMovingAverage(ddpm.parameters(), decay=ema_decay)
        self.device = device
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            self.ddpm.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

        self.scheduler: Optional[LambdaLR] = None
        if warmup_steps > 0:
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(total_iters - current_step) / float(max(1, total_iters - warmup_steps))
                )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.compute_fid = compute_fid_fn

    @classmethod
    def from_config(cls, config: Config, ddpm: DDPM, device: str = get_default_device()) -> "DDPMTrainer":
        return cls(
            ddpm=ddpm,
            ema_decay=config.ddpm_training.ema_decay,
            learning_rate=config.ddpm_training.learning_rate,
            weight_decay=config.ddpm_training.weight_decay,
            betas=config.ddpm_training.betas,
            warmup_steps=config.ddpm_training.warmup_steps,
            total_iters=config.ddpm_training.total_iters,
            grad_clip=config.ddpm_training.grad_clip,
            project_name=config.project_name,
            experiment_name=config.experiment_name,
            compute_fid_fn=get_compute_fid(config),
            device=device,
        )

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
        torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), self.grad_clip)
        self.optimizer.step()
        self.ema.update(self.ddpm.parameters())

        # Update learning rate scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()

    def evaluate(self, step: int, config: Config) -> None:
        """
        Evaluate the model:
        1. Switch to EMA checkpoint
        2. Sample 25 images for visualization
        3. Upload the images to wandb
        4. Sample 50k images for FID computation
        5. Compute the FID score
        6. Upload the FID score to wandb
        7. Save checkpoint
        8. Switch back from the EMA checkpoint
        """
        self.ddpm.eval()
        self.switch_to_ema()

        # Create a temporary config for sampling
        eval_config = copy.deepcopy(config)
        eval_config.sample.step_type = "ddim"
        eval_config.sample.n_steps = 100
        eval_config.sample.noise_schedule_type = config.ddpm.noise_schedule_type

        # Sample 25 images for visualization
        eval_config.sample.n_samples = 25
        sampler = DDPMSampler.from_config(eval_config, ddpm=self.ddpm)
        samples = sampler.sample()

        images = samples["x"]
        images_wandb = [wandb.Image(to_uint8(image).permute(1, 2, 0).numpy()) for image in images]
        wandb.log({"samples": images_wandb}, step=step)

        eval_config.sample.n_samples = config.dataset_config.fid_samples

        sampler = DDPMSampler.from_config(eval_config, ddpm=self.ddpm)
        samples = sampler.sample()

        # Compute FID
        fid_score = self.compute_fid(samples["x"])

        # Log FID to wandb
        wandb.log({f"fid 100 steps": fid_score}, step=step)

        # Save checkpoint
        self.save_checkpoint(step, config)

        # Restore training state
        self.switch_back_from_ema()
        self.ddpm.train()

    def save_checkpoint(self, step: int, config: Config) -> None:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "step": step,
            "model_state_dict": self.ddpm.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, f"{config.checkpoint_dir}/step_{step}.pth")
        torch.save(checkpoint, config.ddpm_checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ddpm.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]

    def train(self, train_generator: Generator[tuple[Tensor, ...], None, None], total_iters: int, config: Config) -> None:
        checkpoint_path = config.ddpm_checkpoint_path
        start_step = 0
        if os.path.exists(checkpoint_path):
            start_step = self.load_checkpoint(checkpoint_path)
        
        wandb.init(
            project = self.project_name,
            name = self.experiment_name,
            config = config.ddpm_training.__dict__,
            resume = "allow",
            id = self.experiment_name,
        )

        self.ddpm.train()

        with trange(start_step + 1, 1 + total_iters) as pbar:
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
                }, step=iter_idx)

                self.optimizer_logic(loss)

                pbar.set_postfix(loss=loss.item(), lr=current_lr)

                # Evaluate every 10k steps
                if iter_idx % config.ddpm_training.eval_steps == 0:
                    self.evaluate(iter_idx, config)

        self.ddpm.eval()
        self.switch_to_ema()

        wandb.finish()
