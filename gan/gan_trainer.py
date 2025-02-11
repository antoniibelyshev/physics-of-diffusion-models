import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader
import wandb
from typing import Generator, Optional, Callable
from tqdm import tqdm

from config import Config
from .generator import GANGenerator
from .discriminator import GANDiscriminator
from .helper_functions import cross_entropy_loss, add_noise


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GANTrainer:
    def __init__(
            self,
            generator: GANGenerator,
            discriminator: GANDiscriminator,
            config: Config,
            compute_fid: Optional[Callable[[Tensor], float]] = None,
            device = DEVICE,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        cfg = config.gan_training

        self.optimizer_g = optim.Adam(
            generator.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999), weight_decay=cfg.weight_decay_g
        )
        self.optimizer_d = optim.Adam(
            discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999), weight_decay=cfg.weight_decay_d
        )

        self.n_iter_g = cfg.n_iter_g
        self.n_iter_d = cfg.n_iter_d
        self.real_p = cfg.real_p
        self.fake_p = cfg.fake_p
        self.temp = cfg.temp
        self.real_temp = cfg.real_temp
        self.eval_steps = cfg.eval_steps

        self.project_name = cfg.project_name
        self.compute_fid = compute_fid
        self.device = device

    @staticmethod
    def generator_loss(logits: Tensor) -> Tensor:
        return cross_entropy_loss(logits, 1)

    def discriminator_loss(self, real_logits: Tensor, fake_logits) -> Tensor:
        return 0.5 * (cross_entropy_loss(real_logits, self.real_p) + cross_entropy_loss(fake_logits, self.fake_p))

    def generator_update(self, fake_imgs: Tensor, noisy_imgs: Tensor) -> float:
        self.optimizer_g.zero_grad()
        loss_g = self.generator_loss(self.discriminator(fake_imgs, noisy_imgs))
        loss_g.backward()
        self.optimizer_g.step()
        return loss_g.item()

    def discriminator_update(self, real_imgs: Tensor, noisy_imgs: Tensor, fake_imgs: Tensor) -> float:
        self.optimizer_d.zero_grad()
        loss_d = self.discriminator_loss(
            self.discriminator(add_noise(real_imgs, self.real_temp), noisy_imgs),
            self.discriminator(fake_imgs.detach(), noisy_imgs)
        )
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d.item()

    def train_step(self, real_imgs: Tensor) -> tuple[float, float]:
        noisy_imgs = add_noise(real_imgs, self.temp)
        fake_imgs = self.generator(noisy_imgs)

        # update discriminator
        loss_d = self.discriminator_update(real_imgs, noisy_imgs, fake_imgs)
        for _ in range(self.n_iter_d - 1):
            loss_d = self.discriminator_update(real_imgs, noisy_imgs, fake_imgs)

        # Generator update
        loss_g = self.generator_update(fake_imgs, noisy_imgs)
        for _ in range(self.n_iter_g - 1):
            loss_g = self.generator_update(fake_imgs. noisy_imgs)

        return loss_g, loss_d


    def train(
            self,
            real_data_generator: Generator[Tensor, None, None],
            total_iters: int = 50000,
            eval_data_loaders: Optional[dict[str, DataLoader[tuple[Tensor, ...]]]] = None,
    ):
        wandb.init(project = self.project_name)

        eval_metrics: dict[str, float] = {}
        with tqdm(total=total_iters) as pbar:
            for iter_idx in range(1, total_iters + 1):
                real_imgs = next(real_data_generator).to(self.device)
                loss_g, loss_d = self.train_step(real_imgs)
                wandb.log({
                    "Step": iter_idx,
                    "G_loss": loss_g,
                    "D_loss": loss_d,
                })

                if iter_idx % self.eval_steps == 0:
                    eval_metrics = self.eval(eval_data_loaders)

                pbar.update(1)
                pbar.set_postfix(g_loss=loss_g, d_loss=loss_d, **eval_metrics)

        wandb.finish()

    @torch.no_grad()
    def eval(self, eval_data_loaders: Optional[dict[str, DataLoader[tuple[Tensor, ...]]]]) -> dict[str, float]:
        if eval_data_loaders is not None and self.compute_fid is not None:
            self.generator.eval()
            fids = {
                f"{name} FID": self.compute_fid(torch.cat([self.generator(batch.to(self.device)).cpu() for batch, in eval_data_loader]))
                for name, eval_data_loader in eval_data_loaders.items()
            }
            wandb.log(fids)
            self.generator.train()
            return fids
        return {}
