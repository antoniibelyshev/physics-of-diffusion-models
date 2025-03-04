import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader
import wandb
from typing import Generator, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from .generator import GANGenerator
from .discriminator import GANDiscriminator
from .helper_functions import cross_entropy_loss, add_noise


class GANTrainer:
    def __init__(
            self,
            generator: GANGenerator,
            discriminator: GANDiscriminator,
            config: Config,
            compute_fid: Optional[Callable[[Tensor], float]] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
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
        self.n_images = cfg.n_images
        self.show_images_steps = cfg.show_images_steps
        self.eval_steps = cfg.eval_steps

        self.project_name = cfg.project_name
        self.compute_fid = compute_fid
        self.device = device

    @staticmethod
    def generator_loss(fake_logits: Tensor) -> Tensor:
        return cross_entropy_loss(fake_logits, 1)

    def discriminator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        return 0.5 * (cross_entropy_loss(real_logits, self.real_p) + cross_entropy_loss(fake_logits, self.fake_p))

    def generator_update(self, noisy_imgs: Tensor, fake_imgs: Tensor) -> float:
        self.optimizer_g.zero_grad()
        loss_g = self.generator_loss(self.discriminator(fake_imgs, noisy_imgs))
        loss_g.backward() # type: ignore
        self.optimizer_g.step()
        return loss_g.item()

    def discriminator_update(self, real_imgs: Tensor, noisy_imgs: Tensor, fake_imgs: Tensor) -> float:
        self.optimizer_d.zero_grad()
        loss_d = self.discriminator_loss(
            self.discriminator(add_noise(real_imgs, self.real_temp), noisy_imgs),
            self.discriminator(fake_imgs.detach(), noisy_imgs)
        )
        # loss_d1 = cross_entropy_loss(
        #     self.discriminator(add_noise(real_imgs, self.real_temp), noisy_imgs), self.real_p
        # )
        # loss_d1.backward() # type: ignore
        # self.optimizer_d.step()
        # self.optimizer_d.zero_grad()
        # loss_d2 = cross_entropy_loss(
        #     self.discriminator(fake_imgs.detach(), noisy_imgs), self.fake_p
        # )
        # loss_d2.backward() # type: ignore
        # self.optimizer_d.step()
        # return 0.5 * (loss_d1.item() + loss_d2.item())
        loss_d.backward() # type: ignore
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
        loss_g = self.generator_update(noisy_imgs, fake_imgs)
        for _ in range(self.n_iter_g - 1):
            loss_g = self.generator_update(noisy_imgs, self.generator(noisy_imgs))

        return loss_g, loss_d


    def train(
            self,
            real_data_generator: Generator[tuple[Tensor, ...], None, None],
            total_iters: int = 50000,
            test_data: Optional[Tensor] = None,
            eval_data_loaders: Optional[dict[str, DataLoader[Tensor]]] = None,
    ) -> None:
        wandb.init(project = self.project_name)

        eval_metrics: dict[str, float] = {}
        with tqdm(total=total_iters) as pbar:
            for iter_idx in range(total_iters + 1):
                real_imgs = next(real_data_generator)[0].to(self.device)
                loss_g, loss_d = self.train_step(real_imgs)
                wandb.log({
                    "Step": iter_idx,
                    "G_loss": loss_g,
                    "D_loss": loss_d,
                })

                if iter_idx % self.show_images_steps == 0:
                    self.show_images(test_data)

                if iter_idx % self.eval_steps == 0:
                    eval_metrics = self.eval(eval_data_loaders)

                pbar.update(1)
                pbar.set_postfix(g_loss=loss_g, d_loss=loss_d, **eval_metrics) # type: ignore

        wandb.finish()

    @torch.no_grad()
    def eval(self, eval_data_loaders: Optional[dict[str, DataLoader[Tensor]]]) -> dict[str, float]:
        fids: dict[str, float] = {}
        if eval_data_loaders is not None and self.compute_fid is not None:
            for name, eval_data_loader in eval_data_loaders.items():
                fake_imgs = torch.cat([self.generator(batch.to(self.device)).cpu() for batch in eval_data_loader])
                fids[f"{name} FID"] = self.compute_fid(fake_imgs)
            wandb.log(fids)

        self.generator.train()
        return fids

    @torch.no_grad()
    def show_images(self, test_data: Optional[Tensor]) -> None:
        self.generator.eval()
        if test_data is not None:
            real_imgs = test_data[:self.n_images]
            noisy_imgs = add_noise(real_imgs, self.temp)
            fake_imgs = self.generator(noisy_imgs.to(self.device)).detach().cpu()

            fig = plt.figure(figsize=(2 * self.n_images, 5))

            for i in range(self.n_images):
                plt.subplot(3, self.n_images, i + 1)
                plt.imshow(real_imgs[i, 0])
                plt.axis("off")

                plt.subplot(3, self.n_images, i + self.n_images + 1)
                plt.imshow(noisy_imgs[i, 0])
                plt.axis("off")

                plt.subplot(3, self.n_images, i + 2 * self.n_images + 1)
                plt.imshow(fake_imgs[i, 0])
                plt.axis("off")

            wandb.log({"Generated samples": wandb.Image(fig)})
            plt.close(fig)
        self.generator.train()
