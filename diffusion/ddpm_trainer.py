from torch import Tensor
import torch
from torch_ema import ExponentialMovingAverage # type: ignore
from torch.nn.functional import mse_loss
from typing import Generator
from tqdm import trange
from .ddpm import DDPM
from config import Config
import wandb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMTrainer:
    def __init__(self, config: Config, ddpm: DDPM, device: torch.device = DEVICE) -> None:
        self.ddpm = ddpm
        self.ddpm.to(device)
        self.ema = ExponentialMovingAverage(ddpm.parameters(), decay=0.999)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.ddpm.parameters(),
            lr = config.ddpm_training.learning_rate,
            weight_decay = config.ddpm_training.weight_decay,
        )

        self.project_name = config.project_name
        self.experiment_name = config.experiment_name

    def switch_to_ema(self) -> None:
        self.ema.store(self.ddpm.parameters())
        self.ema.copy_to(self.ddpm.parameters())

    def switch_back_from_ema(self) -> None:
        self.ema.restore(self.ddpm.parameters())

    def calc_loss(self, x0: Tensor) -> Tensor:
        t, eps, xt = self.ddpm.dynamic(x0)

        pred = self.ddpm(xt, t)
        target = eps if self.ddpm.parametrization == "eps" else x0
        return mse_loss(target, pred)

    def optimizer_logic(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward() # type: ignore
        self.optimizer.step()
        self.ema.update(self.ddpm.parameters())

    def train(
        self,
        train_generator: Generator[Tensor, None, None],
        total_iters: int = 2500,
    ) -> None:
        wandb.init(project = self.project_name, name = self.experiment_name)

        self.ddpm.train()
        
        with trange(1, 1 + total_iters) as pbar:
            for iter_idx in pbar:
                batch = next(train_generator).to(self.device)
                loss = self.calc_loss(batch)

                wandb.log({'iteration': iter_idx, 'loss': loss.item()})

                self.optimizer_logic(loss)

                pbar.set_postfix(loss=loss.item())

        self.ddpm.eval()
        self.switch_to_ema()

        wandb.finish()
