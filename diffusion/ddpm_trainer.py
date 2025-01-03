from torch import Tensor
import torch
from torch_ema import ExponentialMovingAverage # type: ignore
import torch.nn.functional as F
from typing import Generator
from tqdm import trange
from .ddpm import DDPM
from .diffusion_utils import dict_to_device
import wandb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMTrainer:
    def __init__(
        self,
        ddpm: DDPM,
        device: torch.device = DEVICE,
        im_name: str = "images",
    ):
        self.ddpm = ddpm
        self.ddpm.to(device)
        self.ema = ExponentialMovingAverage(ddpm.parameters(), decay=0.999)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.ddpm.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )

        self.im_name = im_name

    def switch_to_ema(self) -> None:
        self.ema.store(self.ddpm.parameters())
        self.ema.copy_to(self.ddpm.parameters())

    def switch_back_from_ema(self) -> None:
        self.ema.restore(self.ddpm.parameters())

    def calc_loss(self, x0: Tensor) -> Tensor:
        t = self.ddpm.dynamic.sample_time_on_device(batch_size=x0.shape[0], device=self.device)
        d = self.ddpm.dynamic({"x0": x0, "t": t})
        xt = d["xt"]
        eps = d["eps"]

        res = self.ddpm(xt, t)
        target = eps if self.ddpm.parametrization == "eps" else x0
        return F.mse_loss(target, res)

    def optimizer_logic(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward() # type: ignore
        self.optimizer.step()
        self.ema.update(self.ddpm.parameters())

    def train(
        self,
        train_generator: Generator[dict[str, Tensor], None, None],
        total_iters: int = 2500,
        project_name: str = 'discrete_time_ddpm',
        experiment_name: str = 'mnist_baseline',
    ) -> None:
        wandb.init(project=project_name, name=experiment_name)

        self.ddpm.train()
        
        with trange(1, 1 + total_iters) as pbar:
            for iter_idx in pbar:
                batch = next(train_generator)
                batch = dict_to_device(batch, device=self.device)

                loss = self.calc_loss(x0=batch[self.im_name])

                wandb.log({'iteration': iter_idx, 'loss': loss.item()})

                self.optimizer_logic(loss)

                pbar.set_postfix(loss=loss.item())

        self.ddpm.eval()
        self.switch_to_ema()

        wandb.finish()
