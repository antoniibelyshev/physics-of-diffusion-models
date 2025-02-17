from torch import nn, Tensor, cat
from config import Config


class GANDiscriminator(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        channels = [
            2 * config.data.obj_size[0],
            *(config.gan.base_channels_d * dim_mult for dim_mult in config.gan.dim_mults_d)
        ]
        self.conv_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for in_channels, out_channels in zip(channels, channels[1:])
            ],
            nn.Conv2d(channels[-1], channels[-1], kernel_size=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last_linear = nn.Linear(channels[-1], 1)

    def forward(self, imgs: Tensor, noisy_imgs: Tensor) -> Tensor:
        return self.last_linear(self.conv_blocks(cat([imgs, noisy_imgs], dim=1)).flatten(1)) # type: ignore
