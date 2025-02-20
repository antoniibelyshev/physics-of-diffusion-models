import torch
from torch import Tensor, nn
from torchmetrics.image.fid import FrechetInceptionDistance
from config import Config

from .lenet import LeNet
from .data import to_uint8


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fid = FrechetInceptionDistance(feature=2048).cuda()

    def forward(self, x: Tensor) -> Tensor:
        return self.fid.inception(to_uint8(x))


class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.lenet = LeNet(1024, 10)
        self.lenet.load_state_dict(torch.load("checkpoints/lenet_mnist.pth"))

    def forward(self, x: Tensor) -> Tensor:
        return self.lenet.features(x)



def get_feature_extractor(config: Config) -> nn.Module:
    match config.data.dataset_name:
        case "mnist":
            return LeNetFeatureExtractor
        case _:
            return InceptionV3FeatureExtractor()
