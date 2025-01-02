from utils import MnistDataset
import torch
from statistics import compute_stats
import numpy as np


if __name__ == "__main__":
    data = MnistDataset()
    y = torch.stack([data[i]["images"] for i in range(len(data))], 0)
    temp = torch.linspace(-10, 10, 500).exp()
    np.savez("results/mnist_forward_stats.npz", temp=temp, **compute_stats(y, temp, 100, 10))
