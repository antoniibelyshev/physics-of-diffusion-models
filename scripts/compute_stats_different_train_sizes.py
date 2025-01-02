from utils import get_samples_tensor
import torch
from statistics import compute_stats
import numpy as np


cfg = {
    "datasets": ["mnist", "cifar10", "cifar100", "fashion_mnist"],
}


if __name__ == "__main__":
    temp = torch.linspace(-10, 10, 500).exp()
    for dataset in cfg["datasets"]:
        y = get_samples_tensor(dataset)

        for size in [len(y), len(y) // 10, len(y) // 100]:
            filename = f"results/{dataset}_forward_stats_{size}.npz"
            train_samples = y[np.random.choice(range(len(y)), size=size, replace=False)]
            np.savez(filename, temp=temp, **compute_stats(train_samples, temp, 100, 10))
