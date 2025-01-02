from utils import MnistDataset
import torch
from statistics import compute_stats, compute_stats_traj_batch
import numpy as np
from tqdm import tqdm
from torch import Tensor


if __name__ == "__main__":
    data = MnistDataset()
    y = torch.stack([data[i]["images"] for i in range(len(data))], 0)

    sde_samples = np.load("results/sde_samples_unet.npz")
    temp = torch.tensor(sde_samples["temp"])
    x_sample = torch.tensor(sde_samples["states"])
    x_batched = x_sample.reshape(-1, 20, *x_sample.shape[1:])

    stats_lst: list[dict[str, Tensor]] = []
    for x in tqdm(x_batched):
        stats_lst.append(compute_stats_traj_batch(x * (1 + temp).sqrt().view(-1, *[1] * len(y.shape[1:])), y, temp))

    stats = {key: torch.stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}
    np.savez("results/mnist_backward_stats.npz", temp=temp, **stats)
