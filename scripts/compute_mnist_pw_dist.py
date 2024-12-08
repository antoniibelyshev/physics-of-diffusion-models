from utils import MnistDataset, compute_pw_dist_sqr
import torch
import numpy as np


if __name__ == "__main__":
    mnist_data = MnistDataset()
    tensor_data = torch.stack([mnist_data[i]["images"].view(-1) for i in range(len(mnist_data))])
    pw_dist = compute_pw_dist_sqr(tensor_data.cuda()).sqrt().cpu()
    np.save("results/mnist_pw_dist.npy", pw_dist.numpy())
