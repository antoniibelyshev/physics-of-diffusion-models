import torch
from torch import from_numpy
import numpy as np
import argparse
from config import load_config
from utils import get_data_tensor, extract_features_statistics, compute_fid, LeNet

def main(samples_path: str) -> None:
    config = load_config()
    train_data = get_data_tensor(config)
    
    lenet = LeNet(1024, 10).cuda()
    lenet.load_state_dict(torch.load("checkpoints/lenet_mnist.pth"))
    lenet.eval()

    mu, sigma = extract_features_statistics(train_data.cuda(), lenet)
    x = torch.clip(from_numpy(np.load(samples_path)["x"]), 0, 1)
    mu_diff, sigma_diff = extract_features_statistics(x.cuda(), lenet)

    # Compute FID
    fid_score = compute_fid(mu, sigma, mu_diff, sigma_diff + 1e-7)[2]
    print(f"FID Score: {fid_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between training data and diffusion samples.")
    parser.add_argument("samples_path", type=str, help="Path to the diffusion samples (.npz file)")
    args = parser.parse_args()

    main(args.samples_path)
