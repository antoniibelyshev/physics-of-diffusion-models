from utils import extract_features_statistics, get_data_tensor, LeNet, compute_fid
from config import with_config, Config
from itertools import product
import numpy as np
import torch
from torch import from_numpy
from pytorch_fid import fid_score # type: ignore
from scipy.linalg import sqrtm # type: ignore
import pandas as pd


noise_schedules = ["linear_beta", "cosine", "flattening_temp_unbiased"]
n_steps_lst = [10, 100, 1000]


@with_config()
def main(config: Config) -> None:
    train_data = get_data_tensor(config)
    test_data = get_data_tensor(config, train=False)

    lenet = LeNet(test_data[0].numel(), 10).to("cuda")
    lenet.load_state_dict(torch.load(f"checkpoints/lenet_{config.data.dataset_name}.pth"))
    lenet.eval()

    mu_real, sigma_real = extract_features_statistics(test_data, lenet)
    mu_train, sigma_train = extract_features_statistics(train_data, lenet)

    mean_diff_term, cov_diff_term, fid = compute_fid(mu_real, sigma_real, mu_train, sigma_train)

    results = [{
        "schedule_type": "Train sample, no schedule",
        "n_steps": 0,
        "mean_diff_term": mean_diff_term,
        "cov_diff_term": cov_diff_term,
        "fid": fid,
    }]

    for n_steps, noise_schedule in product(n_steps_lst, noise_schedules):
        config.diffusion.noise_schedule = noise_schedule
        config.sample.n_steps = n_steps
        fake_data = from_numpy(np.load(config.samples_path)["x"])
        mu_fake, sigma_fake = extract_features_statistics(fake_data, lenet)
        mean_diff_term, cov_diff_term, fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

        print(f"Schedule: {noise_schedule}, number of sampling steps: {n_steps}.")
        print(f"Mean Difference Term: {mean_diff_term}")
        print(f"Covariance Difference Term: {cov_diff_term}")
        print(f"FID: {fid}")

        results.append({
            "noise_schedule": noise_schedule,
            "n_steps": n_steps,
            "mean_diff_term": mean_diff_term,
            "cov_diff_term": cov_diff_term,
            "fid": fid,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/fid_{config.data.dataset_name}.csv", index=False)


if __name__ == "__main__":
    main()
