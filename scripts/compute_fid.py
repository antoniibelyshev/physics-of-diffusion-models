from utils import extract_features_statistics, get_data_tensor, LeNet, compute_fid
from config import with_config, Config
from itertools import product
import numpy as np
import torch
from torch import from_numpy
from pytorch_fid import fid_score # type: ignore
from scipy.linalg import sqrtm # type: ignore
import pandas as pd
from diffusion import get_and_save_samples


schedule_types = ["linear_beta", "cosine", "flattening_temp_unbiased"]
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

    for n_steps, schedule_type in product(n_steps_lst, schedule_types):
        config.ddpm.schedule_type = schedule_type
        config.sample.n_steps = n_steps
        # if schedule_type == "flattening_temp_unbiased":
        #     fake_data = get_and_save_samples(config, save=True)["x"]
        # else:
        #     fake_data = from_numpy(np.load(config.samples_path)["x"])
        fake_data = from_numpy(np.load(config.samples_path)["x"])
        mu_fake, sigma_fake = extract_features_statistics(fake_data, lenet)
        mean_diff_term, cov_diff_term, fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

        print(f"Schedule: {schedule_type}, number of sampling steps: {n_steps}.")
        print(f"Mean Difference Term: {mean_diff_term}")
        print(f"Covariance Difference Term: {cov_diff_term}")
        print(f"FID: {fid}")

        results.append({
            "schedule_type": schedule_type,
            "n_steps": n_steps,
            "mean_diff_term": mean_diff_term,
            "cov_diff_term": cov_diff_term,
            "fid": fid,
        })

        # fid_value = fid_score.calculate_fid_given_paths(
        #     [real_dir, fake_dir],
        #     batch_size=50,
        #     device='cuda' if torch.cuda.is_available() else 'cpu',
        #     dims=2048,
        # )
        # print(f"Schedule: {schedule_type}, number of sampling steps: {n_steps}. FID: {fid_value}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/fid_{config.data.dataset_name}.csv", index=False)


if __name__ == "__main__":
    main()
