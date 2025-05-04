from config import Config
from utils import get_compute_fid, with_config
from itertools import product
from diffusion import get_samples
import pandas as pd
from typing import Any
import numpy as np
from torch import from_numpy


@with_config(parse_args=(__name__ == "__main__"))
def main(config: Config) -> None:
    config.sample.track_ll = False
    config.sample.track_states = False

    compute_fid = get_compute_fid(config)
    fids: list[dict[str, Any]] = []
    parameter_combinations = product(
        config.fid.n_steps,
        config.fid.noise_schedule_type,
        config.fid.min_temp,
    )
    for n_steps, noise_schedule_type, min_temp in parameter_combinations:
        config.sample.n_steps = n_steps
        config.sample.noise_schedule_type = noise_schedule_type
        config.entropy_schedule.min_temp = min_temp
        config.sample.n_samples = config.dataset_config.fid_samples
        if config.fid.sample:
            samples = get_samples(config)
            x = samples["x"]
        else:
            x = from_numpy(np.load(config.samples_path)["x"][:config.dataset_config.fid_samples])
        fid = compute_fid(x)
        results_dict = {
            "fid": fid,
            "n_steps": n_steps,
            "noise_schedule_type": noise_schedule_type,
            "min_temp": min_temp
        }
        print(*[f"{key}: {value}" for key, value in results_dict.items()], sep=", ")
        fids.append(results_dict)

    fids_df = pd.DataFrame(fids)
    fids_df.to_csv(config.fid_results_path)


if __name__ == "__main__":
    main()
