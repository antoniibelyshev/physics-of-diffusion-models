from config import Config, with_config
from utils import get_compute_fid
from itertools import product
from diffusion import get_samples
import pandas as pd
from typing import Any
import numpy as np
from torch import from_numpy


@with_config()
def main(config: Config) -> None:
    config.sample.track_ll = False
    config.sample.track_states = False

    compute_fid = get_compute_fid(config)
    fids: list[dict[str, Any]] = []
    parameter_combinations = product(config.fid.n_steps, config.fid.noise_schedule_types, config.fid.step_types)
    for n_steps, noise_schedule_type, step_type in parameter_combinations:
        config.sample.n_steps = n_steps
        config.sample.n_samples = config.fid.n_samples
        config.diffusion.noise_schedule_type = noise_schedule_type
        config.sample.step_type = step_type
        if config.fid.sample:
            samples = get_samples(config)
            if config.fid.save_imgs:
                np.savez(config.samples_path, **samples) # type: ignore
            x = samples["x"]
        else:
            x = from_numpy(np.load(config.samples_path)["x"][:config.fid.n_samples])
        fid = compute_fid(x)
        print(f"n_steps: {n_steps}, noise_schedule: {noise_schedule_type}, step_type: {step_type}, fid: {fid:.2f}")
        fids.append({"n_steps": n_steps, "noise_schedule": noise_schedule_type, "step_type": step_type, "fid": fid})

    fids_df = pd.DataFrame(fids)
    fids_df.to_csv(config.fid_results_path)


if __name__ == "__main__":
    main()
