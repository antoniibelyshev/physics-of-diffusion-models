from config import Config, with_config
from utils import get_compute_fid
from itertools import product
from diffusion import get_samples
import pandas as pd


@with_config()
def main(config: Config) -> None:
    config.sample.track_ll = False
    config.sample.track_states = False
    config.sample.idx_start = -1

    compute_fid = get_compute_fid(config)
    fids = pd.DataFrame(index = config.fid.n_steps, columns = config.fid.noise_schedules)
    for n_steps, noise_schedule in product(config.fid.n_steps, config.fid.noise_schedules):
        config.sample.n_steps = n_steps
        config.diffusion.noise_schedule = noise_schedule
        samples = get_samples(config)["x"]
        fids.loc[n_steps, noise_schedule] = compute_fid(samples)

    fids.to_csv(config.fid.results_path)


if __name__ == "__main__":
    main()
