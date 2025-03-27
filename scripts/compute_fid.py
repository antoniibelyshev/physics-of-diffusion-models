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
        *config.fid.varied_parameters.values()
    )
    for params in parameter_combinations:
        params_dict = dict(zip(config.fid.varied_parameters, params))
        for name, value in params_dict.items():
            setattr(config.sample, name, config.sample.model_fields[name].annotation(value))
        config.sample.n_samples = config.fid.n_samples
        if config.fid.sample:
            samples = get_samples(config)
            if config.fid.save_imgs:
                np.savez(config.samples_path, **samples) # type: ignore
            x = samples["x"]
        else:
            x = from_numpy(np.load(config.samples_path)["x"][:config.fid.n_samples])
        fid = compute_fid(x)
        results_dict = {**{"fid": fid}, **params_dict}
        print(*[f"{key}: {value}" for key, value in results_dict.items()], sep=", ")
        fids.append(results_dict)

    fids_df = pd.DataFrame(fids)
    fids_df.to_csv(config.fid_results_path)


if __name__ == "__main__":
    main()
