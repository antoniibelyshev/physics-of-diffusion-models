from torch import Tensor, stack
import numpy as np
from tqdm import tqdm

from config import Config, with_config
from utils import compute_stats_traj_batch
from utils import get_data_tensor
from diffusion import get_samples


@with_config()
def main(config: Config) -> None:
    config.sample.n_samples = config.backward_stats.n_samples
    samples = get_samples(config)
    temp = samples["temp"]
    states = samples["states"]
    x_batched = states.reshape(-1, config.backward_stats.batch_size, *states.shape[1:])
    y = get_data_tensor(config)

    stats_lst: list[dict[str, Tensor]] = []
    for x in tqdm(x_batched):
        stats_lst.append(compute_stats_traj_batch(x * (1 + temp).sqrt().view(-1, *[1] * len(y.shape[1:])), y, temp))

    stats = {key: stack([stats[key] for stats in stats_lst], 1).mean(1) for key in stats_lst[0].keys()}
    np.savez("results/mnist_backward_stats.npz", **stats) # type: ignore


if __name__ == "__main__":
    main()
