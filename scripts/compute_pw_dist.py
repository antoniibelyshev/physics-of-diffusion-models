from arrow import get
from utils import get_data_tensor, compute_pw_dist_sqr
import numpy as np
from base_config import BaseConfig
from config import with_config


@with_config()
def main(config: BaseConfig) -> None:
    y = get_data_tensor(config)
    pw_dist = compute_pw_dist_sqr(y.cuda()).sqrt().cpu()
    np.save(f"results/{config.data.dataset_name}_dist.npy", pw_dist)


if __name__ == "__main__":
    main()
