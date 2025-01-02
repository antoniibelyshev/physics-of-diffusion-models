from .data import (
    MnistDataset as MnistDataset,
    get_data_generator as get_data_generator,
    get_samples_tensor as get_samples_tensor,
)
from .distance import (
    compute_pw_dist_sqr as compute_pw_dist_sqr,
    norm_sqr as norm_sqr,
)
from .derivatives import finite_diff_derivative as finite_diff_derivative
from .synthetic_datasets import (
    generate_simplex as generate_simplex,
    generate_cross_polytope as generate_cross_polytope,
    sample_on_hypersphere as sample_on_hypersphere,
    generate_dataset as generate_dataset,
)
from .utils import dict_map as dict_map
