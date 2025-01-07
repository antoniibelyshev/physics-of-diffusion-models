from pydantic import model_validator
from typing_extensions import Self
from typing import Any

from base_config import BaseDataConfig, BaseConfig
from utils import get_obj_size


class DataConfig(BaseDataConfig):
    @model_validator(mode="before")
    @classmethod
    def check_obj_size(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["obj_size"] = data.get("obj_size") or get_obj_size(data["dataset_name"])
        return data


class Config(BaseConfig):
    data: DataConfig

    @model_validator(mode="after")
    def compute_paths(self) -> Self:
        experiment_name = self.experiment_name
        if not experiment_name:
            experiment_name = f"{self.data.dataset_name}_{self.ddpm.model_name}_{self.ddpm_training.total_iters}"
            self.experiment_name = experiment_name
        if not self.checkpoint_path:
            self.checkpoint_path = f"checkpoints/{experiment_name}.pth"
        if not self.samples_path:
            f"results/{experiment_name}_{self.sample.kwargs['step_type']}_samples.npz"
        if not self.samples_from_timestamp_path and self.sample.timestamp is not None:
            f"results/{experiment_name}_{self.sample.kwargs['step_type']}_samples_from_timestamp_{self.sample.timestamp}.npz"
        if not self.forward_stats_path:
            self.forward_stats_path = f"results/{self.data.dataset_name}_forward_stats.npz"
        if not self.backward_stats_path:
            self.backward_stats_path = f"results/{experiment_name}_backward_stats.npz"
        return self
