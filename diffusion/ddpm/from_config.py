from torch import load, compile
import torch
from config import Config
from .ddpm import DDPM
from ..scheduler import scheduler_from_config
from utils import get_data_tensor, get_diffusers_pipeline

def ddpm_from_config(config: Config, pretrained: bool = False) -> DDPM:
    from .unet import DDPMUnet, set_processor_recursively
    from .true_model import DDPMTrue
    from .diffusers_model import DDPMDiffusers
    from diffusers.models.attention_processor import AttnProcessor2_0
    
    scheduler = scheduler_from_config(config)
    model_name = config.ddpm.model_name
    parametrization = config.ddpm.parametrization

    if model_name == "unet":
        ddpm = DDPMUnet(
            scheduler=scheduler,
            parametrization=parametrization,
            image_size=config.dataset_config.image_size,
            unet_config=config.ddpm.unet_config,
        )
        if pretrained:
            checkpoint = load(config.ddpm_checkpoint_path)
            ddpm.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return ddpm
    
    if model_name == "true":
        return DDPMTrue(
            scheduler=scheduler,
            parametrization=parametrization,
            train_data=get_data_tensor(config),
        )
    
    if model_name == "diffusers":
        pipeline = get_diffusers_pipeline(config)
        set_processor_recursively(pipeline.unet, AttnProcessor2_0)  # type: ignore
        torch.set_float32_matmul_precision("high")
        unet = compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)  # type: ignore
        time_scale = pipeline.scheduler.timesteps.max()  # type: ignore
        return DDPMDiffusers(
            scheduler=scheduler,
            parametrization="eps",
            unet=unet,
            time_scale=time_scale,
        )
    
    raise ValueError(f"Unknown model name: {model_name}")
