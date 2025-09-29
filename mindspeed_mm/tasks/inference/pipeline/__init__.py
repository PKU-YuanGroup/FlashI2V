__all__ = ["sora_pipeline_dict"]

from mindspeed_mm.tasks.inference.pipeline.wan_pipeline import WanPipeline
from mindspeed_mm.tasks.inference.pipeline.flashi2v_pipeline import FlashI2VPipeline
sora_pipeline_dict = {
    "WanPipeline": WanPipeline,  
    "FlashI2VPipeline": FlashI2VPipeline,
}

