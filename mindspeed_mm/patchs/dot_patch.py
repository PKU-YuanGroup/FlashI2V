from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from examples.qwen2vl.dot_product_attention import dot_product_attention_forward

mindspeed_args = get_mindspeed_args()
pm.register_patch('mindspeed.core.transformer.dot_product_attention.dot_product_attention_forward', 
    dot_product_attention_forward, force_patch=True)
pm.apply_patches()
