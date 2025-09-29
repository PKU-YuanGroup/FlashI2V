from dataclasses import dataclass
from typing import Dict, Sequence, List, Union, Tuple
import math
from collections import Counter
import random
import warnings

import numpy as np
import torch
from torch.nn import functional as F
from megatron.training import get_args
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS
from mindspeed_mm.data.data_utils.constants import (
    PROMPT,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    START_FRAME,
    VIDEO_MASK,
    MASKED_VIDEO,
    INPUT_MASK,
    FILE_INFO
)
def collate_fn_default(batch):
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        input_ids = torch.cat(input_ids, dim=-1)
        use_mask = True
    elif "mask" in batch[0] and isinstance(batch[0]["mask"], torch.Tensor):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        masks = torch.cat(masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["input_ids"] = input_ids

    return ret

class WanDataCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        video = torch.stack([i[VIDEO] for i in batch])
        prompt_ids = torch.cat([i[PROMPT_IDS] for i in batch])
        prompt_mask = torch.cat([i[PROMPT_MASK] for i in batch]) if batch[0][PROMPT_MASK] is not None else None

        return {
            VIDEO: video,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask
        }

class FlashI2VDataCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        video = torch.stack([i[VIDEO] for i in batch]) if batch[0][VIDEO] is not None else None # in evaluation mode, we have no video gt.
        start_frame = torch.stack([i[START_FRAME] for i in batch])
        prompt_ids = torch.cat([i[PROMPT_IDS] for i in batch])
        prompt_mask = torch.cat([i[PROMPT_MASK] for i in batch]) if batch[0][PROMPT_MASK] is not None else None
        
        return {
            VIDEO: video,
            START_FRAME: start_frame,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
        }

DATA_COLLATOR = {
    'default': collate_fn_default,
    'wan': WanDataCollator,
    'flashi2v': FlashI2VDataCollator,
}
