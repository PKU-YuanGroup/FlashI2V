import torch
from flashi2v.utils.constant import VIDEO, PROMPT_IDS, PROMPT_MASK, START_FRAME

class WanDataCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        video = torch.stack([i[VIDEO] for i in batch]) if batch[0][VIDEO] is not None else None # in evaluation mode, we have no video gt.
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

data_collator = {
    'wan_t2v': WanDataCollator,
    'flashi2v': FlashI2VDataCollator
}