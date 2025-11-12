
import os
import torch    
import random
from concurrent.futures import ThreadPoolExecutor
import copy

from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from flashi2v.utils.constant import VIDEO, PROMPT_IDS, PROMPT_MASK, START_FRAME
from flashi2v.data.utils.utils import LMDBReader
from flashi2v.data.datasets.t2v_dataset import WanT2VDataset, T2VRandomDataset

FlashI2VOutputData = {
    PROMPT_IDS: None,
    PROMPT_MASK: None,
    START_FRAME: None,
    VIDEO: None,
}

class FlashI2VDataset(WanT2VDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(FlashI2VOutputData)
        meta_info = self.dataset_reader.getitem(index)
        text = meta_info["cap"]
        video_path = meta_info["path"]

        drop_text = False
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            drop_text = True

        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text, drop=drop_text)
            
        orig_video = self.get_video_data(video_path, meta_info)
        examples[VIDEO] = orig_video
        examples[START_FRAME] = orig_video[:, 0:1, :, :].clone()

        return examples

    def get_text_data(self, text, drop=False):
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)][0]
        if drop:
            text = ""
        prompt_input_ids, prompt_mask = self.text_processor(text)
        return prompt_input_ids, prompt_mask
        

class I2VRandomDataset(T2VRandomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(FlashI2VOutputData)
        text = ""
        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text)
        orig_video = torch.randn(3, self.sample_num_frames, self.sample_height, self.sample_width)
        examples[VIDEO] = orig_video
        examples[START_FRAME] = orig_video[:, 0:1, :, :].clone()
        return examples

dataset = {
    'flashi2v': FlashI2VDataset,
    'i2v_random': I2VRandomDataset
}