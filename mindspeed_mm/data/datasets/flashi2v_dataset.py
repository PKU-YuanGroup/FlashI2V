# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import random
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings
import copy
import sys

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS,
    PROMPT_MASK,
    START_FRAME,
    VIDEO,
)

from mindspeed_mm.data.datasets.wan_dataset import WanT2VDataset
from mindspeed_mm.data.data_utils.utils import MetaFileReader
from mindspeed_mm.data.data_utils.wan_utils import WanTextProcessor, WanVideoProcessor

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
