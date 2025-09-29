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
    VIDEO,
)

from mindspeed_mm.data.data_utils.utils import MetaFileReader
from mindspeed_mm.data.data_utils.wan_utils import WanTextProcessor, WanVideoProcessor

T2VOutputData = {
    PROMPT_IDS: [],
    PROMPT_MASK: [],
    VIDEO: [],
}


class WanT2VDataset(Dataset):

    def __init__(
        self,
        metafile_or_dir_path,
        text_tokenizer_path,
        parquet_cache_row_group_size=1024,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=16,
        sample_stride=None,
        text_drop_ratio=0.1,
        tokenizer_max_length=512,
        return_prompt_mask=True,
        transform_pipeline=None,
        **kwargs,
    ):
        self.dataset_reader = MetaFileReader.create("LMDBReader", metafile_or_dir_path, parquet_cache_row_group_size=parquet_cache_row_group_size)
        self.data_length = len(self.dataset_reader)
        print(f'Build WanT2VDataset, data length: {self.data_length}...')
        
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames

        self.train_fps = train_fps
        self.sample_stride = sample_stride

        self.sample_mode = None
        if self.train_fps is not None:
            print(f"Using train_fps mode, train_fps: {self.train_fps}")
            self.sample_mode = "train_fps"
        elif self.sample_stride is not None:
            print(f"Using sample_stride mode, sample_stride: {self.sample_stride}")
            self.sample_mode = "sample_stride"
        else:
            raise ValueError("Must specify either train_fps or sample_stride")

        self.text_drop_ratio = text_drop_ratio
        self.tokenizer_max_length = tokenizer_max_length
        self.return_prompt_mask = return_prompt_mask
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=self.tokenizer_max_length,
            return_prompt_mask=self.return_prompt_mask,
        )

        self.video_transform_pipeline = transform_pipeline
        self.video_processor = WanVideoProcessor(
            sample_height=self.sample_height,
            sample_width=self.sample_width,
            sample_num_frames=self.sample_num_frames,
            train_fps=self.train_fps,
            sample_stride=self.sample_stride,
            transform_pipeline=self.video_transform_pipeline,
        )

        print(f"video transform pipeline: \n {self.video_transform_pipeline}")

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.timeout = kwargs.get("timeout", 300) 


    def __getitem__(self, index):
        try:
            future = self.executor.submit(self.getitem, index)
            data = future.result(timeout=self.timeout) 
            return data
        except Exception as e:
            print(f"the error is {e}")
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        # return self.getitem(index)

    def __len__(self):
        return self.data_length

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(T2VOutputData)
        meta_info = self.dataset_reader.getitem(index)
        text = meta_info["cap"]
        video_path = meta_info["path"]
        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text)
        examples[VIDEO] = self.get_video_data(video_path, meta_info)
        return examples


    def get_video_data(self, video_path, meta_info):
        video = self.video_processor(video_path, meta_info)
        return video

    
    def get_text_data(self, text):
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)][0]
        if random.random() < self.text_drop_ratio:
            text = ""
        prompt_input_ids, prompt_mask = self.text_processor(text)
        return prompt_input_ids, prompt_mask
        
