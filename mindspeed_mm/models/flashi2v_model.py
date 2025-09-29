# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from typing import Any, Mapping

import torch
import torch_npu
import random
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.data.data_utils.constants import (
    START_FRAME,
    LATENTS,
    PROMPT,
    PROMPT_MASK,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.utils.utils import unwrap_single, broadcast_tensor
from mindspeed_mm.models import SoRAModel
from mindspeed_mm.utils.filter import HighFrequencyExtractor

from einops import rearrange, repeat
from copy import deepcopy

import time

logger = getLogger(__name__)

class FlashI2VModel(SoRAModel):

    def __init__(self, config):
        super().__init__(config)
        self.low_freq_proj = getattr(config.predictor, "low_freq_proj", False)
        self.low_freq_energy_ratio = getattr(config.predictor, "low_freq_energy_ratio", 0.5)
        self.fft_return_abs = getattr(config.predictor, "fft_return_abs", False)
        print_rank_0(f"low_freq_proj: {self.low_freq_proj}, low_freq_energy_ratio: {self.low_freq_energy_ratio}, fft_return_abs: {self.fft_return_abs}")
        self.high_freq_extractor = HighFrequencyExtractor(
            low_freq_energy_ratio=self.low_freq_energy_ratio,
            return_lowpass=self.low_freq_proj,
            return_abs=self.fft_return_abs,
        )

    def forward(
        self, 
        video,
        prompt_ids, 
        start_frame=None,
        video_mask=None, 
        prompt_mask=None,
        sigmas=None,
        skip_encode=False,
        **kwargs
    ):

        if self.pre_process:
            with torch.autocast("cuda", enabled=False):
                with torch.no_grad():
                    if not skip_encode:
                        self.index = 0

                        # Text Encode
                        if self.load_text_features:
                            prompt = prompt_ids
                            if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                                prompt = [p.npu() for p in prompt]
                        else:
                            prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask,
                                                                        offload_cpu=self.offload_cpu, **kwargs)

                        # Visual Encode
                        if self.load_video_features:
                            latents = video
                        else:
                            latents, _ = self.ae.encode(video)
                            start_frame_latents, _ = self.ae.encode(start_frame)

            # Gather the results after encoding of ae and text_encoder
            if self.enable_encoder_dp or self.interleaved_steps > 1:
                if self.index == 0:
                    self.init_cache(latents, prompt, video_mask, prompt_mask, start_frame_latents)
                latents, prompt, video_mask, prompt_mask, start_frame_latents = self.get_feature_from_cache()
            
            if not self.low_freq_proj:
                fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2))
            else:
                start_frame_latents, fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2))
                start_frame_latents = start_frame_latents.unsqueeze(2)
            fourier_features = fourier_features.unsqueeze(2)
            fourier_features = repeat(fourier_features, 'b c 1 h w -> b c t h w', t=latents.shape[2]).contiguous()
            
            start_frame_latents = repeat(start_frame_latents, 'b c 1 h w -> b c t h w', t=latents.shape[2]).contiguous()

            latent_added_noise = torch.randn_like(start_frame_latents, dtype=latents.dtype, device=latents.device)
            broadcast_tensor(latent_added_noise)

            start_frame_latents = start_frame_latents.to(self.predictor.dtype)
            start_frame_latents = self.predictor.learnable_proj(start_frame_latents)
            start_frame_latents = start_frame_latents.to(latents.dtype)
            latents = latents - start_frame_latents

            prior_dist = latent_added_noise - start_frame_latents

            q_sample_results = self.diffusion.q_sample(latents, prior_dist=prior_dist, sigmas=sigmas, model_kwargs=kwargs)
            x_t = q_sample_results.pop('x_t', None)
            prior_dist = q_sample_results.pop('prior_dist', None)
            sigmas = q_sample_results.pop('sigmas', None)
            timesteps = q_sample_results.pop('timesteps', None)

            x_t_for_input = torch.cat([x_t, fourier_features], dim=1)
  
            predictor_input_latent, predictor_timesteps, predictor_prompt = x_t_for_input, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask

        else:
            if not hasattr(self.predictor, "pipeline_set_prev_stage_tensor"):
                raise ValueError(f"PP has not been implemented for {self.predictor_cls} yet. ")
            predictor_input_list, training_loss_input_list = self.predictor.pipeline_set_prev_stage_tensor(
                self.input_tensor, extra_kwargs=kwargs)
            predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask \
                = predictor_input_list
            latents, prior_dist, video_mask = training_loss_input_list

        output = self.predictor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            loss_list = self.compute_loss(
                output if isinstance(output, torch.Tensor) else output[0],
                latents,
                prior_dist,
                video_mask=video_mask,
                sigmas=sigmas,
                **kwargs,
            )
            return loss_list

        return self.predictor.pipeline_set_next_stage_tensor(
            input_list=[latents, prior_dist, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def compute_loss(
        self, 
        model_output,
        latents,
        prior_dist,
        video_mask=None,
        sigmas=None,
        **kwargs
    ):
        """compute diffusion loss"""
        loss_list = self.diffusion.training_losses(
            model_output=model_output,
            latents=latents,
            prior_dist=prior_dist,
            mask=video_mask,
            sigmas=sigmas,
            **kwargs
        )
        return loss_list

    def init_cache(self, latents, prompt, video_mask, prompt_mask, start_frame):
        # empty cache
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        for key, value in [(LATENTS, latents), (VIDEO_MASK, video_mask)]:
            if value is None or len(value) < 0:
                continue
            self.cache[key] = [[item] for item in value] if not self.enable_encoder_dp \
                else collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

        for key, value in [(PROMPT, prompt), (PROMPT_MASK, prompt_mask)]:
            if value is None or len(value) < 0:
                continue
            if not self.enable_encoder_dp:
                self.cache[key] = [[item] for item in value]
                continue
            self.cache[key] = [[[] for _ in range(self.text_encoder_num)] for _ in range(self.interleaved_steps)]
            for encoder_idx in range(self.text_encoder_num):
                # Features from the same text encoder have identical shapes, concat to reduce communication overhead.
                encoder_step_tensors = torch.stack([value[step][encoder_idx] for step in range(self.interleaved_steps)])
                collected_tensors = collect_tensors_across_ranks(encoder_step_tensors, group=group, dynamic_shape=False)

                for step_idx in range(self.interleaved_steps):
                    for collected_tensor in collected_tensors:
                        self.cache[key][step_idx][encoder_idx].append(
                            collected_tensor[step_idx:step_idx + 1].squeeze(0).contiguous())
        
        if start_frame is None or len(start_frame) < 1:
            return
        for key, value in [(START_FRAME, start_frame)]:
            if not self.enable_encoder_dp:
                self.cache[key] = [[item] for item in value]
                continue
            self.cache[key] = collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

    def get_feature_from_cache(self):
        """Get from the cache while several features have been already encoded and cached."""
        divisor = mpu.get_tensor_and_context_parallel_world_size() if self.enable_encoder_dp else 1
        step_idx = self.index // divisor
        rank_idx = self.index % divisor

        latents = unwrap_single(self.cache[LATENTS][step_idx][rank_idx] if LATENTS in self.cache else None)
        video_mask = unwrap_single(self.cache[VIDEO_MASK][step_idx][rank_idx] if VIDEO_MASK in self.cache else None)
        prompt = unwrap_single([self.cache[PROMPT][step_idx][encoder_idx][rank_idx] \
                                for encoder_idx in range(self.text_encoder_num)] if PROMPT in self.cache else None)
        prompt_mask = unwrap_single([self.cache[PROMPT_MASK][step_idx][encoder_idx][rank_idx] \
                                     for encoder_idx in range(self.text_encoder_num)] if PROMPT_MASK in self.cache else None)
        start_frame = unwrap_single(self.cache[START_FRAME][step_idx][rank_idx] if START_FRAME in self.cache else None)

        self.index += 1
        return latents, prompt, video_mask, prompt_mask, start_frame
