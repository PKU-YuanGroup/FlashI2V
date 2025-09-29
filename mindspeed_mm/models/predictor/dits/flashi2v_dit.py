import math
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from megatron.core import mpu, tensor_parallel
from megatron.legacy.model.enums import AttnType
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed.core.context_parallel.unaligned_cp.mapping import (
    all_to_all,
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.attention import FlashAttention, ParallelAttention
from mindspeed_mm.models.common.embeddings import TextProjection
from mindspeed_mm.models.common.normalize import normalize, FP32LayerNorm


from mindspeed_mm.models.predictor.dits.wan_dit import WanDiT


def zero_initialize(module):
    for param in module.parameters():
        nn.init.zeros_(param)
    return module

class FlashI2VDiT(WanDiT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv3x3x3_proj = kwargs.get("conv3x3x3_proj", True)

        self.low_freq_proj = kwargs.get("low_freq_proj", False)
        self.low_freq_energy_ratio = kwargs.get("low_freq_energy_ratio", 0.5)
        self.fft_return_abs = kwargs.get("fft_return_abs", False)

        self.fourier_embedding = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_dim,
                out_channels=self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ),
            zero_initialize(
                nn.Conv3d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1)
                )
            )
        )

        proj_in_dim = proj_out_dim = self.in_dim
        print(f'conv3x3x3 proj: {self.conv3x3x3_proj}')
        if self.conv3x3x3_proj:
            self.learnable_proj = nn.Sequential(
                nn.Conv3d(
                    proj_in_dim,
                    proj_out_dim * 4,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                ),
                nn.SiLU(),
                zero_initialize(
                    nn.Conv3d(
                        proj_out_dim * 4, 
                        proj_out_dim,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1)
                    )
                )
            )
        else:
            self.learnable_proj = nn.Sequential(
                nn.Conv3d(
                    proj_in_dim,
                    proj_out_dim * 4,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    padding=(0, 1, 1)
                ),
                nn.SiLU(),
                zero_initialize(
                    nn.Conv3d(
                        proj_out_dim * 4, 
                        proj_out_dim,
                        kernel_size=(1, 3, 3),
                        stride=(1, 1, 1),
                        padding=(0, 1, 1)
                    )
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.pre_process:
            timestep = timestep.to(x[0].device)
            # time embeddings
            with torch.autocast("cuda", enabled=False):
                times = self.time_embedding(
                    self.sinusoidal_embedding_1d(self.freq_dim, timestep).float()
                )
                time_emb = self.time_projection(times).unflatten(1, (6, self.hidden_size)).to(self.fp32_dtype)
                if time_emb.dtype != torch.float32:
                    raise ValueError("time_emb dtype error")

            # prompt embeddings
            bs = prompt.size(0)
            prompt = prompt.view(bs, -1, prompt.size(-1))
            if prompt_mask is not None:
                seq_lens = prompt_mask.view(bs, -1).sum(dim=-1)
                for i, seq_len in enumerate(seq_lens):
                    prompt[i, seq_len:] = 0
            prompt_emb = self.text_embedding(prompt)

            # patch embedding
            latents, fourier_features = x[:, :self.in_dim], x[:, self.in_dim:]
            patch_emb = self.patch_embedding(latents)
            fourier_features = self.fourier_embedding(fourier_features)
            patch_emb = fourier_features + patch_emb
            
            # patch_emb = self.patch_embedding(x)

            embs, grid_sizes = self.patchify(patch_emb)

            # rotary positional embeddings
            batch_size, frames, height, width = (
                embs.shape[0],
                grid_sizes[0],
                grid_sizes[1],
                grid_sizes[2],
            )
        else:
            batch_size, _, frames, height, width = kwargs["ori_shape"]
            height, width = height // self.patch_size[1], width // self.patch_size[2]
            prompt_emb = kwargs['prompt_emb']
            time_emb = kwargs['time_emb']
            times = kwargs['times']
            embs = x

        rotary_pos_emb = self.rope(batch_size, frames, height, width)

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # cp split
        if self.context_parallel_algo is not None:
            if self.pre_process:
                embs = split_forward_gather_backward(
                    embs, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
                )  # b s h
            rotary_pos_emb = split_forward_gather_backward(
                rotary_pos_emb,
                mpu.get_context_parallel_group(),
                dim=0,
                grad_scale="down",
            )

        with rng_context:
            if self.recompute_granularity == "full":
                embs = self._checkpointed_forward(
                    self.blocks,
                    embs,
                    prompt_emb,
                    time_emb,
                    rotary_pos_emb,
                )
            else:
                for block in self.blocks:
                    embs = block(embs, prompt_emb, time_emb, rotary_pos_emb)

        out = embs
        if self.post_process:
            if self.context_parallel_algo is not None:
                embs = gather_forward_split_backward(
                    embs, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
                )
            embs_out = self.head(embs, times)
            out = self.unpatchify(embs_out, frames, height, width)

        rtn = (out, prompt, prompt_emb, time_emb, times, prompt_mask)

        return rtn