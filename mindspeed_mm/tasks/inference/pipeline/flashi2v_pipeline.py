# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union
import html
import math
import os

from PIL.Image import Image
import ftfy
import regex as re
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import center_crop
from transformers import CLIPVisionModel
from megatron.training import get_args
from megatron.core import mpu
from mindspeed_mm.utils.utils import get_device
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.utils.filter import HighFrequencyExtractor

from einops import repeat, rearrange
import numpy as np

from .pipeline_base import MMPipeline
from .pipeline_mixin.encode_mixin import MMEncoderMixin
from .pipeline_mixin.inputs_checks_mixin import InputsCheckMixin

NEGATIVE_PROMOPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


class FlashI2VPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, tokenizer, text_encoder, scheduler, predictor, config=None):
        super().__init__()

        args = get_args()
        args = args.mm.model

        self.model_cpu_offload_seq = "text_encoder->predictor->vae"

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            predictor=predictor,
        )

        self.model_type = getattr(args.predictor, 'model_type', 't2v')
        self.cp_size = mpu.get_context_parallel_world_size()
        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.model.config.temperal_downsample) if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.model.config.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.patch_size = self.predictor.patch_size if getattr(self, "predictor", None) else (1, 2, 2)

        self.num_frames, self.height, self.width = config.input_size
        self.generator = None if not hasattr(config, "seed") else torch.Generator().manual_seed(config.seed)

        self.cpu_offload = getattr(config, "cpu_offload", False)
        if self.cpu_offload:
            local_rank = int(os.getenv("LOCAL_RANK"))
            self.enable_model_cpu_offload(local_rank)

        self.transform = get_transforms(is_video=True, train_pipeline=args.data_transform.to_dict())

        self.low_freq_proj = getattr(self.predictor, "low_freq_proj", False)
        self.low_freq_energy_ratio = getattr(self.predictor, "low_freq_energy_ratio", 0.5)
        self.fft_return_abs = getattr(self.predictor, "fft_return_abs", False)
        print(f"low_freq_proj: {self.low_freq_proj}, low_freq_energy_ratio: {self.low_freq_energy_ratio}")
        self.high_freq_extractor = HighFrequencyExtractor(
            low_freq_energy_ratio=self.low_freq_energy_ratio, 
            return_lowpass=self.low_freq_proj, 
            return_abs=self.fft_return_abs
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[Image, List[Image]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: torch.device = get_device("npu"),
        max_sequence_length: int = 512,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMOPT
        self.check_inputs(
            prompt,
            negative_prompt,
            self.height,
            self.width,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        prompt_embeds, negative_prompt_embeds = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 4. Prepare latents and model_kwargs
        if image is not None:
            latents, start_frame_latents = self.prepare_image_latents(
                batch_size, image, device, prompt_embeds.dtype
            )
        else:
            shape = (
                batch_size,
                self.predictor.in_dim,
                (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
                self.height // self.vae_scale_factor_spatial,
                self.width // self.vae_scale_factor_spatial,
            )
            latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=prompt_embeds.dtype)
            start_frame_latents = None

        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

        if not self.low_freq_proj:
            fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2))
        else:
            start_frame_latents, fourier_features = self.high_freq_extractor(start_frame_latents.squeeze(2))
            start_frame_latents = start_frame_latents.unsqueeze(2)
        fourier_features = fourier_features.unsqueeze(2)
        fourier_features = repeat(fourier_features, 'b c 1 h w -> b c t h w', t=latents.shape[2]).contiguous()
        
        start_frame_latents = repeat(start_frame_latents, 'b c 1 h w -> b c t h w', t=latents.shape[2]).contiguous()

        start_frame_latents = start_frame_latents.to(self.predictor.dtype)
        start_frame_latents = self.predictor.learnable_proj(start_frame_latents)
        start_frame_latents = start_frame_latents.to(latents.dtype)

        model_kwargs.update({
            "start_frame_latents": start_frame_latents, 
            "fourier_features": fourier_features,
        })
        
        latents = self.scheduler.sample(model=self.predictor, latents=latents, **model_kwargs)
        
        # 6. Post process latents to get video
        latents = latents.to(self.vae.model.dtype)
        latents_mean = (
            torch.tensor(self.vae.model.config.latents_mean)
            .view(1, self.vae.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = torch.tensor(self.vae.model.config.latents_std).view(
            1, self.vae.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
        video = self.decode_latents(latents)
        return video

    def encode_texts(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

    def prompt_preprocess(self, prompt):

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return whitespace_clean(basic_clean(prompt))

    def prepare_image_latents(self, batch_size, image, device, dtype):
        if not isinstance(image[0], torch.Tensor):
            image = [torch.from_numpy(np.array(i)) for i in image]
        image = torch.stack(image, dim=0) # B H W C
        assert image.shape[0] == batch_size, f"Expected batch size {batch_size} but got {image.shape[0]}."

        latent_h = round(
            self.height
            // self.vae_scale_factor_spatial
            // self.patch_size[1]
            // self.cp_size
            * self.patch_size[1]
            * self.cp_size
        )
        latent_w = round(
            self.width
            // self.vae_scale_factor_spatial
            // self.patch_size[2]
            // self.cp_size
            * self.patch_size[2]
            * self.cp_size
        )

        shape = (
            batch_size,
            self.vae.model.config.z_dim,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            latent_h,
            latent_w,
        )

        noise = self.prepare_latents(shape, generator=self.generator, device=device, dtype=dtype)

        image = rearrange(image, 'b h w c -> b c h w')
        image = self.transform(image) # B C H W
        image = image.unsqueeze(2).to(dtype=self.vae.dtype, device=device) # B C H W -> B C 1 H W
        image_latents = self.vae.encode(image)
        image_latents = image_latents.to(dtype)

        return noise, image_latents

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [self.prompt_preprocess(u) for u in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds.to(self.predictor.dtype)
