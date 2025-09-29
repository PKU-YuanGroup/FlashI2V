# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import torch
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder


class StepVideoDPOModel(nn.Module):
    """
    The hyper model wraps multiple models required in reinforcement learning into a single model,
    maintaining the original distributed perspective unchanged.
    """

    def __init__(self, config):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self._model_provider(config)

    def _model_provider(self, config):
        """Builds the model."""

        print_rank_0("building StepVideo related modules ...")
        self.ae = AEModel(config.ae).eval()
        self.ae.requires_grad_(False)

        self.text_encoder = TextEncoder(config.text_encoder).eval()
        self.text_encoder.requires_grad_(False)

        self.diffusion = DiffusionModel(config.diffusion).get_model()

        self.reference = PredictModel(config.predictor).get_model().eval()
        self.reference.requires_grad_(False)

        self.actor = PredictModel(config.predictor).get_model()
        print_rank_0("finish building StepVideo related modules ...")

        return None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.actor.set_input_tensor(input_tensor)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        if not get_args().dist_train:
            state_dict = self.actor.state_dict(prefix=prefix, keep_vars=keep_vars)
            return state_dict
        from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        if is_in_subworld('dit'):
            return self.actor.state_dict(prefix=prefix, keep_vars=keep_vars)
        return None

    def forward(self, video, video_lose, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        latents, _ = self.ae.encode(video)
        latents_lose, _ = self.ae.encode(video_lose)
        noised_latents, noise, timesteps = self.diffusion.q_sample(torch.cat((latents, latents_lose), dim=0), model_kwargs=kwargs, mask=video_mask)
        prompts, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask)
        prompt = [torch.cat((prompt, prompt), dim=0) for prompt in prompts]
        prompt_mask = [torch.cat((mask, mask), dim=0) for mask in prompt_mask]

        with torch.no_grad():
            refer_output = self.reference(
                noised_latents,
                timestep=timesteps,
                prompt=prompt,
                video_mask=video_mask,
                prompt_mask=prompt_mask,
                **kwargs,
            )
            if isinstance(refer_output, tuple):
                refer_output = refer_output[0]
        actor_output = self.actor(
            noised_latents,
            timestep=timesteps,
            prompt=prompt,
            video_mask=video_mask,
            prompt_mask=prompt_mask,
            **kwargs,
        )
        if isinstance(actor_output, tuple):
            actor_output = actor_output[0]
        output = torch.cat((refer_output, actor_output), dim=0)

        return output, torch.cat((latents, latents_lose), dim=0), noised_latents, noise, timesteps

