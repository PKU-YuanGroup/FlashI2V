import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline


class MMPipeline(DiffusionPipeline):

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, value_range=(-1, 1), normalize=True, **kwargs):
        video = self.vae.decode(latents, **kwargs)  # [b, c, t, h, w]
        if normalize:
            low, high = value_range
            video.clamp_(min=low, max=high)
            video.sub_(low).div_(max(high - low, 1e-5))
        # [b, c, t, h, w] --> [b, t, h, w, c]
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)
        return video
