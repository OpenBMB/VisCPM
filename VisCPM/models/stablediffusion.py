import torch
import os
import VisCPM.models.modeling_utils as utils
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput
from transformers import CLIPImageProcessor, pipeline
from PIL import Image
import numpy as np


class CPMBeeTransBlock(torch.nn.Module):
    def __init__(
        self,
        dim_model=4096,
        dim_ff=1024,
        dim_out=768,
        dtype=torch.float,
        eps=1e-6,
        dropout_p=0,
    ):
        super().__init__()
        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.w_out_res = torch.nn.Linear(dim_model, dim_out, bias=False)
        self.layernorm = torch.nn.LayerNorm(
            dim_out,
            dtype=dtype,
            eps=eps,
        )

    def forward(self, hidden_states: torch.Tensor):
        x_res = self.w_out_res(hidden_states)
        if self.dropout is not None:
            x_res = self.dropout(x_res)
        hidden_states = self.layernorm(x_res)
        return hidden_states


class SDWrapper(torch.nn.Module):
    def __init__(self, image_safety_checker=True):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='vae')
        self.noise_scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
        self.unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(
            'stabilityai/stable-diffusion-2-1-base', subfolder='unet'))

        self.trans_block = CPMBeeTransBlock(4096, 4096 // 4, self.unet.config.cross_attention_dim)
        if image_safety_checker:
            self.image_safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker")
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        else:
            self.image_safety_checker = None
            self.feature_extractor = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def forward(self, pixel_values, text_hidden_states):
        pixel_values = pixel_values.type(text_hidden_states.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        text_hidden_states = text_hidden_states.type(noisy_latents.dtype)
        if self.trans_block is not None:
            text_hidden_states = self.trans_block(text_hidden_states)
        model_pred = self.unet(noisy_latents, timesteps, text_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return loss, model_pred

    @torch.no_grad()
    def generate(self,
                 text_hidden_states,
                 uncond_text_hidden_states,
                 height=None,
                 width=None,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 num_images_per_prompt=1,
                 generator=None,
                 latents=None,
                 scheduler=None,
                 output_type='pil'
                 ):
        device = text_hidden_states.device
        batch_size = text_hidden_states.size(0)
        text_hidden_states = text_hidden_states.type(self.unet.conv_in.weight.dtype)
        uncond_text_hidden_states = uncond_text_hidden_states.type(self.unet.conv_in.weight.dtype)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        if scheduler is not None:
            self.noise_scheduler = scheduler

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_hidden_states.dtype,
            device,
            generator,
            latents,
        )

        if self.trans_block is not None:
            text_hidden_states = self.trans_block(text_hidden_states)
            uncond_text_hidden_states = self.trans_block(uncond_text_hidden_states)

        bs_embed, seq_len, _ = text_hidden_states.shape
        text_hidden_states = text_hidden_states.repeat(1, num_images_per_prompt, 1)
        text_hidden_states = text_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed, seq_len, _ = uncond_text_hidden_states.shape
        uncond_text_hidden_states = uncond_text_hidden_states.repeat(1, num_images_per_prompt, 1)
        uncond_text_hidden_states = uncond_text_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)

        text_hidden_states = torch.cat([uncond_text_hidden_states, text_hidden_states])

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_hidden_states,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        image = self.decode_latents(latents)
        # Run safety checker
        image, has_nsfw_concept = self.run_image_safety_checker(image, device, self.unet.conv_in.weight.dtype)
        if output_type == 'pil':
            image = utils.numpy_to_pil(image)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = utils.randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the noise_scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_image_safety_checker(self, image, device, dtype):
        if self.image_safety_checker is not None:
            image_safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.image_safety_checker(
                images=image, clip_input=image_safety_checker_input.pixel_values.to(dtype)
            )
            if any(has_nsfw_concept):
                print(
                    "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                )
                for idx, _has_nsfw_concept in enumerate(has_nsfw_concept):
                    if _has_nsfw_concept:
                        image[idx] = np.zeros(image[idx].shape)  # black image
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept
