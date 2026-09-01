# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
import math, random  # ★ NEW: inside guidance scheduling / sampling
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AsymmetricAutoencoderKL, AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker
from .rda_guidance import (
    append_reference_tokens,
    load_rda_adapter,
)
from .carf_guidance import load_carf_refiner
from .msdf_guidance import load_msdf_adapter
from .rda_carf_attention import install_rda_carf_attention


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionInpaintPipeline_dynamic(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`, `AsymmetricAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate("skip_prk_steps not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # Check shapes, assume num_channels_latents == 4, num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 9:
            logger.info(f"You have loaded a UNet with {unet.config.in_channels} input channels which.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta, anomaly_strength=0.0):
        # 기존 DDIM용 extra kwargs + anomaly_strength 추가
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # scheduler.step에서 generator 지원 여부 확인
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        # [추가] anomaly_strength를 extra_step_kwargs에 포함
        extra_step_kwargs["anomaly_strength"] = anomaly_strength

        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        mask_image,
        height,
        width,
        strength,
        callback_steps,
        output_type,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        padding_mask_crop=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if padding_mask_crop is not None:
            if not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"The image should be a PIL image when inpainting mask crop, but is of type" f" {type(image)}."
                )
            if not isinstance(mask_image, PIL.Image.Image):
                raise ValueError(
                    f"The mask image should be a PIL image when inpainting mask crop, but is of type"
                    f" {type(mask_image)}."
                )
            if output_type != "pil":
                raise ValueError(f"The output type should be PIL when inpainting mask crop, but is" f" {output_type}.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}")
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # ---- NEW: inside guidance 스케줄 함수 ----
    def _gsi_shape(self, schedule: str, s: float, power: float, exp_k: float, sigmoid_k: float) -> float:
        """
        0..1 → 0..1 가중치. 'down'은 s=0일 때 1, s=1일 때 0이 되도록 설계.
        schedule이 'cosine','linear','poly','exp','sigmoid'처럼 suffix 없음으로 들어오면 'down'으로 간주.
        """
        s = max(0.0, min(1.0, s))
        sch = (schedule or "linear").lower()
        # suffix normalize: (기본 down)
        if sch in ["linear", "cosine", "poly", "exp", "sigmoid", "constant"]:
            if sch != "constant":
                sch = f"{sch}_down"

        if sch == "constant":
            return 1.0
        if sch == "linear_down":
            return 1.0 - s
        if sch == "cosine_down":
            return 0.5 * (1.0 + math.cos(math.pi * s))
        if sch == "poly_down":
            return (1.0 - s) ** max(power, 1e-6)
        if sch == "exp_down":
            return math.exp(-exp_k * s)
        if sch == "sigmoid_down":
            sig = 1.0 / (1.0 + math.exp(sigmoid_k * (s - 0.5)))
            return max(0.0, min(1.0, (sig - 0.5) / 0.5))
        # fallback
        return 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.Tensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        # NEW: spatial guidance scales
        guidance_scale_inside: Optional[float] = None,   # 스케줄/샘플링 미사용 시 스칼라 fallback
        guidance_scale_outside: Optional[float] = None,  # 항상 스칼라 유지(미지정이면 global guidance_scale 사용)
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        anomaly_strength: float = 0.0,
        anomaly_stop_step: int = 999999,
        eta_mask_stop_step: int = 999999,
        use_random_mask = False,
        eta_mask: float = 0.0,

        # ---- NEW: inside guidance 스케줄/샘플링 옵션 ----
        gsi_use_schedule: bool = False,
        gsi_schedule: str = "linear",
        guidance_scale_inside_min: Optional[float] = None,
        guidance_scale_inside_max: Optional[float] = None,
        gsi_power: float = 2.0,
        gsi_exp_k: float = 3.0,
        gsi_sigmoid_k: float = 8.0,
        gsi_sample_per_step: bool = False,
        mdap_prior_image: Optional[PipelineImageInput] = None,
        mdap_strength: float = 0.0,
        mdap_schedule: str = "cosine",
        mdap_end_fraction: float = 0.7,
        rda_enabled: bool = False,
        rda_path: Optional[str] = None,
        rda_reference_image: Optional[PipelineImageInput] = None,
        rda_reference_mask: Optional[PipelineImageInput] = None,
        carf_enabled: bool = False,
        carf_path: Optional[str] = None,
        msdf_enabled: bool = False,
        msdf_path: Optional[str] = None,
        msdf_reference_image: Optional[PipelineImageInput] = None,
        msdf_reference_mask: Optional[PipelineImageInput] = None,
        **kwargs,
    ):
        device = self._execution_device
        do_classifier_free_guidance = (guidance_scale > 1.0)

        # (a) height/width default
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(device=device, dtype=self.vae.dtype)
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        mask_condition = mask_condition.to(device=init_image.device, dtype=init_image.dtype)
        rda_adapter = None
        rda_reference_latents = None
        rda_reference_condition = None

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        batch_size = init_image.shape[0]
        num_channels_latents = self.vae.config.latent_channels
        # (a) vae encode init_image
        init_image_latents = self.vae.encode(init_image).latent_dist.sample(generator=generator)
        init_image_latents = self.vae.config.scaling_factor * init_image_latents
        # MDAP is represented as a latent residual direction. This preserves
        # the frozen 9-channel UNet interface and avoids losing the prior when
        # the standard inpainting path zeros pixels inside the mask.
        mdap_delta_latents = None
        if mdap_prior_image is not None and mdap_strength > 0:
            mdap_image = self.image_processor.preprocess(
                mdap_prior_image, height=height, width=width,
                crops_coords=crops_coords, resize_mode=resize_mode,
            )
            mdap_image = mdap_image.to(device=device, dtype=self.vae.dtype)
            mdap_prior_latents = self.vae.encode(mdap_image).latent_dist.mode()
            mdap_prior_latents = self.vae.config.scaling_factor * mdap_prior_latents
            mdap_delta_latents = mdap_prior_latents - init_image_latents

        if latents is None:
            noise = randn_tensor(
                (batch_size, num_channels_latents, init_image_latents.shape[2], init_image_latents.shape[3]),
                generator=generator, device=device, dtype=init_image_latents.dtype
            )
            if strength == 1.0:
                latents = noise * self.scheduler.init_noise_sigma
            else:
                latents = noise
        else:
            latents = latents.to(device)

        # (c) mask/ masked_image_latents => vae encode
        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        mask = mask.to(device=device, dtype=init_image_latents.dtype)
        mask = torch.nn.functional.interpolate(
            mask, size=(init_image_latents.shape[2], init_image_latents.shape[3])
        )
        carf_refiner = None
        if carf_enabled:
            if not carf_path:
                raise ValueError("carf_path is required when CARF is enabled")
            if getattr(self, "_carf_path", None) != carf_path:
                self._carf_refiner = load_carf_refiner(
                    carf_path, init_image_latents.device
                )
                self._carf_path = carf_path
            carf_refiner = self._carf_refiner
            print(f"[CARF] enabled | checkpoint={carf_path}")
        msdf_adapter = None
        msdf_reference_latents = None
        msdf_reference_condition = None
        if msdf_enabled:
            if not msdf_path or msdf_reference_image is None or msdf_reference_mask is None:
                raise ValueError(
                    "msdf_path, msdf_reference_image and msdf_reference_mask "
                    "are required when MSDF is enabled"
                )
            if getattr(self, "_msdf_path", None) != msdf_path:
                if getattr(self, "_msdf_adapter", None) is not None:
                    self._msdf_adapter.detach()
                self._msdf_adapter = load_msdf_adapter(
                    msdf_path, self.unet, init_image_latents.device
                )
                self._msdf_path = msdf_path
            msdf_adapter = self._msdf_adapter
            msdf_reference = self.image_processor.preprocess(
                msdf_reference_image, height=height, width=width
            ).to(device=device, dtype=self.vae.dtype)
            msdf_reference_condition = self.mask_processor.preprocess(
                msdf_reference_mask, height=height, width=width
            ).to(device=device, dtype=torch.float32)
            msdf_reference_latents = self.vae.encode(
                msdf_reference
            ).latent_dist.mode()
            msdf_reference_latents = (
                self.vae.config.scaling_factor * msdf_reference_latents
            )
            print(f"[MSDF] enabled | reference={tuple(msdf_reference.shape)}")
        elif getattr(self, "_msdf_adapter", None) is not None:
            self._msdf_adapter.clear()
        if rda_enabled:
            if not rda_path or rda_reference_image is None or rda_reference_mask is None:
                raise ValueError(
                    "rda_path, rda_reference_image and rda_reference_mask "
                    "are required when RDA is enabled"
                )
            if getattr(self, "_rda_path", None) != rda_path:
                self._rda_adapter = load_rda_adapter(rda_path, init_image_latents.device)
                self._rda_path = rda_path
            rda_adapter = self._rda_adapter
            rda_reference = self.image_processor.preprocess(
                rda_reference_image, height=height, width=width
            ).to(device=device, dtype=self.vae.dtype)
            rda_reference_condition = self.mask_processor.preprocess(
                rda_reference_mask, height=height, width=width
            ).to(device=device, dtype=torch.float32)
            rda_reference_latents = self.vae.encode(
                rda_reference
            ).latent_dist.mode()
            rda_reference_latents = (
                self.vae.config.scaling_factor * rda_reference_latents
            )
            print(f"[RDA] enabled | reference={tuple(rda_reference.shape)}")
        masked_image = masked_image.to(device=device, dtype=init_image_latents.dtype)
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

        # ---------------------------------------------------
        # 3) CFG => cat latents/mask -> (2B,...)
        # ---------------------------------------------------
        if msdf_adapter is not None:
            msdf_adapter.clear()

        if do_classifier_free_guidance:
            latents = torch.cat([latents, latents], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
            if mdap_delta_latents is not None:
                mdap_delta_latents = torch.cat([mdap_delta_latents, mdap_delta_latents], dim=0)
            batch_size = batch_size * 2

        # ---------------------------------------------------
        # 4) Prompt encode
        # ---------------------------------------------------
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs else None
        prompt_embeds, negative_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        if do_classifier_free_guidance:
            # concat uncond + cond
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        if rda_adapter is not None:
            reference_tokens = rda_adapter(
                rda_reference_latents.float(),
                rda_reference_condition,
            )
            prompt_embeds = append_reference_tokens(
                prompt_embeds,
                reference_tokens,
                do_classifier_free_guidance,
            )
        joint_spatial_attention = rda_adapter is not None and carf_refiner is not None
        if joint_spatial_attention:
            attention_key = (rda_path, carf_path)
            if getattr(self, "_rda_carf_attention_key", None) != attention_key:
                installed = install_rda_carf_attention(self.unet)
                self._rda_carf_attention_key = attention_key
                print(f"[RDA+CARF] spatial attention processors={installed}")

        timesteps, num_inference_steps = self.scheduler.set_timesteps(num_inference_steps, device=device), num_inference_steps
        # e.g. self.scheduler.init_noise_sigma, ...

        # ---------------------------------------------------
        # 6) extra_step_kwargs + mask_for_anomaly
        # ---------------------------------------------------
        step_params = set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if "eta" in step_params:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_params:
            extra_step_kwargs["generator"] = generator

        accepts_eta_mask       = ("eta_mask" in step_params)
        accepts_anomaly        = ("anomaly_strength" in step_params)
        accepts_mask_for_anom  = ("mask_for_anomaly" in step_params)
        accepts_use_random     = ("use_random_mask" in step_params)

        # ---- NEW: inside/outside guidance 설정 준비 ----
        # outside는 스칼라 그대로 사용(미지정 시 global guidance_scale 사용)
        gs_out_scalar = float(guidance_scale_outside) if (guidance_scale_outside is not None) else float(guidance_scale)

        use_gsi_schedule = bool(gsi_use_schedule and (guidance_scale_inside_min is not None) and (guidance_scale_inside_max is not None))
        use_gsi_sampling = bool((not use_gsi_schedule) and gsi_sample_per_step and (guidance_scale_inside_min is not None) and (guidance_scale_inside_max is not None))

        # (7) Denoising loop
        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(self.scheduler.timesteps):
                anomaly_strength_current = anomaly_strength if i < anomaly_stop_step else 0.0
                step_eta_mask = eta_mask if i < eta_mask_stop_step else 0.0

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                active_mask = mask
                step_cross_attention_kwargs = (
                    dict(cross_attention_kwargs)
                    if cross_attention_kwargs is not None
                    else {}
                )
                if carf_refiner is not None:
                    active_mask = carf_refiner(
                        latent_model_input.float(),
                        masked_image_latents.float(),
                        mask.float(),
                        t,
                    ).to(dtype=latent_model_input.dtype)
                if msdf_adapter is not None:
                    msdf_gates = msdf_adapter.prepare(
                        msdf_reference_latents.float(),
                        msdf_reference_condition,
                        mask[: msdf_reference_latents.shape[0]],
                        t,
                        int(getattr(self.scheduler.config, "num_train_timesteps", 1000)),
                        classifier_free_guidance=do_classifier_free_guidance,
                        reference_pixels=msdf_reference,
                    )
                    if not bool(torch.isfinite(msdf_gates).all()):
                        raise FloatingPointError(
                            f"MSDF produced non-finite gates at denoising step {i}: "
                            f"{msdf_path}"
                        )
                unet_mask = mask if joint_spatial_attention else active_mask
                if joint_spatial_attention:
                    step_cross_attention_kwargs.update(
                        {
                            "carf_attention_gate": carf_refiner.attention_gate(),
                            "rda_token_count": rda_adapter.num_tokens,
                        }
                    )

                if self.unet.config.in_channels == 9:
                    unet_input = torch.cat(
                        [latent_model_input, unet_mask, masked_image_latents], dim=1
                    )
                else:
                    unet_input = latent_model_input

                noise_pred = self.unet(
                    unet_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=(
                        step_cross_attention_kwargs
                        if step_cross_attention_kwargs
                        else None
                    ),
                    return_dict=False,
                )[0]
                if msdf_adapter is not None and not bool(torch.isfinite(noise_pred).all()):
                    raise FloatingPointError(
                        f"MSDF/UNet produced NaN or Inf at denoising step {i}: "
                        f"{msdf_path}"
                    )

                if do_classifier_free_guidance:
                    half = noise_pred.shape[0] // 2
                    noise_pred_uncond, noise_pred_cond = noise_pred[:half], noise_pred[half:]

                    # === NEW: Spatial CFG with inside scheduling / per-step sampling ===
                    # inside 스칼라 결정 (스케줄 > 샘플링 > 스칼라 fallback 순서)
                    if use_gsi_schedule:
                        s = 0.0 if (num_inference_steps <= 1) else (i / float(num_inference_steps - 1))  # 0→1
                        w01_down = self._gsi_shape(gsi_schedule, s, gsi_power, gsi_exp_k, gsi_sigmoid_k)  # 1→0 (down)
                        w01_up = 1.0 - w01_down  # ★ CHANGED: 0→1 (up) 로 변환
                        gsi_min = float(guidance_scale_inside_min)
                        gsi_max = float(guidance_scale_inside_max)
                        # ★ CHANGED: w01_up=0(첫 스텝)→min, w01_up=1(마지막)→max : min→max 선형 보간
                        gs_in_scalar = gsi_min + (gsi_max - gsi_min) * w01_up
                    elif use_gsi_sampling:
                        gs_in_scalar = random.uniform(float(guidance_scale_inside_min), float(guidance_scale_inside_max))
                    else:
                        base = guidance_scale_inside if (guidance_scale_inside is not None) else guidance_scale
                        gs_in_scalar = float(base)

                    # spatial guidance 적용: cond 배치의 마스크 사용
                    mask_cond = active_mask[half:]  # (B,1,H,W)
                    # import pdb;pdb.set_trace()
                    gs_in  = torch.as_tensor(gs_in_scalar,  device=noise_pred.device, dtype=noise_pred.dtype)
                    gs_out = torch.as_tensor(gs_out_scalar, device=noise_pred.device, dtype=noise_pred.dtype)
                    guidance_map = (gs_out * (1.0 - mask_cond)) + (gs_in * mask_cond)  # (B,1,H,W)
                    # 채널 브로드캐스트
                    if guidance_map.shape[1] != noise_pred_cond.shape[1]:
                        guidance_map = guidance_map.expand(-1, noise_pred_cond.shape[1], -1, -1)

                    noise_pred = noise_pred_uncond + guidance_map * (noise_pred_cond - noise_pred_uncond)
                else:
                    # guidance_scale<=1인 경우 CFG 미사용
                    noise_pred = noise_pred

                # 스텝별 kwargs
                step_kwargs = dict(extra_step_kwargs)
                if accepts_eta_mask:
                    step_kwargs["eta_mask"] = step_eta_mask
                if accepts_mask_for_anom:
                    step_kwargs["mask_for_anomaly"] = active_mask
                if accepts_anomaly:
                    step_kwargs["anomaly_strength"] = anomaly_strength_current
                if accepts_use_random:
                    step_kwargs["use_random_mask"] = use_random_mask
                step_kwargs["return_dict"] = False

                latents = self.scheduler.step(noise_pred, t, latents, **step_kwargs)[0]
                if msdf_adapter is not None and not bool(torch.isfinite(latents).all()):
                    raise FloatingPointError(
                        f"MSDF scheduler latents became NaN or Inf at step {i}: "
                        f"{msdf_path}"
                    )

                if mdap_delta_latents is not None:
                    progress = i / float(max(num_inference_steps - 1, 1))
                    if progress < mdap_end_fraction:
                        local_progress = progress / max(float(mdap_end_fraction), 1e-6)
                        if mdap_schedule == "linear":
                            schedule_weight = 1.0 - local_progress
                        elif mdap_schedule == "constant":
                            schedule_weight = 1.0
                        else:
                            schedule_weight = 0.5 * (1.0 + math.cos(math.pi * local_progress))
                        # Divide by the active step count so strength describes
                        # the approximate total residual injection.
                        active_steps = max(1, int(math.ceil(num_inference_steps * mdap_end_fraction)))
                        step_weight = float(mdap_strength) * schedule_weight / active_steps
                        latent_mask = mask
                        if latent_mask.shape[1] != mdap_delta_latents.shape[1]:
                            latent_mask = latent_mask.expand(
                                -1, mdap_delta_latents.shape[1], -1, -1
                            )
                        latents = latents + step_weight * mdap_delta_latents * latent_mask

                pbar.update()

        if msdf_adapter is not None:
            msdf_adapter.clear()

        if do_classifier_free_guidance:
            half = latents.shape[0] // 2
            latents = latents[half:]

        # decode
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if msdf_adapter is not None and not bool(torch.isfinite(image).all()):
            raise FloatingPointError(
                f"MSDF VAE decode produced NaN or Inf: {msdf_path}"
            )

        # safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, latents.dtype)
        do_denormalize = [True]*image.shape[0] if (has_nsfw_concept is None) else [not x for x in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # return
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
