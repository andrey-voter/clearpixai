"""Stable Diffusion based inpainting utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from ..utils.random import set_random_seed

logger = logging.getLogger(__name__)


@dataclass
class StableDiffusionSettings:
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    prompt: str = (
        "restore the original background realistically, remove watermark, seamless integration"
    )
    negative_prompt: str = (
        "watermark, text, logo, signature, writing, letters, words, blurry, distorted, artifacts"
    )
    num_inference_steps: int = 45
    guidance_scale: float = 5.0
    strength: float = 0.85
    padding: int = 24
    scheduler: str = "euler"
    guidance_rescale: float | None = None
    seed: int | None = None
    blend_with_original: float = 0.0
    mask_feather: int = 6


class StableDiffusionInpainter:
    def __init__(
        self,
        device: str,
        dtype="auto",
        settings: StableDiffusionSettings | None = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._settings = settings or StableDiffusionSettings()
        self._cache_dir = cache_dir
        self._pipe = None

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return

        logger.info("Loading Stable Diffusion inpainting pipeline: %s", self._settings.model_id)
        from diffusers import AutoPipelineForInpainting
        import torch

        dtype = torch.float16 if self._device == "cuda" and torch.cuda.is_available() else torch.float32

        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self._settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=str(self._cache_dir) if self._cache_dir else None,
        )

        scheduler_name = (self._settings.scheduler or "").lower()
        scheduler_cls = None
        try:
            if scheduler_name in {"dpm++", "dpmsolver", "dpmsolver++", "dpm-solver"}:
                from diffusers import DPMSolverMultistepScheduler as scheduler_cls  # type: ignore
            elif scheduler_name in {"euler_a", "euler-ancestral"}:
                from diffusers import EulerAncestralDiscreteScheduler as scheduler_cls  # type: ignore
            elif scheduler_name in {"euler", "k_euler"}:
                from diffusers import EulerDiscreteScheduler as scheduler_cls  # type: ignore
            elif scheduler_name in {"ddim"}:
                from diffusers import DDIMScheduler as scheduler_cls  # type: ignore
        except Exception:  # pragma: no cover - fallback if scheduler import fails
            scheduler_cls = None

        if scheduler_cls is not None:
            try:
                self._pipe.scheduler = scheduler_cls.from_config(self._pipe.scheduler.config)
            except Exception:
                logger.warning("Failed to set scheduler '%s'; using default", scheduler_name)

        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()

        if self._device == "cuda" and torch.cuda.is_available():
            self._pipe.to("cuda")
            self._pipe.enable_attention_slicing(1)
        else:
            self._pipe.to(self._device, torch.float32)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self._ensure_pipeline()

        set_random_seed(self._settings.seed, enable_cuda=self._device == "cuda")

        np_mask = np.array(mask.convert("L"))
        coords = np.argwhere(np_mask > 0)
        if coords.size == 0:
            logger.warning("Stable Diffusion inpainting received empty mask; returning original image")
            return image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        padding = self._settings.padding
        width, height = image.size
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        crop_box = (x_min, y_min, x_max, y_max)
        logger.info("Stable Diffusion crop box: %s", crop_box)

        cropped_image = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box).convert("L")
        feather_radius = max(0, int(self._settings.mask_feather))
        if feather_radius > 0:
            soft_mask = cropped_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        else:
            soft_mask = cropped_mask

        logger.info(
            "Running Stable Diffusion inpainting (size=%s) steps=%d guidance=%.2f",
            cropped_image.size,
            self._settings.num_inference_steps,
            self._settings.guidance_scale,
        )

        pipe_kwargs: dict[str, object] = {
            "prompt": self._settings.prompt,
            "negative_prompt": self._settings.negative_prompt,
            "image": cropped_image,
            "mask_image": cropped_mask,
            "num_inference_steps": self._settings.num_inference_steps,
            "guidance_scale": self._settings.guidance_scale,
            "strength": self._settings.strength,
        }

        if self._settings.guidance_rescale is not None:
            pipe_kwargs["guidance_rescale"] = self._settings.guidance_rescale

        generator = None
        if self._settings.seed is not None:
            try:
                import torch

                gen_device = "cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(self._settings.seed)
            except Exception:
                logger.warning("Failed to create torch.Generator; ignoring provided seed")

        if generator is not None:
            pipe_kwargs["generator"] = generator

        try:
            result = self._pipe(**pipe_kwargs).images[0]
        except TypeError as exc:
            if "guidance_rescale" in pipe_kwargs:
                logger.debug("Pipeline does not support guidance_rescale: %s", exc)
                pipe_kwargs.pop("guidance_rescale", None)
                result = self._pipe(**pipe_kwargs).images[0]
            else:
                raise

        if result.size != cropped_image.size:
            logger.debug(
                "Stable Diffusion result size %s differs from crop size %s; resizing",
                result.size,
                cropped_image.size,
            )
            resample = getattr(Image, "Resampling", Image).LANCZOS
            result = result.resize(cropped_image.size, resample)

        blend_factor = self._settings.blend_with_original
        if isinstance(blend_factor, (int, float)):
            blend_factor = max(0.0, min(float(blend_factor), 1.0))
        else:
            blend_factor = 0.0

        if blend_factor > 0:
            result = Image.blend(result, cropped_image, blend_factor)

        blended = Image.composite(result, cropped_image, soft_mask)

        final_image = image.copy()
        final_image.paste(blended, (x_min, y_min))
        return final_image

    def close(self) -> None:
        pipeline = getattr(self, "_pipe", None)
        if pipeline is None:
            return
        try:
            pipeline.to("cpu")
        except Exception:  # pragma: no cover - defensive cleanup
            pass


