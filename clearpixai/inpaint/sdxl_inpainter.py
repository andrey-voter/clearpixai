"""SDXL-based high-quality inpainting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .base import BaseInpainter, InpainterSettings
from ..utils.random import set_random_seed

logger = logging.getLogger(__name__)


@dataclass
class SDXLInpainterSettings(InpainterSettings):
    """SDXL inpainting configuration."""
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt: str = (
        "high quality, photorealistic, clean surface, seamless, natural lighting, "
        "professional photography, 8k uhd, sharp focus"
    )
    negative_prompt: str = (
        "watermark, text, logo, signature, writing, letters, words, blurry, "
        "distorted, artifacts, low quality, jpeg artifacts, duplicate"
    )
    num_inference_steps: int = 50
    guidance_scale: float = 8.0
    strength: float = 0.999
    padding: int = 32
    scheduler: str = "euler"
    guidance_rescale: float | None = 0.7
    seed: int | None = None
    blend_with_original: float = 0.0
    mask_feather: int = 8


class SDXLInpainter(BaseInpainter):
    """SDXL-based inpainter for high-quality results."""
    
    def __init__(
        self,
        device: str,
        dtype="auto",
        settings: SDXLInpainterSettings | None = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(device, dtype, settings or SDXLInpainterSettings(), cache_dir)
        self._pipe = None
    
    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        
        settings = self._settings
        logger.info("Loading SDXL inpainting pipeline: %s", settings.model_id)
        
        try:
            from diffusers import AutoPipelineForInpainting
            import torch
        except ImportError as e:
            raise ImportError(
                "diffusers and torch are required for SDXL inpainting. "
                "Install them with: pip install diffusers torch"
            ) from e
        
        # Determine device and dtype
        if self._device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
                dtype = torch.float32
            else:
                device = self._device
                dtype = torch.float16  # Use FP16 for SDXL on GPU
        else:
            device = self._device
            dtype = torch.float32
        
        # Load the pipeline with retry logic for network issues
        import time
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self._pipe = AutoPipelineForInpainting.from_pretrained(
                    settings.model_id,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                    use_safetensors=True,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None,
                    resume_download=True,  # Resume interrupted downloads
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Failed to load model (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        str(e)[:100],
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Failed to load model after %d attempts: %s", max_retries, e)
                    raise
        
        # Set scheduler
        scheduler_name = (settings.scheduler or "").lower()
        scheduler_cls = None
        try:
            if scheduler_name in {"dpm++", "dpmsolver", "dpmsolver++", "dpm-solver"}:
                from diffusers import DPMSolverMultistepScheduler as scheduler_cls
            elif scheduler_name in {"euler_a", "euler-ancestral"}:
                from diffusers import EulerAncestralDiscreteScheduler as scheduler_cls
            elif scheduler_name in {"euler", "k_euler"}:
                from diffusers import EulerDiscreteScheduler as scheduler_cls
            elif scheduler_name in {"ddim"}:
                from diffusers import DDIMScheduler as scheduler_cls
        except Exception:
            scheduler_cls = None
        
        if scheduler_cls is not None:
            try:
                self._pipe.scheduler = scheduler_cls.from_config(self._pipe.scheduler.config)
            except Exception:
                logger.warning("Failed to set scheduler '%s'; using default", scheduler_name)
        
        # Enable memory optimizations
        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        
        # Move to device
        if device == "cuda":
            self._pipe.to(device)
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing(1)
            # Enable model CPU offload for large models like SDXL
            if hasattr(self._pipe, "enable_model_cpu_offload"):
                self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(device, torch.float32)
        
        logger.info("SDXL pipeline loaded successfully on %s", device)
    
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint using SDXL."""
        self._ensure_pipeline()
        settings = self._settings
        
        set_random_seed(settings.seed, enable_cuda=self._device == "cuda")
        
        # Check if mask has content
        np_mask = np.array(mask.convert("L"))
        coords = np.argwhere(np_mask > 0)
        if coords.size == 0:
            logger.warning("SDXL inpainting received empty mask; returning original image")
            return image
        
        # Calculate crop region with padding
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        padding = settings.padding
        width, height = image.size
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # Ensure dimensions are multiples of 8 (required for SDXL)
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        crop_width = ((crop_width + 7) // 8) * 8
        crop_height = ((crop_height + 7) // 8) * 8
        
        # Adjust bounds to fit rounded dimensions
        if x_min + crop_width > width:
            x_min = max(0, width - crop_width)
        x_max = x_min + crop_width
        
        if y_min + crop_height > height:
            y_min = max(0, height - crop_height)
        y_max = y_min + crop_height
        
        crop_box = (x_min, y_min, x_max, y_max)
        logger.info("SDXL crop box: %s (size: %dx%d)", crop_box, crop_width, crop_height)
        
        # Crop image and mask
        cropped_image = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box).convert("L")
        
        # Apply mask feathering
        feather_radius = max(0, int(settings.mask_feather))
        if feather_radius > 0:
            soft_mask = cropped_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        else:
            soft_mask = cropped_mask
        
        logger.info(
            "Running SDXL inpainting (size=%dx%d) steps=%d guidance=%.2f",
            crop_width, crop_height,
            settings.num_inference_steps,
            settings.guidance_scale,
        )
        
        # Prepare pipeline kwargs
        pipe_kwargs = {
            "prompt": settings.prompt,
            "negative_prompt": settings.negative_prompt,
            "image": cropped_image,
            "mask_image": cropped_mask,
            "num_inference_steps": settings.num_inference_steps,
            "guidance_scale": settings.guidance_scale,
            "strength": settings.strength,
        }
        
        if settings.guidance_rescale is not None:
            pipe_kwargs["guidance_rescale"] = settings.guidance_rescale
        
        # Set up generator for reproducibility
        generator = None
        if settings.seed is not None:
            try:
                import torch
                gen_device = "cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(settings.seed)
            except Exception:
                logger.warning("Failed to create torch.Generator; ignoring provided seed")
        
        if generator is not None:
            pipe_kwargs["generator"] = generator
        
        # Run inpainting
        try:
            result = self._pipe(**pipe_kwargs).images[0]
        except TypeError as exc:
            if "guidance_rescale" in pipe_kwargs:
                logger.debug("Pipeline does not support guidance_rescale: %s", exc)
                pipe_kwargs.pop("guidance_rescale", None)
                result = self._pipe(**pipe_kwargs).images[0]
            else:
                raise
        
        # Ensure result size matches crop size
        if result.size != cropped_image.size:
            logger.debug(
                "SDXL result size %s differs from crop size %s; resizing",
                result.size,
                cropped_image.size,
            )
            resample = getattr(Image, "Resampling", Image).LANCZOS
            result = result.resize(cropped_image.size, resample)
        
        # Optional blending with original
        blend_factor = settings.blend_with_original
        if isinstance(blend_factor, (int, float)):
            blend_factor = max(0.0, min(float(blend_factor), 1.0))
        else:
            blend_factor = 0.0
        
        if blend_factor > 0:
            result = Image.blend(result, cropped_image, blend_factor)
        
        # Composite with soft mask
        blended = Image.composite(result, cropped_image, soft_mask)
        
        # Paste back into original image
        final_image = image.copy()
        final_image.paste(blended, (x_min, y_min))
        return final_image
    
    def close(self) -> None:
        """Release pipeline resources."""
        pipeline = getattr(self, "_pipe", None)
        if pipeline is None:
            return
        try:
            pipeline.to("cpu")
            del self._pipe
            self._pipe = None
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

