"""Core watermark removal pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .detection.segmentation import WatermarkSegmentationDetector
from .inpaint import (
    BaseInpainter,
    StableDiffusionInpainter,
    StableDiffusionSettings,
    SDXLInpainter,
    SDXLInpainterSettings,
)
from .mask import predictions_to_mask
from .utils.random import set_random_seed

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "clearpixai" / "weights"


@dataclass
class MaskConfig:
    """Mask processing configuration.
    
    These defaults are the single source of truth.
    CLI only overrides when explicitly provided.
    """
    expand_ratio: float = 0.15
    dilate: int = 10
    blur_radius: int = 5


@dataclass
class DiffusionConfig:
    """Diffusion inpainting configuration.
    
    These defaults are the single source of truth.
    CLI only overrides when explicitly provided.
    """
    backend: str = "sdxl"  # Options: "sd", "sdxl"
    model_id: str | None = None  # Auto-selected based on backend if None
    prompt: str = (
        "high quality, photorealistic, clean surface, seamless, natural lighting, "
        "professional photography, sharp focus"
    )
    negative_prompt: str = (
        "watermark, text, logo, signature, writing, letters, words, blurry, "
        "distorted, artifacts, low quality, jpeg artifacts"
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


@dataclass
class SegmentationConfig:
    """Segmentation detection configuration.
    
    These defaults are the single source of truth.
    CLI only overrides when explicitly provided.
    """
    weights: Path = Path(__file__).parent / "detection" / "best_watermark_model_mit_b5_best.pth"
    encoder: str = "mit_b5"
    encoder_weights: str | None = None
    image_size: int | None = None
    threshold: float = 0.004
    device: str = "auto"
    seed: int | None = None


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    save_mask: bool = False
    seed: int | None = None


def _auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def remove_watermark(input_path: Path, output_path: Path, config: PipelineConfig) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Loading image from %s", input_path)
    image = Image.open(input_path).convert("RGB")

    set_random_seed(config.seed)

    # Validate segmentation weights
    if not config.segmentation.weights.exists():
        raise ValueError(
            f"Segmentation weights not found at {config.segmentation.weights}. "
            "Please provide --segmentation-weights pointing to a valid checkpoint, "
            "or place the weights at the default location."
        )

    # Initialize segmentation detector
    device = _auto_device(config.segmentation.device)
    resolved_seed = config.segmentation.seed if config.segmentation.seed is not None else config.seed
    set_random_seed(resolved_seed, enable_cuda=device == "cuda")

    logger.info("Initializing segmentation detector")
    detector = WatermarkSegmentationDetector(
        weights_path=config.segmentation.weights,
        encoder_name=config.segmentation.encoder,
        encoder_weights=config.segmentation.encoder_weights,
        image_size=config.segmentation.image_size,
        threshold=config.segmentation.threshold,
        device=device,
    )

    try:
        set_random_seed(resolved_seed, enable_cuda=device == "cuda")
        detection_result = detector.detect(image)
    finally:
        detector.close()

    if detection_result.is_empty():
        logger.warning("No watermarks detected; saving original image")
        image.save(output_path, quality=95)
        return output_path

    # Generate mask from detection result
    mask = predictions_to_mask(
        detection_result,
        image_size=image.size,
        expand_ratio=config.mask.expand_ratio,
        dilate=config.mask.dilate,
        blur_radius=config.mask.blur_radius,
    )

    if config.save_mask:
        mask_path = output_path.with_name(f"{output_path.stem}_mask.png")
        logger.info("Saving mask to %s", mask_path)
        mask.save(mask_path)

    # Select and configure inpainter based on backend
    backend = (config.diffusion.backend or "sdxl").lower()
    logger.info("Inpainting with backend: %s", backend)
    
    inpainter: BaseInpainter
    if backend == "sd":
        # Legacy Stable Diffusion 2.0
        model_id = config.diffusion.model_id or "stabilityai/stable-diffusion-2-inpainting"
        settings = StableDiffusionSettings(
            model_id=model_id,
            prompt=config.diffusion.prompt,
            negative_prompt=config.diffusion.negative_prompt,
            num_inference_steps=config.diffusion.num_inference_steps,
            guidance_scale=config.diffusion.guidance_scale,
            strength=config.diffusion.strength,
            padding=config.diffusion.padding,
            scheduler=config.diffusion.scheduler,
            guidance_rescale=config.diffusion.guidance_rescale,
            seed=(
                config.diffusion.seed
                if config.diffusion.seed is not None
                else config.seed
            ),
            blend_with_original=config.diffusion.blend_with_original,
            mask_feather=config.diffusion.mask_feather,
        )
        inpainter = StableDiffusionInpainter(device=device, settings=settings)
    elif backend == "sdxl":
        # SDXL for better quality
        model_id = config.diffusion.model_id or "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        settings = SDXLInpainterSettings(
            model_id=model_id,
            prompt=config.diffusion.prompt,
            negative_prompt=config.diffusion.negative_prompt,
            num_inference_steps=config.diffusion.num_inference_steps,
            guidance_scale=config.diffusion.guidance_scale,
            strength=config.diffusion.strength,
            padding=config.diffusion.padding,
            scheduler=config.diffusion.scheduler,
            guidance_rescale=config.diffusion.guidance_rescale,
            seed=(
                config.diffusion.seed
                if config.diffusion.seed is not None
                else config.seed
            ),
            blend_with_original=config.diffusion.blend_with_original,
            mask_feather=config.diffusion.mask_feather,
        )
        inpainter = SDXLInpainter(device=device, settings=settings)
    else:
        raise ValueError(f"Unknown inpainting backend: {backend}. Choose 'sd' or 'sdxl'.")
    
    try:
        result = inpainter.inpaint(image, mask)
    finally:
        inpainter.close()

    result.save(output_path, quality=95)
    logger.info("Saved cleaned image to %s", output_path)
    return output_path
