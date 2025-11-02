"""Core watermark removal pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image

from .detection.base import DetectionError, DetectionResult, DetectorNotAvailable
from .detection.easyocr import EasyOCRDetector
from .detection.florence2 import Florence2Detector
from .detection.grounding_sam import GroundingSAMDetector
from .inpaint.opencv import inpaint as opencv_inpaint
from .inpaint.stable_diffusion import StableDiffusionInpainter, StableDiffusionSettings
from .mask import predictions_to_mask
from .utils.random import set_random_seed

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "clearpixai" / "weights"


@dataclass
class MaskConfig:
    expand_ratio: float = 0.15
    dilate: int = 10
    blur_radius: int = 5


@dataclass
class DiffusionConfig:
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    # model_id: str= "stabilityai/stable-diffusion-xl-inpainting-1.0"
    prompt: str = (
        "clean background, people look natural, seamless surface, no watermark, no text, natural texture, high quality"
    )
    negative_prompt: str = (
        "watermark, text, logo, signature, writing, letters, words, blurry, distorted, artifacts, objects"
    )
    num_inference_steps: int = 65
    guidance_scale: float = 10.0
    strength: float = 0.99
    padding: int = 5
    scheduler: str = "dpm++"
    guidance_rescale: float | None = None
    seed: int | None = None
    blend_with_original: float = 0.0
    mask_feather: int = 2


@dataclass
class DetectionConfig:
    detector: str = "easyocr"
    fallback_detectors: Sequence[str] = field(default_factory=lambda: ("easyocr",))
    prompt: str | None = None
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    device: str = "auto"
    hf_token: str | None = None
    easyocr_languages: Sequence[str] = field(default_factory=lambda: ("en",))
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    florence_num_beams: int = 3
    florence_max_new_tokens: int = 256
    seed: int | None = None


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    mode: str = "fast"  # fast | quality
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


def _build_detector(name: str, cfg: DetectionConfig, device: str):
    name = name.lower()
    if name == "easyocr":
        gpu = device == "cuda"
        return EasyOCRDetector(languages=cfg.easyocr_languages, gpu=gpu)
    if name in {"grounding_sam", "groundingsam", "grounding-dino"}:
        return GroundingSAMDetector(
            weights_dir=cfg.weights_dir,
            text_prompt=cfg.prompt,
            device=device,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )
    if name in {"florence2", "florence"}:
        return Florence2Detector(
            prompt=cfg.prompt,
            device=device,
            num_beams=cfg.florence_num_beams,
            max_new_tokens=cfg.florence_max_new_tokens,
            hf_token=cfg.hf_token,
        )
    raise ValueError(f"Unknown detector '{name}'")


def _unique_sequence(sequence: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in sequence:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _detect_with_fallbacks(
    image: Image.Image,
    cfg: DetectionConfig,
    global_seed: int | None = None,
) -> DetectionResult:
    device = _auto_device(cfg.device)
    resolved_seed = cfg.seed if cfg.seed is not None else global_seed
    candidates = _unique_sequence((cfg.detector,) + tuple(cfg.fallback_detectors))
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            set_random_seed(resolved_seed, enable_cuda=device == "cuda")
            detector = _build_detector(candidate, cfg, device)
        except DetectorNotAvailable as exc:
            logger.warning("Detector '%s' unavailable: %s", candidate, exc)
            last_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialise detector '%s'", candidate)
            last_error = exc
            continue

        try:
            set_random_seed(resolved_seed, enable_cuda=device == "cuda")
            result = detector.detect(image)
            if result and not result.is_empty():
                logger.info("Detector '%s' succeeded", candidate)
                return result
            logger.warning("Detector '%s' returned no regions", candidate)
        except DetectionError as exc:
            logger.warning("Detector '%s' failed: %s", candidate, exc)
            last_error = exc
        except Exception as exc:  # pragma: no cover - catch-all
            logger.exception("Detector '%s' crashed", candidate)
            last_error = exc
        finally:
            close = getattr(detector, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass

    if last_error:
        raise DetectionError(f"All detectors failed; last error: {last_error}") from last_error
    raise DetectionError("All detectors failed to find any regions")


def remove_watermark(input_path: Path, output_path: Path, config: PipelineConfig) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Loading image from %s", input_path)
    image = Image.open(input_path).convert("RGB")
    config.detection.weights_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(config.seed)

    detection_result = _detect_with_fallbacks(
        image,
        config.detection,
        global_seed=config.seed,
    )

    if detection_result.is_empty():
        logger.warning("No watermarks detected; saving original image")
        image.save(output_path, quality=95)
        return output_path

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

    mode = config.mode.lower()
    if mode == "quality":
        logger.info("Using quality mode (Stable Diffusion)")
        settings = StableDiffusionSettings(
            model_id=config.diffusion.model_id,
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
        device = _auto_device(config.detection.device)
        inpainter = StableDiffusionInpainter(device=device, settings=settings)
        try:
            result = inpainter.inpaint(image, mask)
        finally:
            inpainter.close()
    else:
        logger.info("Using fast mode (OpenCV)")
        result = opencv_inpaint(image, mask)

    result.save(output_path, quality=95)
    logger.info("Saved cleaned image to %s", output_path)
    return output_path


