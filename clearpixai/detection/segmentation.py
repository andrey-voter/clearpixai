"""Segmentation-based watermark detector using Diffusion Dynamics model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult, DetectorNotAvailable, RegionPrediction

logger = logging.getLogger(__name__)


def _load_state_dict(model, state_dict):
    """Load state dict while stripping common Lightning prefixes."""

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned[key[len("model."):]] = value
        elif key.startswith("net."):
            cleaned[key[len("net."):]] = value
        else:
            cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing keys when loading segmentation weights: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading segmentation weights: %s", unexpected)


class WatermarkSegmentationDetector(BaseDetector):
    """Detector wrapping Diffusion-Dynamics watermark segmentation model."""

    name = "segmentation"

    def __init__(
        self,
        weights_path: Path,
        encoder_name: str = "mit_b5",
        encoder_weights: Optional[str] = None,
        image_size: Optional[int] = None,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        try:
            import segmentation_models_pytorch as smp
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DetectorNotAvailable(
                "Segmentation detector requires `segmentation-models-pytorch`. Install it with `uv pip install segmentation-models-pytorch`."
            ) from exc

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise DetectorNotAvailable(
                f"Segmentation weights not found at {weights_path}. Download the checkpoint from Diffusion Dynamics watermark-segmentation repo."
            )

        self._torch = torch
        self._device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._threshold = float(threshold)
        self._image_size = image_size

        logger.info("Loading segmentation model (encoder=%s)", encoder_name)
        self._model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

        checkpoint = torch.load(str(weights_path), map_location="cpu")
        _load_state_dict(self._model, checkpoint)

        self._model.to(self._device)
        self._model.eval()

        preprocessing_fn = None
        try:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
        except ValueError:  # pragma: no cover - optional
            logger.debug("No preprocessing function for encoder %s", encoder_name)

        self._preprocess = preprocessing_fn

    def _prepare_tensor(self, image: Image.Image) -> "torch.Tensor":
        import torch

        np_image = np.array(image).astype("float32")
        if self._image_size:
            np_image = np.array(image.resize((self._image_size, self._image_size), Image.BICUBIC)).astype("float32")

        if self._preprocess:
            np_image = self._preprocess(np_image)
        else:
            np_image = np_image / 255.0

        tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device, dtype=torch.float32)

    def detect(self, image: Image.Image) -> DetectionResult:
        torch = self._torch
        original_size = image.size

        tensor = self._prepare_tensor(image.convert("RGB"))

        with torch.no_grad():
            logits = self._model(tensor)
            mask = torch.sigmoid(logits)[0, 0]

        mask_np = mask.cpu().numpy()
        binary_mask = (mask_np >= self._threshold).astype(np.uint8)

        if self._image_size and (self._image_size != original_size[0] or self._image_size != original_size[1]):
            resized = Image.fromarray(binary_mask * 255).resize(original_size, Image.BILINEAR)
            binary_mask = (np.array(resized) > 127).astype(np.uint8)

        predictions: List[RegionPrediction] = [
            RegionPrediction(mask=binary_mask.astype(bool), score=float(mask_np.max()))
        ]

        logger.info("Segmentation produced mask with %.2f%% foreground", binary_mask.mean() * 100)
        return DetectionResult(predictions)

    def close(self) -> None:
        try:
            self._model.to("cpu")
        except Exception:  # pragma: no cover
            pass
