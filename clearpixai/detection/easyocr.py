"""EasyOCR-based watermark detector."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult, RegionPrediction

logger = logging.getLogger(__name__)


class EasyOCRDetector(BaseDetector):
    name = "easyocr"

    def __init__(self, languages: Sequence[str] | None = None, gpu: bool | None = None):
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "EasyOCR is not installed. Install it with `uv pip install easyocr`."
            ) from exc

        self.languages: Sequence[str] = languages or ("en",)
        self.reader = easyocr.Reader(list(self.languages), gpu=gpu)

    def detect(self, image: Image.Image) -> DetectionResult:
        logger.info("Detecting regions with EasyOCR (languages=%s)", ",".join(self.languages))
        img_array = np.array(image)
        results: Iterable = self.reader.readtext(img_array)

        predictions: List[RegionPrediction] = []
        for bbox, text, score in results:
            polygon = [(float(x), float(y)) for x, y in bbox]
            predictions.append(RegionPrediction(polygon=polygon, score=float(score), label=text))

        logger.info("EasyOCR found %d region(s)", len(predictions))
        return DetectionResult(predictions)


