"""Detection interfaces and shared dataclasses for ClearPixAi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


Polygon = Sequence[Tuple[float, float]]
MaskArray = np.ndarray


@dataclass
class RegionPrediction:
    """Represents a detected watermark region."""

    polygon: Optional[Polygon] = None
    score: Optional[float] = None
    label: Optional[str] = None
    mask: Optional[MaskArray] = None


class DetectionError(RuntimeError):
    """Base class for detection-related failures."""


class DetectorNotAvailable(DetectionError):
    """Raised when a detector cannot be instantiated (missing deps, weights, etc.)."""


class DetectionResult:
    """Container for the output of a detector."""

    def __init__(self, regions: Iterable[RegionPrediction]):
        self.regions: List[RegionPrediction] = list(regions)

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def is_empty(self) -> bool:
        return len(self.regions) == 0

    def __bool__(self) -> bool:  # pragma: no cover - small proxy
        return not self.is_empty()


class BaseDetector:
    """Abstract interface for all detectors."""

    name: str = "base"

    def detect(self, image: Image.Image) -> DetectionResult:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional override
        """Free any underlying resources (GPU memory, external handles, etc.)."""


