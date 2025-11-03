"""Detection modules exposed for ClearPixAi."""

from .base import BaseDetector, DetectionResult, DetectorNotAvailable, RegionPrediction
from .segmentation import WatermarkSegmentationDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "DetectorNotAvailable",
    "RegionPrediction",
    "WatermarkSegmentationDetector",
]
