"""Detection modules exposed for ClearPixAi."""

from .base import BaseDetector, DetectionResult, DetectorNotAvailable, RegionPrediction
from .easyocr import EasyOCRDetector
from .florence2 import Florence2Detector
from .grounding_sam import GroundingSAMDetector
from .segmentation import WatermarkSegmentationDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "DetectorNotAvailable",
    "RegionPrediction",
    "EasyOCRDetector",
    "Florence2Detector",
    "GroundingSAMDetector",
    "WatermarkSegmentationDetector",
]


