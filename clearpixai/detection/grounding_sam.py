"""GroundingDINO + SAM detector implementation."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult, DetectorNotAvailable, RegionPrediction

logger = logging.getLogger(__name__)


DEFAULT_GROUNDING_MODEL = "groundingdino_swint_ogc.pth"
DEFAULT_SAM_MODEL = "sam_vit_h_4b8939.pth"


def _resolve_grounding_config() -> str:
    try:
        import importlib.resources as resources

        return str(resources.files("groundingdino").joinpath("config/GroundingDINO_SwinT_OGC.py"))
    except Exception as exc:  # pragma: no cover - dependency guard
        raise DetectorNotAvailable(
            "Unable to locate GroundingDINO config. Ensure the package is installed from GitHub."
        ) from exc


class GroundingSAMDetector(BaseDetector):
    name = "grounding_sam"

    def __init__(
        self,
        weights_dir: Path,
        text_prompt: str | None = None,
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            from groundingdino.util.inference import load_model, predict
            from segment_anything import SamPredictor, sam_model_registry
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DetectorNotAvailable(
                "GroundingDINO and Segment Anything must be installed from their GitHub repositories."
            ) from exc

        self._torch = torch
        self._predict = predict
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._prompt = text_prompt or "watermark. logo. text. signature."

        weights_dir = Path(weights_dir).expanduser().resolve()
        grounding_ckpt = weights_dir / DEFAULT_GROUNDING_MODEL
        sam_ckpt = weights_dir / DEFAULT_SAM_MODEL

        if not grounding_ckpt.exists():
            raise DetectorNotAvailable(
                f"GroundingDINO checkpoint not found at {grounding_ckpt}. "
                "Download it as documented in INSTALL_GROUNDING_SAM.md."
            )
        if not sam_ckpt.exists():
            raise DetectorNotAvailable(
                f"SAM checkpoint not found at {sam_ckpt}. "
                "Download it as documented in INSTALL_GROUNDING_SAM.md."
            )

        config_path = _resolve_grounding_config()
        logger.info("Loading GroundingDINO model from %s", config_path)
        self._grounding_model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=str(grounding_ckpt),
            device=device,
        )

        logger.info("Loading SAM model from %s", sam_ckpt)
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
        sam.to(device=device)
        self._sam_predictor = SamPredictor(sam)

    def detect(self, image: Image.Image) -> DetectionResult:
        logger.info("Detecting regions with GroundingDINO + SAM using prompt: %s", self._prompt)
        img_array = np.array(image)

        boxes, scores, phrases = self._predict(
            model=self._grounding_model,
            image=img_array,
            caption=self._prompt,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            device=self._device,
        )

        num_boxes = len(boxes)
        logger.info("GroundingDINO produced %d candidate boxes", num_boxes)
        if num_boxes == 0:
            return DetectionResult([])

        self._sam_predictor.set_image(img_array)
        h, w = img_array.shape[:2]
        boxes_xyxy = boxes * self._torch.tensor([w, h, w, h], device=self._device)

        predictions: List[RegionPrediction] = []

        for idx, box in enumerate(boxes_xyxy):
            box_np = box.detach().cpu().numpy()
            masks, _, _ = self._sam_predictor.predict(box=box_np, multimask_output=False)
            mask = np.asarray(masks[0], dtype=np.uint8)

            mask_bool = mask.astype(bool)
            coords = np.argwhere(mask_bool)
            if coords.size == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            polygon = [
                (float(x_min), float(y_min)),
                (float(x_max), float(y_min)),
                (float(x_max), float(y_max)),
                (float(x_min), float(y_max)),
            ]
            score = float(scores[idx]) if idx < len(scores) else None
            label = phrases[idx] if idx < len(phrases) else None
            predictions.append(RegionPrediction(polygon=polygon, score=score, label=label, mask=mask_bool))

        logger.info("GroundingDINO + SAM produced %d refined masks", len(predictions))
        return DetectionResult(predictions)

    def close(self) -> None:
        try:
            self._grounding_model.to("cpu")
        except Exception:  # pragma: no cover - defensive
            pass


