"""Florence-2 based watermark detector."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from .base import BaseDetector, DetectionResult, DetectorNotAvailable, RegionPrediction

logger = logging.getLogger(__name__)


class Florence2Detector(BaseDetector):
    name = "florence2"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        prompt: str | None = None,
        device: str = "cuda",
        num_beams: int = 3,
        max_new_tokens: int = 1024,
        task_token: str = "<OCR_WITH_REGION>",
        hf_token: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DetectorNotAvailable(
                "Florence2 detector requires `transformers`, `torch`, `timm`, and `einops`."
            ) from exc

        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv(
                "HUGGINGFACEHUB_API_TOKEN"
            )

        if hf_token:
            try:
                from huggingface_hub import login

                login(token=hf_token, add_to_git_credential=False)
            except Exception as exc:  # pragma: no cover - user facing warning
                logger.warning("Failed to authenticate with Hugging Face Hub: %s", exc)

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32

        logger.info("Loading Florence-2 model %s", model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=dtype,
            token=hf_token,
        )
        self._model.to(device)
        self._model.eval()

        self._device = device
        self._dtype = dtype
        self._text_input = (prompt or "").strip()
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens
        self._task_token = task_token.strip() or "<OCR_WITH_REGION>"

    def detect(self, image: Image.Image) -> DetectionResult:
        task = self._task_token
        if self._text_input:
            logger.debug(
                "Ignoring Florence-2 prompt '%s' because <OCR_WITH_REGION> expects only the task token.",
                self._text_input,
            )

        prompt = task
        logger.info("Detecting regions with Florence-2 using prompt: %s", prompt)

        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                early_stopping=False,
                num_beams=self._num_beams,
                use_cache=False,
            )

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logger.debug("Florence-2 raw output: %s", text)

        parsed = self._processor.post_process_generation(text=text, task=task, image_size=image.size)

        task_key = task
        payload = parsed.get(task_key, {}) if isinstance(parsed, dict) else {}

        predictions: List[RegionPrediction] = []

        quad_boxes: Sequence[Sequence[float]] = payload.get("quad_boxes", []) or []
        labels: Sequence[str] = payload.get("labels", []) or []
        scores: Sequence[float] = payload.get("scores", []) or []

        for idx, quad in enumerate(quad_boxes):
            polygon = _quad_to_polygon(quad)
            if not polygon:
                continue
            label = labels[idx] if idx < len(labels) else None
            score = scores[idx] if idx < len(scores) else None
            predictions.append(RegionPrediction(polygon=polygon, label=label, score=score))

        logger.info("Florence-2 produced %d OCR region(s)", len(predictions))
        return DetectionResult(predictions)

    def close(self) -> None:
        # Move model back to CPU to free GPU memory if needed.
        try:
            self._model.to("cpu")
        except Exception:  # pragma: no cover - defensive
            pass


def _quad_to_polygon(quad: Iterable[float]) -> List[tuple[float, float]]:
    if quad is None:
        return []

    try:
        values = [float(v) for v in quad]
    except (TypeError, ValueError):
        return []

    if len(values) == 8:
        points = list(zip(values[0::2], values[1::2]))
    elif len(values) == 4:
        x_min, y_min, x_max, y_max = values
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    else:
        return []

    return [(float(x), float(y)) for x, y in points]


