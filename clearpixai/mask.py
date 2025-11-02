"""Mask generation utilities."""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .detection.base import DetectionResult, RegionPrediction

logger = logging.getLogger(__name__)


def _draw_polygon(mask: Image.Image, polygon) -> None:
    if not polygon:
        return
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(polygon), fill=255)


def _expand_polygon(polygon, ratio: float, image_size: tuple[int, int]):
    if not polygon:
        return polygon
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0.0, x_min - width * ratio)
    y_min = max(0.0, y_min - height * ratio)
    x_max = min(image_size[0], x_max + width * ratio)
    y_max = min(image_size[1], y_max + height * ratio)
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]


def predictions_to_mask(
    predictions: DetectionResult,
    image_size: tuple[int, int],
    expand_ratio: float = 0.15,
    dilate: int = 10,
    blur_radius: int = 5,
) -> Image.Image:
    """Convert detection results into a soft mask ready for inpainting."""

    mask = Image.new("L", image_size, 0)

    for region in predictions:
        if region.mask is not None:
            mask_array = np.where(region.mask, 255, 0).astype(np.uint8)
            region_mask = Image.fromarray(mask_array, mode="L")
            mask = Image.fromarray(
                np.maximum(np.array(mask, dtype=np.uint8), np.array(region_mask, dtype=np.uint8))
            )
            continue

        polygon = _expand_polygon(region.polygon, expand_ratio, image_size)
        _draw_polygon(mask, polygon)

    mask_np = np.array(mask, dtype=np.uint8)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    mask_img = Image.fromarray(mask_np, mode="L")
    if blur_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask_img


