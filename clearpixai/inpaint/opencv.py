"""OpenCV based inpainting utilities."""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def inpaint(
    image: Image.Image,
    mask: Image.Image,
    radius: int = 7,
    method: Literal["telea", "ns"] = "telea",
) -> Image.Image:
    """Run fast OpenCV inpainting."""

    logger.info("Running OpenCV inpainting (%s, radius=%d)", method.upper(), radius)

    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_array = np.array(mask.convert("L"))
    mask_uint8 = np.where(mask_array > 0, 255, 0).astype("uint8")

    result = cv2.inpaint(img_cv, mask_uint8, inpaintRadius=radius, flags=flags)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


