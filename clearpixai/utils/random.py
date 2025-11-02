"""Random seeding utilities."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch optional for CPU-only installs
    torch = None  # type: ignore


def set_random_seed(seed: Optional[int], enable_cuda: bool = True) -> None:
    """Seed Python, NumPy, and torch (if available).

    Parameters
    ----------
    seed:
        Seed value to apply. If ``None`` this function is a no-op.
    enable_cuda:
        When ``True`` and CUDA is available, ``torch.cuda.manual_seed_all`` is also invoked.
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed % (2**32))

    if torch is None:
        return

    try:
        torch.manual_seed(seed)
        if enable_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to apply torch seed: %s", exc)

