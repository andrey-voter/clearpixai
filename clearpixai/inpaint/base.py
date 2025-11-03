"""Base inpainter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from PIL import Image


@dataclass
class InpainterSettings:
    """Base settings for inpainters."""
    seed: int | None = None


class BaseInpainter(ABC):
    """Abstract base class for image inpainters."""
    
    def __init__(
        self,
        device: str,
        dtype="auto",
        settings: InpainterSettings | None = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._settings = settings or InpainterSettings()
        self._cache_dir = cache_dir
    
    @abstractmethod
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Inpaint the masked regions of an image.
        
        Args:
            image: Input image (RGB)
            mask: Mask image (L mode, white=known, black=to inpaint)
            
        Returns:
            Inpainted image
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass

