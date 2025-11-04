"""Data preprocessing utilities for watermark detection.

This module contains testable preprocessing functions separated from the Dataset class.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_mask_from_difference(
    watermarked: np.ndarray,
    clean: np.ndarray,
    threshold: int = 10
) -> np.ndarray:
    """Create binary mask from watermarked and clean image pair.
    
    This function is separated for easy testing.
    
    Args:
        watermarked: Image with watermark (RGB, uint8)
        clean: Clean image (RGB, uint8)
        threshold: Difference threshold for binarization
        
    Returns:
        Binary mask where watermark was present (float32, [0, 1])
        
    Raises:
        ValueError: If images have incompatible shapes
    """
    # Validate inputs
    if watermarked.shape != clean.shape:
        raise ValueError(
            f"Image shape mismatch: watermarked {watermarked.shape} != clean {clean.shape}"
        )
    
    if watermarked.dtype != np.uint8 or clean.dtype != np.uint8:
        raise ValueError(
            f"Invalid image dtype: expected uint8, got watermarked={watermarked.dtype}, "
            f"clean={clean.dtype}"
        )
    
    if len(watermarked.shape) != 3 or watermarked.shape[2] != 3:
        raise ValueError(
            f"Invalid image shape: expected (H, W, 3), got {watermarked.shape}"
        )
    
    # Convert to grayscale
    watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_RGB2GRAY)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(watermarked_gray, clean_gray)
    
    # Threshold to create binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Normalize to [0, 1]
    mask = mask.astype(np.float32) / 255.0
    
    return mask


def resize_if_needed(
    image: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Resize image to target shape if needed.
    
    Args:
        image: Input image (H, W, C)
        target_shape: Target shape (H, W, C)
        
    Returns:
        Resized image if needed, otherwise original
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
    if len(target_shape) != 3:
        raise ValueError(f"Expected 3D target shape, got {target_shape}")
    
    if image.shape == target_shape:
        return image
    
    # Check if only spatial dimensions differ
    if image.shape[2] != target_shape[2]:
        raise ValueError(
            f"Channel mismatch: image has {image.shape[2]} channels, "
            f"target has {target_shape[2]}"
        )
    
    h, w = target_shape[:2]
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    logger.debug(f"Resized image from {image.shape} to {resized.shape}")
    
    return resized


def validate_image_array(
    image: np.ndarray,
    name: str = "image"
) -> None:
    """Validate image array format and properties.
    
    Args:
        image: Image array to validate
        name: Name for error messages
        
    Raises:
        ValueError: If image is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be numpy array, got {type(image)}")
    
    if image.ndim != 3:
        raise ValueError(f"{name} must be 3D (H, W, C), got shape {image.shape}")
    
    if image.shape[2] not in [1, 3, 4]:
        raise ValueError(
            f"{name} must have 1, 3, or 4 channels, got {image.shape[2]}"
        )
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(
            f"{name} must be uint8 or float, got {image.dtype}"
        )
    
    if image.dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            raise ValueError(f"{name} uint8 values out of range [0, 255]")
    elif image.dtype in [np.float32, np.float64]:
        if image.min() < 0.0 or image.max() > 1.0:
            logger.warning(
                f"{name} float values may be out of expected range [0, 1]: "
                f"min={image.min():.3f}, max={image.max():.3f}"
            )


def validate_mask_array(
    mask: np.ndarray,
    name: str = "mask"
) -> None:
    """Validate mask array format and properties.
    
    Args:
        mask: Mask array to validate
        name: Name for error messages
        
    Raises:
        ValueError: If mask is invalid
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"{name} must be numpy array, got {type(mask)}")
    
    if mask.ndim not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D, got shape {mask.shape}")
    
    if mask.ndim == 3 and mask.shape[2] != 1:
        raise ValueError(f"{name} 3D mask must have 1 channel, got {mask.shape[2]}")
    
    if mask.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"{name} must be uint8 or float, got {mask.dtype}")
    
    if mask.dtype == np.uint8:
        unique_values = np.unique(mask)
        if not np.all(np.isin(unique_values, [0, 255])):
            logger.warning(f"{name} uint8 mask has non-binary values: {unique_values}")
    elif mask.dtype in [np.float32, np.float64]:
        if mask.min() < 0.0 or mask.max() > 1.0:
            raise ValueError(
                f"{name} float values out of range [0, 1]: "
                f"min={mask.min():.3f}, max={mask.max():.3f}"
            )


def normalize_image_array(image: np.ndarray) -> np.ndarray:
    """Normalize image array to [0, 1] float32 range.
    
    Args:
        image: Input image (uint8 or float)
        
    Returns:
        Normalized image as float32 in [0, 1]
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype in [np.float32, np.float64]:
        return image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {image.dtype}")


def denormalize_image_array(image: np.ndarray) -> np.ndarray:
    """Denormalize image array from [0, 1] to [0, 255] uint8.
    
    Args:
        image: Normalized image in [0, 1]
        
    Returns:
        Denormalized image as uint8 in [0, 255]
    """
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)
    elif image.dtype == np.uint8:
        return image
    else:
        raise ValueError(f"Unsupported dtype: {image.dtype}")

