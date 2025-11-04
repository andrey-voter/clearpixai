"""Post-processing utilities for model predictions.

This module handles conversion of raw model outputs to usable predictions.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to probabilities using sigmoid.
    
    Args:
        logits: Raw model output logits (any shape)
        
    Returns:
        Probabilities in [0, 1] range (same shape as input)
        
    Raises:
        ValueError: If input is not a tensor
    """
    if not isinstance(logits, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(logits)}")
    
    return torch.sigmoid(logits)


def probabilities_to_binary_mask(
    probabilities: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Convert probability map to binary mask.
    
    Args:
        probabilities: Probability values in [0, 1]
        threshold: Classification threshold
        
    Returns:
        Binary mask (0 or 1)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(probabilities, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(probabilities)}")
    
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
    
    # Check if probabilities are in valid range
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        logger.warning(
            f"Probabilities out of range [0, 1]: "
            f"min={probabilities.min():.3f}, max={probabilities.max():.3f}"
        )
    
    return (probabilities > threshold).float()


def logits_to_binary_mask(
    logits: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Convert logits directly to binary mask (convenience function).
    
    Args:
        logits: Raw model output
        threshold: Classification threshold
        
    Returns:
        Binary mask (0 or 1)
    """
    probabilities = logits_to_probabilities(logits)
    return probabilities_to_binary_mask(probabilities, threshold)


def mask_tensor_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """Convert PyTorch mask tensor to NumPy array.
    
    Args:
        mask: PyTorch tensor (B, C, H, W) or (C, H, W) or (H, W)
        
    Returns:
        NumPy array (H, W) with values in [0, 1]
        
    Raises:
        ValueError: If input shape is invalid
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(mask)}")
    
    # Detach and move to CPU
    mask_np = mask.detach().cpu().numpy()
    
    # Handle different input shapes
    if mask_np.ndim == 4:  # (B, C, H, W)
        # Take first batch item and first channel
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:  # (C, H, W) or (B, H, W)
        # Take first element
        mask_np = mask_np[0]
    elif mask_np.ndim == 2:  # (H, W)
        pass
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {mask.shape}")
    
    # Ensure values are in [0, 1]
    mask_np = np.clip(mask_np, 0.0, 1.0)
    
    return mask_np


def mask_numpy_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert float mask to uint8 format.
    
    Args:
        mask: Float mask in [0, 1]
        
    Returns:
        Uint8 mask in [0, 255]
        
    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(mask)}")
    
    if mask.dtype not in [np.float32, np.float64, np.uint8]:
        raise ValueError(f"Expected float or uint8, got {mask.dtype}")
    
    if mask.dtype == np.uint8:
        return mask
    
    # Convert float to uint8
    mask = np.clip(mask, 0.0, 1.0)
    return (mask * 255).astype(np.uint8)


def validate_model_output(
    output: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """Validate model output tensor.
    
    Args:
        output: Model output tensor
        expected_shape: Expected shape (optional, checks batch/channel dims if provided)
        
    Raises:
        ValueError: If output is invalid
    """
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(output)}")
    
    if output.ndim not in [2, 3, 4]:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {output.shape}")
    
    if expected_shape is not None:
        if len(expected_shape) != len(output.shape):
            raise ValueError(
                f"Shape dimension mismatch: expected {len(expected_shape)} dims, "
                f"got {len(output.shape)} dims"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, output.shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                )


def extract_prediction_for_api(
    logits: torch.Tensor,
    threshold: float = 0.5,
    output_format: str = "numpy"
) -> dict:
    """Extract and format prediction for API response.
    
    This function handles the complete pipeline from raw model output to API-ready format.
    
    Args:
        logits: Raw model output (B, C, H, W)
        threshold: Classification threshold
        output_format: Output format ('numpy', 'list', 'binary')
        
    Returns:
        Dictionary with:
            - probabilities: Probability map
            - binary_mask: Binary prediction
            - confidence: Average confidence score
            - has_watermark: Boolean indicating watermark presence
            
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate input
    validate_model_output(logits)
    
    if output_format not in ['numpy', 'list', 'binary']:
        raise ValueError(f"Invalid output_format: {output_format}")
    
    # Convert to probabilities
    probabilities = logits_to_probabilities(logits)
    
    # Convert to binary mask
    binary_mask = probabilities_to_binary_mask(probabilities, threshold)
    
    # Calculate statistics
    confidence = probabilities.mean().item()
    max_confidence = probabilities.max().item()
    watermark_ratio = binary_mask.mean().item()
    has_watermark = watermark_ratio > 0.01  # More than 1% of pixels
    
    # Format output
    result = {
        'has_watermark': bool(has_watermark),
        'confidence': float(confidence),
        'max_confidence': float(max_confidence),
        'watermark_ratio': float(watermark_ratio),
        'threshold': float(threshold),
    }
    
    if output_format == 'numpy':
        result['probabilities'] = mask_tensor_to_numpy(probabilities)
        result['binary_mask'] = mask_tensor_to_numpy(binary_mask)
    elif output_format == 'list':
        prob_np = mask_tensor_to_numpy(probabilities)
        mask_np = mask_tensor_to_numpy(binary_mask)
        result['probabilities'] = prob_np.tolist()
        result['binary_mask'] = mask_np.tolist()
    elif output_format == 'binary':
        # Only include binary decision, no arrays
        pass
    
    return result


def batch_predictions_to_list(
    logits: torch.Tensor,
    threshold: float = 0.5
) -> list:
    """Process batch of predictions for API.
    
    Args:
        logits: Batch of model outputs (B, C, H, W)
        threshold: Classification threshold
        
    Returns:
        List of prediction dictionaries, one per batch item
    """
    if not isinstance(logits, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(logits)}")
    
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {logits.shape}")
    
    results = []
    batch_size = logits.shape[0]
    
    for i in range(batch_size):
        single_logit = logits[i:i+1]  # Keep batch dimension
        result = extract_prediction_for_api(
            single_logit,
            threshold=threshold,
            output_format='binary'
        )
        results.append(result)
    
    return results

