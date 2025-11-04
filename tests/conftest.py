"""Pytest configuration and fixtures for testing."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create 64x64 RGB image with random values
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask for testing."""
    # Create 64x64 binary mask
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0  # Square watermark region
    return mask


@pytest.fixture
def sample_image_pair(sample_image, sample_mask):
    """Create a pair of watermarked and clean images."""
    watermarked = sample_image.copy()
    clean = sample_image.copy()
    
    # Add watermark to watermarked image
    mask_3d = np.expand_dims(sample_mask, axis=2).repeat(3, axis=2)
    watermark = np.full_like(watermarked, 50, dtype=np.uint8)
    watermarked = np.where(mask_3d > 0.5, watermark, watermarked).astype(np.uint8)
    
    return watermarked, clean


@pytest.fixture
def sample_tensor():
    """Create a sample PyTorch tensor."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def sample_logits():
    """Create sample model logits."""
    # Create logits that after sigmoid will give reasonable probabilities
    logits = torch.randn(1, 1, 64, 64)
    # Make center region have higher values (will be detected as watermark)
    logits[0, 0, 16:48, 16:48] = torch.randn(32, 32) + 2.0
    return logits


@pytest.fixture
def sample_probabilities():
    """Create sample probability map."""
    probs = torch.zeros(1, 1, 64, 64)
    probs[0, 0, 16:48, 16:48] = 0.8  # High probability in center
    return probs


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create temporary directory with sample images."""
    # Create clean and watermarked subdirectories
    clean_dir = tmp_path / "clean"
    watermarked_dir = tmp_path / "watermarked"
    clean_dir.mkdir()
    watermarked_dir.mkdir()
    
    # Create sample images
    for i in range(3):
        # Create clean image
        clean_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        # Create watermarked version
        watermarked_img = clean_img.copy()
        # Add simple watermark
        watermarked_img[30:60, 30:60] = 200
        
        # Save using PIL to ensure valid image files
        from PIL import Image
        Image.fromarray(clean_img).save(clean_dir / f"image_{i:03d}.jpg")
        Image.fromarray(watermarked_img).save(watermarked_dir / f"image_{i:03d}_v1.jpg")
    
    return tmp_path

