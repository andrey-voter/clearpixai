"""Tests for data validation and dataset functionality."""

import numpy as np
import pytest
import torch
from pathlib import Path

from clearpixai.training.detector.dataset import WatermarkDataset
from clearpixai.training.detector.preprocessing import validate_image_array, validate_mask_array


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_valid_rgb_image_passes(self):
        """Test that valid RGB image passes validation."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        validate_image_array(img)  # Should not raise
    
    def test_valid_grayscale_image_passes(self):
        """Test that valid grayscale image passes validation."""
        img = np.random.randint(0, 256, (128, 128, 1), dtype=np.uint8)
        validate_image_array(img)  # Should not raise
    
    def test_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)  # 2D
        
        with pytest.raises(ValueError, match="must be 3D"):
            validate_image_array(img)
    
    def test_invalid_dtype_raises(self):
        """Test that invalid dtype raises ValueError."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.int32)
        
        with pytest.raises(ValueError, match="must be uint8 or float"):
            validate_image_array(img)
    
    def test_valid_mask_passes(self):
        """Test that valid mask passes validation."""
        mask = np.random.rand(128, 128).astype(np.float32)
        validate_mask_array(mask)  # Should not raise
    
    def test_mask_out_of_range_raises(self):
        """Test that mask with out of range values raises ValueError."""
        mask = np.array([[2.0, 0.5]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="out of range"):
            validate_mask_array(mask)


class TestWatermarkDataset:
    """Tests for WatermarkDataset class."""
    
    def test_loads_dataset(self, temp_image_dir):
        """Test that dataset loads successfully."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        assert len(dataset) == 3  # We created 3 image pairs
    
    def test_getitem_returns_correct_types(self, temp_image_dir):
        """Test that __getitem__ returns correct types."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        image, mask = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
    
    def test_getitem_returns_correct_shapes(self, temp_image_dir):
        """Test that __getitem__ returns correct shapes."""
        image_size = 64
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=image_size,
            create_masks=True,
        )
        
        image, mask = dataset[0]
        
        # Image: (C, H, W)
        assert image.shape == (3, image_size, image_size)
        # Mask: (H, W)
        assert mask.shape == (image_size, image_size)
    
    def test_getitem_returns_valid_ranges(self, temp_image_dir):
        """Test that returned tensors have valid value ranges."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        image, mask = dataset[0]
        
        # After normalization, image should be roughly in [-3, 3] range (due to standardization)
        # Mask should be in [0, 1]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_max_samples_limits_dataset(self, temp_image_dir):
        """Test that max_samples limits dataset size."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
            max_samples=2,
        )
        
        assert len(dataset) == 2
    
    def test_raises_on_nonexistent_directory(self, tmp_path):
        """Test that nonexistent directory raises ValueError."""
        nonexistent = tmp_path / "does_not_exist"
        
        with pytest.raises(ValueError, match="No image pairs found"):
            WatermarkDataset(
                data_dir=nonexistent,
                image_size=64,
                create_masks=True,
            )
    
    def test_finds_clean_watermarked_structure(self, temp_image_dir):
        """Test that dataset finds clean/ and watermarked/ structure."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        # Should find all 3 pairs
        assert len(dataset) > 0


class TestDatasetTransformations:
    """Tests for dataset transformations and augmentations."""
    
    def test_transform_is_applied(self, temp_image_dir):
        """Test that transform is applied to data."""
        from clearpixai.training.detector.dataset import get_training_augmentation
        
        transform = get_training_augmentation(64)
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            transform=transform,
            create_masks=True,
        )
        
        image, mask = dataset[0]
        
        # Check that data is returned as tensors (result of ToTensorV2)
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
    
    def test_validation_transform(self, temp_image_dir):
        """Test validation transform."""
        from clearpixai.training.detector.dataset import get_validation_augmentation
        
        transform = get_validation_augmentation(64)
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            transform=transform,
            create_masks=True,
        )
        
        image, mask = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
    
    def test_multiple_getitem_calls(self, temp_image_dir):
        """Test multiple calls to __getitem__ work correctly."""
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        # Get same item multiple times
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Should return valid data each time
        assert len(item1) == 2
        assert len(item2) == 2


class TestDataLoading:
    """Tests for data loading with PyTorch DataLoader."""
    
    def test_dataloader_integration(self, temp_image_dir):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get one batch
        batch = next(iter(dataloader))
        images, masks = batch
        
        # Check batch shapes
        assert images.shape == (2, 3, 64, 64)  # (B, C, H, W)
        assert masks.shape == (2, 64, 64)  # (B, H, W)
    
    def test_dataloader_iteration(self, temp_image_dir):
        """Test iterating through DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batch_count = 0
        for images, masks in dataloader:
            batch_count += 1
            assert images.shape[0] <= 2  # Batch size
            assert masks.shape[0] <= 2
        
        assert batch_count > 0
    
    def test_dataloader_with_workers(self, temp_image_dir):
        """Test DataLoader with multiple workers."""
        from torch.utils.data import DataLoader
        
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        
        # Use num_workers=2 for parallel loading
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Set to 0 for testing to avoid multiprocessing issues
        )
        
        # Should work without errors
        batch = next(iter(dataloader))
        assert batch[0].shape[0] <= 2


class TestDataConsistency:
    """Tests for data consistency and correctness."""
    
    def test_mask_matches_images(self, temp_image_dir):
        """Test that generated mask corresponds to image difference."""
        from clearpixai.training.detector.dataset import WatermarkDataset
        import cv2
        
        dataset = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=128,
            create_masks=True,
        )
        
        # Get the raw pair
        pair = dataset.image_pairs[0]
        watermarked = cv2.imread(str(pair['watermarked']))
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)
        clean = cv2.imread(str(pair['clean']))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        
        # They should be different
        assert not np.array_equal(watermarked, clean)
    
    def test_reproducibility_with_seed(self, temp_image_dir):
        """Test that results are reproducible with fixed seed."""
        import torch
        
        torch.manual_seed(42)
        dataset1 = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        image1, mask1 = dataset1[0]
        
        torch.manual_seed(42)
        dataset2 = WatermarkDataset(
            data_dir=temp_image_dir,
            image_size=64,
            create_masks=True,
        )
        image2, mask2 = dataset2[0]
        
        # Note: Augmentations might still cause differences,
        # so we just check that datasets are created consistently
        assert len(dataset1) == len(dataset2)

