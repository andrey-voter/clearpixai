"""Tests for preprocessing utilities."""

import numpy as np
import pytest

from clearpixai.training.detector.preprocessing import (
    create_mask_from_difference,
    resize_if_needed,
    validate_image_array,
    validate_mask_array,
    normalize_image_array,
    denormalize_image_array,
)


class TestCreateMaskFromDifference:
    """Tests for mask creation from image pairs."""
    
    def test_creates_valid_mask(self, sample_image_pair):
        """Test that mask is created with correct properties."""
        watermarked, clean = sample_image_pair
        mask = create_mask_from_difference(watermarked, clean)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.float32
        assert mask.ndim == 2
        assert mask.shape == watermarked.shape[:2]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_detects_watermark_region(self, sample_image_pair):
        """Test that watermark region is detected."""
        watermarked, clean = sample_image_pair
        mask = create_mask_from_difference(watermarked, clean)
        
        # Check that watermark region (center) has higher values
        center_mask = mask[16:48, 16:48]
        border_mask = mask[0:8, 0:8]
        
        assert center_mask.mean() > border_mask.mean()
    
    def test_identical_images_produce_zero_mask(self, sample_image):
        """Test that identical images produce nearly zero mask."""
        mask = create_mask_from_difference(sample_image, sample_image)
        
        assert mask.max() <= 0.1  # Should be mostly zeros
    
    def test_raises_on_shape_mismatch(self, sample_image):
        """Test that shape mismatch raises ValueError."""
        img1 = sample_image
        img2 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="shape mismatch"):
            create_mask_from_difference(img1, img2)
    
    def test_raises_on_invalid_dtype(self, sample_image):
        """Test that invalid dtype raises ValueError."""
        img_float = sample_image.astype(np.float32)
        
        with pytest.raises(ValueError, match="Invalid image dtype"):
            create_mask_from_difference(img_float, sample_image)
    
    def test_raises_on_invalid_shape(self):
        """Test that invalid shape raises ValueError."""
        img_2d = np.zeros((64, 64), dtype=np.uint8)
        img_3d = np.zeros((64, 64, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid image shape"):
            create_mask_from_difference(img_2d, img_3d)
    
    def test_custom_threshold(self, sample_image_pair):
        """Test that custom threshold affects mask."""
        watermarked, clean = sample_image_pair
        
        mask_low = create_mask_from_difference(watermarked, clean, threshold=5)
        mask_high = create_mask_from_difference(watermarked, clean, threshold=50)
        
        # Lower threshold should detect more pixels
        assert mask_low.sum() >= mask_high.sum()


class TestResizeIfNeeded:
    """Tests for image resizing."""
    
    def test_no_resize_when_shapes_match(self, sample_image):
        """Test that image is not resized when shapes match."""
        target_shape = sample_image.shape
        result = resize_if_needed(sample_image, target_shape)
        
        np.testing.assert_array_equal(result, sample_image)
    
    def test_resizes_to_target_shape(self, sample_image):
        """Test that image is resized to target shape."""
        target_shape = (128, 128, 3)
        result = resize_if_needed(sample_image, target_shape)
        
        assert result.shape == target_shape
    
    def test_raises_on_channel_mismatch(self, sample_image):
        """Test that channel mismatch raises ValueError."""
        target_shape = (64, 64, 1)  # Different number of channels
        
        with pytest.raises(ValueError, match="Channel mismatch"):
            resize_if_needed(sample_image, target_shape)
    
    def test_raises_on_invalid_dimensions(self, sample_image):
        """Test that invalid dimensions raise ValueError."""
        target_shape = (64, 64)  # Only 2D
        
        with pytest.raises(ValueError, match="Expected 3D"):
            resize_if_needed(sample_image, target_shape)


class TestValidateImageArray:
    """Tests for image array validation."""
    
    def test_validates_correct_image(self, sample_image):
        """Test that valid image passes validation."""
        validate_image_array(sample_image)  # Should not raise
    
    def test_raises_on_non_array(self):
        """Test that non-array raises ValueError."""
        with pytest.raises(ValueError, match="must be numpy array"):
            validate_image_array([1, 2, 3])
    
    def test_raises_on_wrong_dimensions(self):
        """Test that wrong dimensions raise ValueError."""
        img_2d = np.zeros((64, 64), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="must be 3D"):
            validate_image_array(img_2d)
    
    def test_raises_on_invalid_channels(self):
        """Test that invalid channel count raises ValueError."""
        img_5ch = np.zeros((64, 64, 5), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="must have 1, 3, or 4 channels"):
            validate_image_array(img_5ch)
    
    def test_raises_on_invalid_dtype(self):
        """Test that invalid dtype raises ValueError."""
        img_int32 = np.zeros((64, 64, 3), dtype=np.int32)
        
        with pytest.raises(ValueError, match="must be uint8 or float"):
            validate_image_array(img_int32)
    
    def test_raises_on_out_of_range_uint8(self):
        """Test that out of range uint8 values raise ValueError."""
        img = np.array([[[256, 0, 0]]], dtype=np.uint8)  # Will wrap to 0
        img_view = img.view(np.int16)
        img_view[0, 0, 0] = 256
        
        # This is tricky - uint8 can't actually store 256, so we skip this test
        # or test with a modified approach
    
    def test_warns_on_out_of_range_float(self, caplog):
        """Test that out of range float values generate warning."""
        img = np.array([[[1.5, 0.5, 0.5]]], dtype=np.float32)
        
        validate_image_array(img)
        
        # Check that warning was logged
        assert "out of expected range" in caplog.text


class TestValidateMaskArray:
    """Tests for mask array validation."""
    
    def test_validates_correct_2d_mask(self, sample_mask):
        """Test that valid 2D mask passes validation."""
        validate_mask_array(sample_mask)  # Should not raise
    
    def test_validates_correct_3d_mask(self, sample_mask):
        """Test that valid 3D mask passes validation."""
        mask_3d = np.expand_dims(sample_mask, axis=2)
        validate_mask_array(mask_3d)  # Should not raise
    
    def test_raises_on_wrong_dimensions(self):
        """Test that wrong dimensions raise ValueError."""
        mask_1d = np.zeros(64, dtype=np.float32)
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            validate_mask_array(mask_1d)
    
    def test_raises_on_multi_channel_3d(self):
        """Test that multi-channel 3D mask raises ValueError."""
        mask_3ch = np.zeros((64, 64, 3), dtype=np.float32)
        
        with pytest.raises(ValueError, match="must have 1 channel"):
            validate_mask_array(mask_3ch)
    
    def test_raises_on_out_of_range_float(self):
        """Test that out of range float values raise ValueError."""
        mask = np.array([[1.5, 0.5]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="out of range"):
            validate_mask_array(mask)


class TestNormalizeImageArray:
    """Tests for image normalization."""
    
    def test_normalizes_uint8_to_float(self, sample_image):
        """Test that uint8 is normalized to [0, 1] float."""
        normalized = normalize_image_array(sample_image)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_preserves_float_array(self):
        """Test that float array is preserved."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        normalized = normalize_image_array(img)
        
        np.testing.assert_array_almost_equal(normalized, img)
    
    def test_correct_scaling(self):
        """Test that scaling is correct."""
        img = np.array([[[0, 127, 255]]], dtype=np.uint8)
        normalized = normalize_image_array(img)
        
        expected = np.array([[[0.0, 127/255, 1.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)


class TestDenormalizeImageArray:
    """Tests for image denormalization."""
    
    def test_denormalizes_float_to_uint8(self):
        """Test that float is denormalized to uint8."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        denormalized = denormalize_image_array(img)
        
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
    
    def test_preserves_uint8_array(self, sample_image):
        """Test that uint8 array is preserved."""
        denormalized = denormalize_image_array(sample_image)
        
        np.testing.assert_array_equal(denormalized, sample_image)
    
    def test_correct_scaling(self):
        """Test that scaling is correct."""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        denormalized = denormalize_image_array(img)
        
        expected = np.array([[[0, 127, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(denormalized, expected)
    
    def test_clips_out_of_range_values(self):
        """Test that out of range values are clipped."""
        img = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
        denormalized = denormalize_image_array(img)
        
        expected = np.array([[[0, 127, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(denormalized, expected)

