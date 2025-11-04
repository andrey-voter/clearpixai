"""Tests for postprocessing utilities."""

import numpy as np
import pytest
import torch

from clearpixai.training.detector.postprocessing import (
    logits_to_probabilities,
    probabilities_to_binary_mask,
    logits_to_binary_mask,
    mask_tensor_to_numpy,
    mask_numpy_to_uint8,
    validate_model_output,
    extract_prediction_for_api,
    batch_predictions_to_list,
)


class TestLogitsToProbabilities:
    """Tests for logits to probabilities conversion."""
    
    def test_converts_logits_to_probabilities(self, sample_logits):
        """Test that logits are converted to probabilities in [0, 1]."""
        probs = logits_to_probabilities(sample_logits)
        
        assert isinstance(probs, torch.Tensor)
        assert probs.shape == sample_logits.shape
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
    
    def test_zero_logits_give_half_probability(self):
        """Test that zero logits give 0.5 probability."""
        logits = torch.zeros(1, 1, 10, 10)
        probs = logits_to_probabilities(logits)
        
        torch.testing.assert_close(probs, torch.full_like(probs, 0.5))
    
    def test_high_positive_logits_give_high_probability(self):
        """Test that high positive logits give high probability."""
        logits = torch.full((1, 1, 10, 10), 5.0)
        probs = logits_to_probabilities(logits)
        
        assert probs.mean() > 0.99
    
    def test_high_negative_logits_give_low_probability(self):
        """Test that high negative logits give low probability."""
        logits = torch.full((1, 1, 10, 10), -5.0)
        probs = logits_to_probabilities(logits)
        
        assert probs.mean() < 0.01
    
    def test_raises_on_non_tensor(self):
        """Test that non-tensor input raises ValueError."""
        with pytest.raises(ValueError, match="Expected torch.Tensor"):
            logits_to_probabilities(np.array([1, 2, 3]))


class TestProbabilitiesToBinaryMask:
    """Tests for probabilities to binary mask conversion."""
    
    def test_converts_to_binary(self, sample_probabilities):
        """Test that probabilities are converted to binary values."""
        mask = probabilities_to_binary_mask(sample_probabilities)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == sample_probabilities.shape
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())
    
    def test_default_threshold(self):
        """Test default threshold of 0.5."""
        probs = torch.tensor([[[[0.3, 0.5, 0.7]]]])
        mask = probabilities_to_binary_mask(probs)
        
        expected = torch.tensor([[[[0.0, 0.0, 1.0]]]])
        torch.testing.assert_close(mask, expected)
    
    def test_custom_threshold(self):
        """Test custom threshold."""
        probs = torch.tensor([[[[0.3, 0.5, 0.7]]]])
        mask = probabilities_to_binary_mask(probs, threshold=0.6)
        
        expected = torch.tensor([[[[0.0, 0.0, 1.0]]]])
        torch.testing.assert_close(mask, expected)
    
    def test_raises_on_invalid_threshold(self, sample_probabilities):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            probabilities_to_binary_mask(sample_probabilities, threshold=1.5)
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            probabilities_to_binary_mask(sample_probabilities, threshold=-0.1)
    
    def test_raises_on_non_tensor(self):
        """Test that non-tensor input raises ValueError."""
        with pytest.raises(ValueError, match="Expected torch.Tensor"):
            probabilities_to_binary_mask(np.array([0.5, 0.6]))


class TestLogitsToBinaryMask:
    """Tests for direct logits to binary mask conversion."""
    
    def test_combines_sigmoid_and_threshold(self, sample_logits):
        """Test that function combines sigmoid and thresholding."""
        mask = logits_to_binary_mask(sample_logits)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == sample_logits.shape
        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())
    
    def test_equivalent_to_two_step(self, sample_logits):
        """Test that result is equivalent to two-step conversion."""
        mask1 = logits_to_binary_mask(sample_logits, threshold=0.5)
        
        probs = logits_to_probabilities(sample_logits)
        mask2 = probabilities_to_binary_mask(probs, threshold=0.5)
        
        torch.testing.assert_close(mask1, mask2)


class TestMaskTensorToNumpy:
    """Tests for mask tensor to numpy conversion."""
    
    def test_converts_4d_tensor(self):
        """Test conversion of 4D tensor (B, C, H, W)."""
        tensor = torch.rand(2, 1, 32, 32)
        numpy_array = mask_tensor_to_numpy(tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.ndim == 2
        assert numpy_array.shape == (32, 32)
    
    def test_converts_3d_tensor(self):
        """Test conversion of 3D tensor (C, H, W)."""
        tensor = torch.rand(1, 32, 32)
        numpy_array = mask_tensor_to_numpy(tensor)
        
        assert numpy_array.shape == (32, 32)
    
    def test_converts_2d_tensor(self):
        """Test conversion of 2D tensor (H, W)."""
        tensor = torch.rand(32, 32)
        numpy_array = mask_tensor_to_numpy(tensor)
        
        assert numpy_array.shape == (32, 32)
    
    def test_clips_values_to_valid_range(self):
        """Test that values are clipped to [0, 1]."""
        tensor = torch.tensor([[[-0.5, 0.5, 1.5]]])
        numpy_array = mask_tensor_to_numpy(tensor)
        
        assert numpy_array.min() >= 0.0
        assert numpy_array.max() <= 1.0
    
    def test_raises_on_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        tensor = torch.rand(2, 3, 4, 5, 6)  # 5D
        
        with pytest.raises(ValueError, match="Expected 2D, 3D, or 4D"):
            mask_tensor_to_numpy(tensor)


class TestMaskNumpyToUint8:
    """Tests for mask numpy to uint8 conversion."""
    
    def test_converts_float_to_uint8(self):
        """Test conversion of float mask to uint8."""
        mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        uint8_mask = mask_numpy_to_uint8(mask)
        
        assert uint8_mask.dtype == np.uint8
        expected = np.array([[0, 127, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(uint8_mask, expected)
    
    def test_preserves_uint8(self):
        """Test that uint8 mask is preserved."""
        mask = np.array([[0, 127, 255]], dtype=np.uint8)
        uint8_mask = mask_numpy_to_uint8(mask)
        
        np.testing.assert_array_equal(uint8_mask, mask)
    
    def test_clips_out_of_range(self):
        """Test that out of range values are clipped."""
        mask = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
        uint8_mask = mask_numpy_to_uint8(mask)
        
        expected = np.array([[0, 127, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(uint8_mask, expected)
    
    def test_raises_on_invalid_type(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Expected numpy array"):
            mask_numpy_to_uint8([0.5, 0.6])


class TestValidateModelOutput:
    """Tests for model output validation."""
    
    def test_validates_correct_output(self, sample_logits):
        """Test that valid output passes validation."""
        validate_model_output(sample_logits)  # Should not raise
    
    def test_validates_with_expected_shape(self):
        """Test validation with expected shape."""
        output = torch.rand(4, 1, 32, 32)
        validate_model_output(output, expected_shape=(4, 1, 32, 32))
    
    def test_validates_with_partial_expected_shape(self):
        """Test validation with partial expected shape."""
        output = torch.rand(4, 1, 32, 32)
        validate_model_output(output, expected_shape=(4, 1, None, None))
    
    def test_raises_on_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        output = torch.rand(2, 3, 4, 5, 6)  # 5D
        
        with pytest.raises(ValueError, match="Expected 2D, 3D, or 4D"):
            validate_model_output(output)
    
    def test_raises_on_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        output = torch.rand(4, 1, 32, 32)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_model_output(output, expected_shape=(4, 1, 64, 64))


class TestExtractPredictionForAPI:
    """Tests for API prediction extraction."""
    
    def test_returns_correct_structure(self, sample_logits):
        """Test that result has correct structure."""
        result = extract_prediction_for_api(sample_logits)
        
        assert isinstance(result, dict)
        assert 'has_watermark' in result
        assert 'confidence' in result
        assert 'max_confidence' in result
        assert 'watermark_ratio' in result
        assert 'threshold' in result
        assert 'probabilities' in result
        assert 'binary_mask' in result
    
    def test_has_watermark_detection(self):
        """Test watermark detection logic."""
        # High probability in large region
        logits_with = torch.full((1, 1, 64, 64), 3.0)
        result_with = extract_prediction_for_api(logits_with)
        assert result_with['has_watermark'] is True
        
        # Low probability everywhere
        logits_without = torch.full((1, 1, 64, 64), -3.0)
        result_without = extract_prediction_for_api(logits_without)
        assert result_without['has_watermark'] is False
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        logits = torch.full((1, 1, 10, 10), 2.0)  # High positive values
        result = extract_prediction_for_api(logits)
        
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['confidence'] > 0.8  # Should be high
    
    def test_numpy_output_format(self, sample_logits):
        """Test numpy output format."""
        result = extract_prediction_for_api(sample_logits, output_format='numpy')
        
        assert isinstance(result['probabilities'], np.ndarray)
        assert isinstance(result['binary_mask'], np.ndarray)
    
    def test_list_output_format(self, sample_logits):
        """Test list output format."""
        result = extract_prediction_for_api(sample_logits, output_format='list')
        
        assert isinstance(result['probabilities'], list)
        assert isinstance(result['binary_mask'], list)
    
    def test_binary_output_format(self, sample_logits):
        """Test binary output format (no arrays)."""
        result = extract_prediction_for_api(sample_logits, output_format='binary')
        
        assert 'probabilities' not in result
        assert 'binary_mask' not in result
        assert 'has_watermark' in result
    
    def test_raises_on_invalid_format(self, sample_logits):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid output_format"):
            extract_prediction_for_api(sample_logits, output_format='invalid')


class TestBatchPredictionsToList:
    """Tests for batch predictions processing."""
    
    def test_processes_batch(self):
        """Test processing of batch predictions."""
        logits = torch.randn(3, 1, 32, 32)
        results = batch_predictions_to_list(logits)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def test_each_result_has_required_fields(self):
        """Test that each result has required fields."""
        logits = torch.randn(2, 1, 32, 32)
        results = batch_predictions_to_list(logits)
        
        for result in results:
            assert 'has_watermark' in result
            assert 'confidence' in result
            assert 'watermark_ratio' in result
    
    def test_raises_on_non_4d_tensor(self):
        """Test that non-4D tensor raises ValueError."""
        logits = torch.randn(3, 32, 32)  # 3D
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            batch_predictions_to_list(logits)
    
    def test_single_batch_item(self):
        """Test processing of single batch item."""
        logits = torch.randn(1, 1, 32, 32)
        results = batch_predictions_to_list(logits)
        
        assert len(results) == 1

