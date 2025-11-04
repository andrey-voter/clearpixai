# Testing Guide for ClearPixAI

Comprehensive testing documentation for MLOps Assignment 3.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [CI/CD Integration](#cicd-integration)
6. [Code Quality Tools](#code-quality-tools)

---

## 🎯 Overview

ClearPixAI implements comprehensive testing following MLOps best practices:

✅ **Unit Tests**: Test individual functions and components  
✅ **Data Validation Tests**: Verify data format, types, and ranges  
✅ **Preprocessing Tests**: Test data transformations and augmentations  
✅ **Postprocessing Tests**: Verify model output processing for API  
✅ **Integration Tests**: Test complete pipelines  
✅ **CI/CD**: Automatic testing on every commit via GitHub Actions  
✅ **Code Quality**: Linters, formatters, and type checkers

### Test Philosophy

Tests focus on **correctness of preprocessing and pipeline**, not model quality:
- ✅ Data format validation
- ✅ Value range checks
- ✅ Transformation correctness
- ✅ API response format
- ❌ Model accuracy (not tested in unit tests)

---

## 📁 Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures and configuration
├── test_preprocessing.py          # Preprocessing function tests
├── test_postprocessing.py         # Model output processing tests
└── test_data_validation.py        # Data validation and dataset tests
```

### Testable Modules

Code has been refactored into testable modules:

```
clearpixai/training/detector/
├── preprocessing.py               # Testable preprocessing functions
│   ├── create_mask_from_difference()
│   ├── validate_image_array()
│   ├── validate_mask_array()
│   ├── normalize_image_array()
│   └── denormalize_image_array()
│
├── postprocessing.py              # Testable postprocessing functions
│   ├── logits_to_probabilities()
│   ├── probabilities_to_binary_mask()
│   ├── extract_prediction_for_api()
│   └── batch_predictions_to_list()
│
├── dataset.py                     # Dataset with validation
├── model.py                       # PyTorch Lightning model
└── train_from_config.py           # Training script
```

---

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
make test

# Run fast tests only (no GPU, no slow tests)
make test-fast

# Run with coverage report
make test-cov

# Run specific test category
make test-preprocessing
make test-postprocessing
make test-data
```

### Using pytest directly

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run specific test class
pytest tests/test_preprocessing.py::TestCreateMaskFromDifference -v

# Run specific test function
pytest tests/test_preprocessing.py::TestCreateMaskFromDifference::test_creates_valid_mask -v

# Run with coverage
pytest tests/ --cov=clearpixai --cov-report=html

# Run fast tests only (exclude slow and GPU tests)
pytest tests/ -v -m "not gpu and not slow"

# Run tests in parallel (faster)
pytest tests/ -n auto
```

### Test Markers

Tests are categorized using markers:

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only preprocessing tests
pytest tests/ -m preprocessing

# Run only postprocessing tests
pytest tests/ -m postprocessing

# Skip slow tests
pytest tests/ -m "not slow"

# Skip GPU tests
pytest tests/ -m "not gpu"
```

---

## 🧪 Test Categories

### 1. Preprocessing Tests (`test_preprocessing.py`)

**Purpose**: Verify data preprocessing functions work correctly.

**What's Tested**:
- ✅ Mask creation from image pairs
- ✅ Image resizing and dimension handling
- ✅ Image/mask array validation
- ✅ Normalization and denormalization
- ✅ Type and range checks

**Example Test**:
```python
def test_creates_valid_mask(sample_image_pair):
    """Test that mask is created with correct properties."""
    watermarked, clean = sample_image_pair
    mask = create_mask_from_difference(watermarked, clean)
    
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.float32
    assert mask.shape == watermarked.shape[:2]
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
```

**Run**:
```bash
pytest tests/test_preprocessing.py -v
# Or
make test-preprocessing
```

### 2. Postprocessing Tests (`test_postprocessing.py`)

**Purpose**: Verify model output processing for API responses.

**What's Tested**:
- ✅ Logits → Probabilities conversion (sigmoid)
- ✅ Probabilities → Binary mask (thresholding)
- ✅ Tensor → NumPy conversion
- ✅ API response format
- ✅ Batch processing
- ✅ Value range validation

**Example Test**:
```python
def test_extract_prediction_for_api(sample_logits):
    """Test API prediction extraction."""
    result = extract_prediction_for_api(sample_logits)
    
    assert 'has_watermark' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert isinstance(result['has_watermark'], bool)
    assert 0.0 <= result['confidence'] <= 1.0
```

**Run**:
```bash
pytest tests/test_postprocessing.py -v
# Or
make test-postprocessing
```

### 3. Data Validation Tests (`test_data_validation.py`)

**Purpose**: Verify data loading, validation, and dataset functionality.

**What's Tested**:
- ✅ Dataset loading from directories
- ✅ Image pair detection
- ✅ Data format validation
- ✅ Type and range checks
- ✅ DataLoader integration
- ✅ Transformation application

**Example Test**:
```python
def test_getitem_returns_correct_shapes(temp_image_dir):
    """Test that dataset returns correct tensor shapes."""
    dataset = WatermarkDataset(
        data_dir=temp_image_dir,
        image_size=64,
        create_masks=True,
    )
    
    image, mask = dataset[0]
    
    assert image.shape == (3, 64, 64)  # (C, H, W)
    assert mask.shape == (64, 64)       # (H, W)
```

**Run**:
```bash
pytest tests/test_data_validation.py -v
# Or
make test-data
```

---

## 🔄 CI/CD Integration

### GitHub Actions Workflow

Automatic testing runs on every commit and pull request:

**File**: `.github/workflows/ci.yml`

**Jobs**:
1. **Lint**: Code formatting and style checks
2. **Test**: Run unit tests on multiple Python versions
3. **Data Validation**: Test data processing
4. **Postprocessing**: Test model output processing
5. **Test Summary**: Aggregate results

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**What Gets Checked**:
- ✅ Code formatting (Black, isort)
- ✅ Style guide compliance (Flake8)
- ✅ Type checking (MyPy)
- ✅ All unit tests
- ✅ Test coverage
- ✅ Python 3.10 and 3.11 compatibility

### Viewing CI Results

1. Go to GitHub repository
2. Click "Actions" tab
3. View latest workflow run
4. Check individual job results
5. Download coverage reports (artifacts)

### CI Configuration

```yaml
# .github/workflows/ci.yml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=clearpixai
```

---

## 🛠️ Code Quality Tools

### 1. Black (Code Formatter)

**Purpose**: Automatic code formatting.

```bash
# Check formatting
black --check clearpixai tests

# Format code
black clearpixai tests

# Or use Makefile
make format
```

**Configuration**: `pyproject.toml`
```toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311']
```

### 2. isort (Import Sorter)

**Purpose**: Organize imports consistently.

```bash
# Check import sorting
isort --check-only clearpixai tests

# Sort imports
isort clearpixai tests
```

**Configuration**: `pyproject.toml`
```toml
[tool.isort]
profile = "black"
line_length = 100
```

### 3. Flake8 (Linter)

**Purpose**: Style guide enforcement (PEP 8).

```bash
# Run Flake8
flake8 clearpixai tests

# With max line length
flake8 clearpixai tests --max-line-length=100
```

**Configuration**: `.flake8`
```ini
[flake8]
max-line-length = 100
extend-ignore = E203,W503
```

### 4. MyPy (Type Checker)

**Purpose**: Static type checking.

```bash
# Run MyPy
mypy clearpixai --ignore-missing-imports
```

**Configuration**: `pyproject.toml`
```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

### Run All Quality Checks

```bash
# Run all linters
make lint

# Format and lint
make format
make lint

# Pre-commit checks (format + lint + test)
make pre-commit
```

---

## 📊 Coverage Reports

### Generate Coverage Report

```bash
# HTML report
pytest tests/ --cov=clearpixai --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal report
pytest tests/ --cov=clearpixai --cov-report=term-missing

# XML report (for CI)
pytest tests/ --cov=clearpixai --cov-report=xml
```

### Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Preprocessing | ≥ 90% |
| Postprocessing | ≥ 90% |
| Dataset | ≥ 80% |
| Model | ≥ 70% |
| Overall | ≥ 80% |

---

## 🔧 Writing New Tests

### Test Template

```python
"""Tests for new_module."""

import pytest
from clearpixai.module import function_to_test


class TestFunctionName:
    """Tests for function_to_test."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        result = function_to_test(input_data)
        
        assert result is not None
        assert isinstance(result, ExpectedType)
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test(edge_case_input)
        
        assert result == expected_edge_case_output
    
    def test_raises_on_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="error message"):
            function_to_test(invalid_input)
```

### Best Practices

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Use descriptive test names (`test_creates_valid_mask`)
3. **Arrange-Act-Assert**: Structure tests in three parts
4. **Use Fixtures**: Reuse test data with pytest fixtures
5. **Test Edge Cases**: Include boundary conditions
6. **Test Errors**: Verify error handling with `pytest.raises`
7. **Fast Tests**: Keep tests fast (< 1s each)

---

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Install in development mode
pip install -e .
```

**2. Missing Dependencies**
```bash
# Solution: Install test dependencies
pip install -r requirements.txt
```

**3. Slow Tests**
```bash
# Solution: Run fast tests only
pytest tests/ -m "not slow"
```

**4. GPU Tests Fail on CI**
```bash
# Solution: Skip GPU tests in CI
pytest tests/ -m "not gpu"
```

### Debug Test Failures

```bash
# Show local variables on failure
pytest tests/ --showlocals

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Increase verbosity
pytest tests/ -vv
```

---

## 📝 Summary Checklist

### For MLOps Assignment 3:

- ✅ Code refactored into testable modules
- ✅ Preprocessing tests (format, types, ranges)
- ✅ Postprocessing tests (API response format)
- ✅ Data validation tests (required fields, structure)
- ✅ Prediction processing tests (logits → API format)
- ✅ CI/CD with GitHub Actions
- ✅ Automatic test execution on commits
- ✅ Linters and formatters (Black, Flake8, isort, MyPy)
- ✅ Test coverage reporting
- ✅ Documentation

### Quick Commands

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linters
make lint

# Format code
make format

# Pre-commit checks
make pre-commit

# CI simulation
make ci
```

---

**Last Updated**: 2025-11-04  
**Version**: 1.0.0  
**Assignment**: MLOps Task 3 - Testing and CI/CD

