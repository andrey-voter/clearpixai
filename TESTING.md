## Testing

This document describes how to run and extend the ClearPixAI test suite and code-quality checks. It is intended to be a practical reference for local development and CI.

### Table of contents

- [Overview](#overview)
- [Test suite layout](#test-suite-layout)
- [Install test dependencies](#install-test-dependencies)
- [Run tests (recommended)](#run-tests-recommended)
- [Run tests with pytest](#run-tests-with-pytest)
- [Markers and selecting tests](#markers-and-selecting-tests)
- [Coverage](#coverage)
- [Code quality checks](#code-quality-checks)
- [CI/CD](#cicd)
- [Troubleshooting](#troubleshooting)
- [Writing new tests](#writing-new-tests)

### Overview

The repository includes automated tests for:

- **Preprocessing**: image pair handling, mask creation, resizing/normalization utilities
- **Postprocessing**: model outputs to API-friendly formats (sigmoid/thresholding, tensor/NumPy conversion)
- **Data validation / dataset**: dataset discovery, pairing logic, shapes/dtypes/ranges, DataLoader integration

Some tests may be marked as **slow** or **gpu** to allow fast development cycles without requiring special hardware.

### Test suite layout

```text
tests/
├── __init__.py
├── conftest.py                 # pytest fixtures and configuration
├── test_preprocessing.py       # preprocessing function tests
├── test_postprocessing.py      # output/prediction processing tests
└── test_data_validation.py     # dataset and data validation tests
```

### Install test dependencies

The easiest way to install development dependencies is via the Makefile target:

```bash
make install-dev
```

If you prefer a plain pip workflow, ensure your environment has the repository installed and test dependencies available (see `requirements.txt` / `pyproject.toml` as applicable).

### Run tests (recommended)

Use the Makefile targets for consistent local and CI behavior:

```bash
# Run all tests
make test

# Run fast tests only (skip GPU and slow tests)
make test-fast

# Run tests with coverage reporting
make test-cov

# Run focused suites
make test-preprocessing
make test-postprocessing
make test-data
```

### Run tests with pytest

You can run `pytest` directly when you need finer control:

```bash
# Run all tests (verbose)
pytest tests/ -v

# Run one file
pytest tests/test_preprocessing.py -v

# Run one class
pytest tests/test_preprocessing.py::TestCreateMaskFromDifference -v

# Run one test
pytest tests/test_preprocessing.py::TestCreateMaskFromDifference::test_creates_valid_mask -v

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### Markers and selecting tests

Tests are categorized using markers. Common selections:

```bash
# Only unit tests (if present)
pytest tests/ -m unit

# Only preprocessing tests
pytest tests/ -m preprocessing

# Only postprocessing tests
pytest tests/ -m postprocessing

# Skip slow tests
pytest tests/ -m "not slow"

# Skip GPU tests
pytest tests/ -m "not gpu"

# Fast development run (skip both)
pytest tests/ -m "not gpu and not slow" -v
```

### Coverage

Generate coverage reports:

```bash
# Terminal report with missing lines
pytest tests/ --cov=clearpixai --cov-report=term-missing

# HTML report (writes to htmlcov/)
pytest tests/ --cov=clearpixai --cov-report=html

# XML report (useful for CI tooling)
pytest tests/ --cov=clearpixai --cov-report=xml
```

Open the HTML report locally:

```bash
xdg-open htmlcov/index.html
```

Suggested coverage targets:

| Component | Target coverage |
|----------:|:----------------|
| Preprocessing | ≥ 90% |
| Postprocessing | ≥ 90% |
| Dataset | ≥ 80% |
| Model | ≥ 70% |
| Overall | ≥ 80% |

### Code quality checks

The repository uses standard Python tooling for formatting, import sorting, linting, and type checking.

```bash
# Format code
make format

# Run linters/type checks
make lint

# Run format + lint + tests (pre-commit style)
make pre-commit
```

Tool-specific commands (optional):

```bash
# Formatting
black --check clearpixai tests
black clearpixai tests

# Import sorting
isort --check-only clearpixai tests
isort clearpixai tests

# Linting
flake8 clearpixai tests

# Type checking
mypy clearpixai --ignore-missing-imports
```

Configuration is maintained in `pyproject.toml` (Black/isort/MyPy) and `.flake8` (Flake8).

### CI/CD

Automated checks run on every push and pull request via GitHub Actions.

- **Workflow file**: `.github/workflows/ci.yml`
- **Typical checks**:
  - formatting/import ordering (Black, isort)
  - style/lint rules (Flake8)
  - type checking (MyPy)
  - unit tests + coverage (pytest)
  - Python compatibility (commonly 3.10 and 3.11)

Minimal workflow snippet (illustrative):

```yaml
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

### Troubleshooting

Common issues and fixes:

- **Import errors**: install the project in editable mode.

```bash
pip install -e .
```

- **Missing dependencies**: install project and dev/test dependencies.

```bash
pip install -r requirements.txt
```

- **Slow test runs**: skip slow and GPU tests locally.

```bash
pytest tests/ -m "not slow and not gpu" -v
```

Debugging helpers:

```bash
# Show local variables on failure
pytest tests/ --showlocals

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Extra verbosity
pytest tests/ -vv
```

### Writing new tests

General guidance:

- **Keep tests deterministic**: avoid network calls and non-seeded randomness.
- **Prefer fixtures**: define reusable inputs in `tests/conftest.py`.
- **Use Arrange–Act–Assert**: keep each test focused on one behavior.
- **Use markers**: tag expensive tests as `slow` and hardware-dependent tests as `gpu`.

Template:

```python
"""Tests for new_module."""

import pytest


class TestFunctionName:
    def test_basic_functionality(self):
        result = function_to_test(input_data)
        assert result is not None

    def test_raises_on_invalid_input(self):
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

