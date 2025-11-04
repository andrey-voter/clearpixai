.PHONY: help test test-fast test-cov lint format install clean

# Default target
help:
	@echo "ClearPixAI - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install dependencies with dev tools"
	@echo "  test         - Run all tests"
	@echo "  test-fast    - Run fast tests only (no GPU, no slow)"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run all linters"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean up generated files"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv pip install -e .
	@echo "Installation complete!"

# Install dev dependencies
install-dev:
	@echo "Installing dev dependencies..."
	uv pip install -e ".[dev]"
	@echo "Dev installation complete!"

# Run all tests
test:
	@echo "Running all tests..."
	pytest tests/ -v

# Run fast tests only
test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v -m "not gpu and not slow"

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=clearpixai --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

# Run specific test file
test-preprocessing:
	@echo "Running preprocessing tests..."
	pytest tests/test_preprocessing.py -v

test-postprocessing:
	@echo "Running postprocessing tests..."
	pytest tests/test_postprocessing.py -v

test-data:
	@echo "Running data validation tests..."
	pytest tests/test_data_validation.py -v

# Run linters
lint:
	@echo "Running linters..."
	@echo ""
	@echo "=== Black (code formatting) ==="
	black --check --diff clearpixai tests
	@echo ""
	@echo "=== isort (import sorting) ==="
	isort --check-only --diff clearpixai tests
	@echo ""
	@echo "=== Flake8 (style guide) ==="
	flake8 clearpixai tests
	@echo ""
	@echo "=== MyPy (type checking) ==="
	mypy clearpixai --ignore-missing-imports
	@echo ""
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	black clearpixai tests
	isort clearpixai tests
	@echo "Formatting complete!"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "Cleanup complete!"

# Pre-commit checks (run before committing)
pre-commit: format lint test-fast
	@echo ""
	@echo "✅ Pre-commit checks passed!"

# CI simulation (run what CI will run)
ci: lint test-cov
	@echo ""
	@echo "✅ CI simulation complete!"

