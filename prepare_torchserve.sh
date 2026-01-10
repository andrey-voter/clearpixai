#!/bin/bash
# Script to prepare TorchServe deployment artifacts

set -e

MODEL_NAME="mymodel"
MODEL_STORE_DIR="model-store"
EXPORTED_MODEL_DIR="exported_models/latest"

# Detect Python command (prefer uv run, fallback to python3/python)
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python or activate virtual environment."
    exit 1
fi

echo "Preparing TorchServe deployment artifacts..."
echo "Using Python: ${PYTHON_CMD}"
echo ""

# Create model store directory
mkdir -p "${MODEL_STORE_DIR}"

# Step 1: Export model to TorchScript
echo "Step 1: Exporting model to TorchScript..."
${PYTHON_CMD} export_torchscript.py \
    --weights "${EXPORTED_MODEL_DIR}/pytorch_model.pth" \
    --config "${EXPORTED_MODEL_DIR}/config.json" \
    --output "${MODEL_STORE_DIR}/model.pt" \
    --image-size 512

# Step 2: Create model archive
echo ""
echo "Step 2: Creating TorchServe archive..."
${PYTHON_CMD} create_torchserve_archive.py \
    --model "${MODEL_STORE_DIR}/model.pt" \
    --handler torchserve_handler.py \
    --config "${EXPORTED_MODEL_DIR}/config.json" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${MODEL_STORE_DIR}"

echo ""
echo "TorchServe artifacts prepared successfully!"
echo ""
echo "Next steps:"
echo "1. Build Docker image: docker build -f Dockerfile.torchserve -t mymodel-serve:v1 ."
echo "2. Run container: docker run -d -p 8080:8080 -p 8081:8081 mymodel-serve:v1"
echo "3. Test service: curl -X POST http://localhost:8080/predictions/${MODEL_NAME} -T sample_input.json"

