# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for image processing and PyTorch
# Note: If build fails with 502 Bad Gateway, retry - this is a transient Debian repo issue
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install certifi for better SSL certificate handling
RUN pip install --no-cache-dir certifi

# Copy source code
COPY clearpixai/ ./clearpixai/
COPY src/ ./src/

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Set Hugging Face cache directory and increase timeouts for model downloads
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_DOWNLOAD_RETRY=10
ENV HF_HUB_DOWNLOAD_RETRY_DELAY=5

# Note: SDXL model will be downloaded on first run (cached for subsequent runs)
# Pre-downloading during build is skipped due to network/SSL issues in Docker build environment
# Model download happens automatically when container runs with better retry logic

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]

