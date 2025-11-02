# Installing GroundingDINO + SAM

GroundingDINO and SAM need to be installed from GitHub (not PyPI).

## Quick Install

```bash
# Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Download Model Weights

Create a `weights/` directory and download models:

```bash
mkdir -p weights

# Download GroundingDINO weights (~700MB)
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download SAM weights (~2.4GB)
wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Full Installation Script

```bash
cd /home/vodjanyjan/ClearPixAi

# Install base dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install GroundingDINO + SAM from GitHub
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download weights
mkdir -p weights
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage After Installation

```bash
# Use GroundingDINO + SAM
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam

# Fallback to EasyOCR if GroundingDINO not installed
uv run python run.py -i image.jpg -o clean.jpg --quality
```

## Why Not PyPI?

GroundingDINO and SAM are research projects that:
- Have frequent updates
- Depend on specific CUDA versions
- Are best installed from source

The tool will automatically fall back to EasyOCR if GroundingDINO/SAM aren't available.

