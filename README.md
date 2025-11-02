# ClearPixAi

AI-powered watermark removal tool following the [ComfyUI workflow](https://comfyui.org/en/ai-powered-watermark-removal-workflow).

## Features

Implements the complete ComfyUI workflow:

### 🔍 Detection Methods

**1. EasyOCR** ✅ **WORKS OUT OF THE BOX**
- Fast text detection
- Good for text watermarks
- No additional installation needed

**2. GroundingDINO + SAM** ⭐ (Optional - Module 2 from ComfyUI)
- Combines object detection (DINO) + segmentation (SAM)
- Best for logos, images, and complex watermarks
- Requires manual installation (see `INSTALL_GROUNDING_SAM.md`)

### 🎨 Inpainting Workflow

**Quality Mode** - Follows ComfyUI exactly:
1. **InpaintCrop**: Crop watermark region
2. **KSampler**: Apply Stable Diffusion inpainting
3. **InpaintStitch**: Stitch back to original

**Fast Mode** - OpenCV classical inpainting

### 🎭 Mask Processing

- **GrowMaskWithBlur**: Expands mask edges with blur for natural blending

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

**Note**: First run will download models:
- GroundingDINO weights (~700MB)
- SAM weights (~2.4GB)  
- Stable Diffusion (~5GB)

### 2. Usage Examples

**Quality Mode with GroundingDINO + SAM** ⭐ **RECOMMENDED**:
```bash
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam
```

**With custom prompt**:
```bash
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam --prompt "logo. watermark. stamp."
```

**Fast mode with EasyOCR**:
```bash
uv run python run.py -i image.jpg -o clean.jpg
```

**Quality mode with EasyOCR** (text-only):
```bash
uv run python run.py -i image.jpg -o clean.jpg --quality
```

## How It Works

### Complete ComfyUI Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DETECTION                                                    │
│    GroundingDINO → finds objects matching text prompt          │
│    SAM → creates precise segmentation masks                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. MASK PROCESSING                                              │
│    GrowMaskWithBlur → expands + blurs mask edges               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. INPAINTING (Quality Mode)                                    │
│    InpaintCrop → crop watermark region with padding            │
│    KSampler → Stable Diffusion inpainting on crop              │
│    InpaintStitch → stitch back to original image               │
└─────────────────────────────────────────────────────────────────┘
```

## Models Used

| Component | Model | Size | Source |
|-----------|-------|------|--------|
| **Object Detection** | GroundingDINO | ~700MB | GitHub |
| **Segmentation** | SAM (vit_h) | ~2.4GB | Meta AI |
| **Inpainting** | Stable Diffusion 2.0 | ~5GB | HuggingFace |
| **Text Detection** | EasyOCR | ~500MB | PyPI |

## Usage Options

```bash
# Basic usage - GroundingDINO + SAM + Quality mode
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam

# EasyOCR for text-only
uv run python run.py -i image.jpg -o clean.jpg --quality

# Fast mode (OpenCV)
uv run python run.py -i image.jpg -o clean.jpg

# Save detection mask
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam --save-mask

# Custom detection prompt
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam \
  --prompt "company logo. brand mark. signature."

# Specific GPU
uv run python run.py -i image.jpg -o clean.jpg --quality --grounding-sam --gpu 6

# CPU mode
uv run python run.py -i image.jpg -o clean.jpg --cpu
```

## Detection Method Comparison

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **GroundingDINO + SAM** | Logos, images, complex watermarks | Medium | Excellent ⭐ |
| **EasyOCR** | Text-only watermarks | Fast | Good |

## Requirements

- Python 3.10+
- CUDA GPU recommended (12GB+ VRAM for all models)
- CPU mode available but slower

## Performance

| Configuration | Time | Memory |
|---------------|------|--------|
| Quality + GroundingDINO + SAM | ~30-60s | 12GB VRAM |
| Quality + EasyOCR | ~16s | 6GB VRAM |
| Fast + EasyOCR | ~6s | 2GB RAM |

## Based On

This implementation follows the [ComfyUI AI-Powered Watermark Removal Workflow](https://comfyui.org/en/ai-powered-watermark-removal-workflow):

✅ **Module 2**: GroundingDINO + SAM for detection  
✅ **GrowMaskWithBlur**: Mask expansion and blurring  
✅ **InpaintCrop → KSampler → InpaintStitch**: Proper inpainting workflow  
✅ **Stable Diffusion**: High-quality inpainting

## Troubleshooting

**Model Download Issues**:
```bash
# Models will auto-download on first run to:
# - ~/.cache/huggingface/ (Stable Diffusion)
# - weights/ (GroundingDINO, SAM)
```

**CUDA Out of Memory**:
```bash
# Use CPU mode or smaller models
uv run python run.py -i image.jpg -o clean.jpg --cpu
```

**GroundingDINO not working**:
```bash
# Falls back to EasyOCR automatically
# Or use EasyOCR explicitly:
uv run python run.py -i image.jpg -o clean.jpg --quality
```

## License

MIT License - Free for personal and commercial use
