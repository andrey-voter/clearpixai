# Inpainting Model Upgrade

## Overview

ClearPixAI has been upgraded with a **modular inpainting system** that supports multiple backends for improved quality and flexibility.

## What Changed?

### 1. New Default: SDXL Inpainting

The default inpainting backend is now **Stable Diffusion XL (SDXL)**, which provides:
- **Higher quality** results compared to SD 2.0
- **Better coherence** between inpainted and original regions
- **More photorealistic** outputs
- **Better handling** of complex textures and patterns

### 2. Modular Architecture

The codebase now has a clean, modular inpainting architecture:
- **Base interface** (`BaseInpainter`) for all inpainting models
- **Easy to extend** with new models (PowerPaint, HD-Painter, etc.)
- **Backward compatible** - old SD 2.0 backend still available

### 3. Backend Options

#### SDXL (Recommended - Default)
```bash
# Uses SDXL by default
clearpixai -i input.jpg -o output.jpg

# Or explicitly specify SDXL
clearpixai -i input.jpg -o output.jpg --diffusion-backend sdxl
```

**Pros:**
- Best quality results
- State-of-the-art photorealism
- Better at preserving image details

**Cons:**
- Requires more GPU memory (~10GB VRAM)
- Slightly slower than SD 2.0

#### Stable Diffusion 2.0 (Legacy)
```bash
clearpixai -i input.jpg -o output.jpg --diffusion-backend sd
```

**Pros:**
- Faster inference
- Lower GPU memory requirement (~4GB VRAM)
- Good for batch processing

**Cons:**
- Lower quality than SDXL
- Less photorealistic results

## Usage Examples

### Basic Usage with SDXL (Default)
```bash
clearpixai -i watermarked.jpg -o clean.jpg
```

### Using Specific GPU
```bash
# Use GPU 2 specifically
clearpixai -i input.jpg -o output.jpg --gpu 2

# Or use CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2 clearpixai -i input.jpg -o output.jpg
```

### Custom Settings
```bash
# SDXL with custom parameters
clearpixai -i input.jpg -o output.jpg \
  --diffusion-backend sdxl \
  --diffusion-steps 50 \
  --diffusion-guidance 8.0

# Legacy SD 2.0 for faster processing
clearpixai -i input.jpg -o output.jpg \
  --diffusion-backend sd \
  --diffusion-steps 30 \
  --diffusion-guidance 5.0
```

### Custom Model
```bash
# Use a custom SDXL model from HuggingFace
clearpixai -i input.jpg -o output.jpg \
  --diffusion-backend sdxl \
  --diffusion-model "stabilityai/stable-diffusion-xl-base-1.0"
```

## Configuration Details

### Default SDXL Settings
```python
backend: "sdxl"
model_id: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
num_inference_steps: 50
guidance_scale: 8.0
strength: 0.999
padding: 32
scheduler: "euler"
guidance_rescale: 0.7
mask_feather: 8
```

### Default SD 2.0 Settings
```python
backend: "sd"
model_id: "stabilityai/stable-diffusion-2-inpainting"
num_inference_steps: 100
guidance_scale: 10.0
strength: 0.999
padding: 16
scheduler: "dpm++"
mask_feather: 0
```

## Architecture

### File Structure
```
clearpixai/inpaint/
├── __init__.py              # Exports all inpainters
├── base.py                  # Base interface for all inpainters
├── stable_diffusion.py      # SD 2.0 implementation
└── sdxl_inpainter.py        # SDXL implementation (new)
```

### Adding New Backends

To add a new inpainting backend (e.g., PowerPaint, HD-Painter):

1. **Create a new inpainter class**:
```python
from .base import BaseInpainter, InpainterSettings
from dataclasses import dataclass

@dataclass
class MyNewInpainterSettings(InpainterSettings):
    # Your custom settings
    model_path: str = "path/to/model"
    # ... other settings

class MyNewInpainter(BaseInpainter):
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # Your inpainting implementation
        pass
    
    def close(self) -> None:
        # Cleanup resources
        pass
```

2. **Export from `__init__.py`**:
```python
from .my_new_inpainter import MyNewInpainter, MyNewInpainterSettings
```

3. **Add to pipeline.py**:
```python
elif backend == "mynew":
    settings = MyNewInpainterSettings(...)
    inpainter = MyNewInpainter(device=device, settings=settings)
```

4. **Add CLI option** in `cli.py`

## Performance Comparison

| Backend | Quality | Speed | VRAM | Use Case |
|---------|---------|-------|------|----------|
| **SDXL** | ★★★★★ | ★★★☆☆ | ~10GB | Production, best quality |
| **SD 2.0** | ★★★☆☆ | ★★★★☆ | ~4GB | Batch processing, prototyping |

## Future Enhancements

The modular architecture makes it easy to add more advanced models:

### Planned Additions
1. **PowerPaint** - Specialized for object removal
2. **HD-Painter** - High-resolution inpainting
3. **RePaint** - DDPM-based inpainting
4. **Custom fine-tuned models** - Domain-specific inpainting

### How to Request a Model

If you want a specific inpainting model added:
1. Open an issue with the model name and repository
2. Provide example use cases
3. If possible, test the model independently first

## Troubleshooting

### Out of Memory Errors
```bash
# Use SD 2.0 instead of SDXL
clearpixai -i input.jpg -o output.jpg --diffusion-backend sd

# Or reduce inference steps
clearpixai -i input.jpg -o output.jpg --diffusion-steps 25
```

### Slow Performance
```bash
# Reduce inference steps
clearpixai -i input.jpg -o output.jpg --diffusion-steps 30

# Use SD 2.0
clearpixai -i input.jpg -o output.jpg --diffusion-backend sd --diffusion-steps 50
```

### Model Download Issues
```bash
# Models are cached in ~/.cache/huggingface/
# First run will download ~6GB for SDXL, ~2GB for SD 2.0
# Ensure stable internet connection for first run
```

## API Usage

### Python API
```python
from pathlib import Path
from clearpixai.pipeline import PipelineConfig, DiffusionConfig, remove_watermark

# Configure with SDXL
config = PipelineConfig(
    diffusion=DiffusionConfig(
        backend="sdxl",
        num_inference_steps=50,
        guidance_scale=8.0,
    )
)

# Run
remove_watermark(
    input_path=Path("input.jpg"),
    output_path=Path("output.jpg"),
    config=config
)
```

### Using Custom GPU
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2

# Then run as normal
```

## Credits

- **SDXL**: Stability AI
- **Diffusers**: Hugging Face
- **Original SD 2.0**: Stability AI

## Questions?

If you have questions about the inpainting upgrade:
1. Check this documentation
2. Review the code in `clearpixai/inpaint/`
3. Open an issue on the repository

