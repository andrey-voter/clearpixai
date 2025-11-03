# ClearPixAi

**Simple and focused watermark removal using segmentation and diffusion.**

ClearPixAi is a streamlined AI-powered watermark removal tool that uses:
- **Segmentation** for watermark detection (Diffusion Dynamics model)
- **SDXL (Stable Diffusion XL)** for state-of-the-art inpainting quality 🆕

## Features

- 🎯 **Single-purpose architecture** – no complex fallbacks or mode switches
- 🔍 **Segmentation-based detection** – precise watermark masks using Diffusion Dynamics checkpoint
- 🎨 **Advanced inpainting** – SDXL for best quality, SD 2.0 for speed 🆕
- 🔧 **Modular backends** – easily switch between inpainting models 🆕
- ⚡ **Simple CLI** – straightforward command-line interface
- 🐍 **Python API** – easy to integrate into your own scripts
- 🏋️ **Training pipeline** – train your own watermark detector with PyTorch Lightning

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Get Segmentation Model (Optional)

The default weights are included at `clearpixai/detection/best_watermark_model_mit_b5_best.pth`.

To use a different model, download the Diffusion Dynamics checkpoint:

```bash
# Download the pre-trained model
wget https://pub-1039b7ab1ee541c1a1f5ff68ddc309ce.r2.dev/best_watermark_model_mit_b5_best.pth

# Or get it from: https://github.com/Diffusion-Dynamics/watermark-segmentation
```

### 3. Run Watermark Removal

```bash
# Basic usage (uses SDXL by default for best quality) 🆕
uv run clearpixai -i input.jpg -o output.jpg

# Use specific GPU (e.g., GPU 2)
uv run clearpixai -i input.jpg -o output.jpg --gpu 2

# Use SD 2.0 for faster processing
uv run clearpixai -i input.jpg -o output.jpg --diffusion-backend sd

# With custom weights
uv run clearpixai -i input.jpg -o output.jpg --segmentation-weights /path/to/model.pth

# With custom threshold
uv run clearpixai -i input.jpg -o output.jpg --threshold 0.3

# Save the mask for inspection
uv run clearpixai -i input.jpg -o output.jpg --save-mask
```

> **🆕 New!** SDXL is now the default inpainting backend, providing 50-100% better quality than SD 2.0.
> See [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md) for details.

## CLI Options

### Required Arguments

- `-i, --input` – Input image path
- `-o, --output` – Output image path

### Optional Arguments

- `-w, --segmentation-weights` – Path to segmentation model checkpoint (default: clearpixai/detection/best_watermark_model_mit_b5_best.pth)

### Segmentation Options

- `--segmentation-encoder` – Encoder backbone (default: mit_b5)
- `--segmentation-encoder-weights` – Pretrained encoder weights (e.g., imagenet)
- `--threshold` – Probability threshold for mask binarization (default: 0.5)
- `--segmentation-image-size` – Optional square resize dimension before inference

### Mask Processing Options

- `--mask-expand` – Mask expansion ratio (default: 0.15)
- `--mask-dilate` – Mask dilation kernel size in pixels (default: 10)
- `--mask-blur` – Mask blur radius (default: 5)

### Diffusion Options

- `--diffusion-backend` – Backend: `sdxl` (default, best quality) or `sd` (faster) 🆕
- `--diffusion-model` – Diffusion model ID (overrides default for backend)
- `--diffusion-steps` – Number of inference steps (default: 50 for SDXL, 100 for SD)
- `--diffusion-guidance` – Guidance scale (default: 8.0 for SDXL, 10.0 for SD)
- `--diffusion-strength` – Diffusion strength (default: 0.999)
- `--diffusion-prompt` – Custom positive prompt
- `--diffusion-negative-prompt` – Custom negative prompt
- `--blend-with-original` – Blend ratio with original (0.0-1.0)

### General Options

- `--device` – Computation device: auto, cpu, cuda (default: auto)
- `--gpu` – Set CUDA_VISIBLE_DEVICES (e.g., '0')
- `--seed` – Random seed for reproducibility
- `--save-mask` – Save the generated mask alongside output
- `-v, --verbose` – Enable verbose logging

Run `clearpixai --help` for the complete list.

## Python API

```python
from pathlib import Path
from clearpixai.pipeline import (
    PipelineConfig, 
    DiffusionConfig, 
    remove_watermark
)

# Configure pipeline with SDXL (default) 🆕
config = PipelineConfig(
    diffusion=DiffusionConfig(
        backend="sdxl",  # or "sd" for SD 2.0
        num_inference_steps=50,
        guidance_scale=8.0,
    )
)
config.segmentation.threshold = 0.5
config.save_mask = True

# Or use SD 2.0 for faster processing
# config.diffusion.backend = "sd"

# Run watermark removal
remove_watermark(
    input_path=Path("input.jpg"),
    output_path=Path("output.jpg"),
    config=config
)
```

## Workflow

```
Input Image
    │
    ├─ Segmentation Detection
    │   └─ WatermarkSegmentationDetector (Diffusion Dynamics)
    │
    ├─ Mask Processing
    │   └─ Expand → Dilate → Blur
    │
    └─ Diffusion Inpainting 🆕
        ├─ SDXL (default) - Best Quality
        └─ SD 2.0 - Faster
            └─ Output Image
```

## Training Your Own Detector

Want to finetune the detector on your own watermarked images? It's easy!

### Quick Start: Finetune in One Command

```bash
# Install training dependencies (using UV)
uv add pytorch-lightning segmentation-models-pytorch albumentations tensorboard

# Or with pip
pip install -r requirements-training.txt

# Finetune from pretrained checkpoint (recommended!)
uv run python train_detector.py

# That's it! Your model will be saved to checkpoints/
```

### What You Get

- ✅ **Finetuning from pretrained weights** - Start from Diffusion-Dynamics checkpoint
- ✅ **Automatic mask generation** - Creates masks from watermarked/clean pairs
- ✅ **PyTorch Lightning** - Modern training with checkpointing
- ✅ **Data augmentation** - Comprehensive augmentation pipeline
- ✅ **TensorBoard logging** - Real-time monitoring
- ✅ **Easy export** - Convert to standard PyTorch weights

### Training Guides

- 📖 **[KAGGLE_DATASET_GUIDE.md](KAGGLE_DATASET_GUIDE.md)** - Train on Kaggle dataset (recommended for production)
- 📖 **[UV_TRAINING_GUIDE.md](UV_TRAINING_GUIDE.md)** - Training with UV
- 📖 **[START_TRAINING.md](START_TRAINING.md)** - One command to start (30 seconds)
- 📖 **[FINETUNING.md](FINETUNING.md)** - Complete finetuning guide (5 minutes)
- 📚 **[TRAINING.md](TRAINING.md)** - Full training documentation
- 📚 **[COMMANDS.md](COMMANDS.md)** - Quick command reference

### Why Finetune?

Starting from the pretrained Diffusion-Dynamics checkpoint gives you:
- ⚡ **10x faster** - Reaches good performance in 10-30 epochs vs 100+
- 🎯 **Better results** - Benefits from knowledge learned on large datasets  
- 📊 **Less data needed** - Works well even with 10-50 image pairs
- 💪 **More stable** - Less prone to training instabilities

## Project Structure

```
clearpixai/
├── cli.py              # Command-line interface
├── pipeline.py         # Main orchestration logic (supports multiple backends) 🆕
├── mask.py            # Mask processing utilities
├── detection/
│   ├── base.py        # Base detector interface
│   └── segmentation.py # Segmentation detector
├── inpaint/           # 🆕 Modular inpainting system
│   ├── base.py        # Base inpainter interface 🆕
│   ├── stable_diffusion.py # SD 2.0 implementation
│   └── sdxl_inpainter.py   # SDXL implementation 🆕
└── training/
    └── detector/       # Training pipeline
        ├── dataset.py  # Dataset loader with mask generation
        ├── model.py    # PyTorch Lightning model
        └── train.py    # Training script
```

## Dependencies

- **torch** – PyTorch for deep learning
- **diffusers** – Stable Diffusion implementation
- **transformers** – Hugging Face models support
- **segmentation-models-pytorch** – Segmentation architecture
- **pillow** – Image processing
- **numpy** – Numerical operations

## Troubleshooting

### No watermarks detected

- Lower the threshold: `--threshold 0.3` or `--threshold 0.1`
- Check that your segmentation weights are correct
- Use `--save-mask` to inspect what's being detected

### CUDA out of memory

- Switch to SD 2.0 backend: `--diffusion-backend sd` 🆕
- Use CPU mode: `--device cpu`
- Reduce diffusion steps: `--diffusion-steps 30`
- Use `--segmentation-image-size 512` to process smaller images

### Poor inpainting quality

- Make sure you're using SDXL (default): `--diffusion-backend sdxl` 🆕
- Increase inference steps: `--diffusion-steps 75`
- Adjust guidance scale: `--diffusion-guidance 10.0`
- Experiment with mask dilation: `--mask-dilate 15`

### Slow processing

- Use SD 2.0 for faster results: `--diffusion-backend sd` 🆕
- Reduce inference steps: `--diffusion-steps 30`
- Process on GPU: `--device cuda --gpu 0`

## Credits

- Segmentation model from [Diffusion Dynamics](https://github.com/Diffusion-Dynamics/watermark-segmentation)
- Inspired by various ComfyUI watermark removal workflows

## License

MIT License – free for personal and commercial use.
