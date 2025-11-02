# ClearPixAi

AI-powered watermark removal pipeline inspired by the [ComfyUI workflow](https://comfyui.org/en/ai-powered-watermark-removal-workflow). ClearPixAi now ships as a modular toolkit with interchangeable detectors and inpainting strategies that can be composed from the command line or imported in Python code.

## Highlights

- **Modular architecture** – decoupled detection, mask building, and inpainting modules
- **Multiple detectors** – EasyOCR (text), Florence-2 (vision-language), GroundingDINO + SAM (logos & graphics)
- **Two inpainting modes** – fast OpenCV Telea and quality Stable Diffusion crop→sample→stitch
- **Robust fallbacks** – automatically cascade through detectors when dependencies or weights are missing
- **`uv` native** – reproducible environments via `uv sync` and `uv run`

## Detection Options

| Detector | Best for | Requirements |
|----------|----------|--------------|
| `easyocr` (default) | Text-based watermarks | Works out of the box |
| `florence2` | Mixed text/logo prompts | `transformers`, `timm`, `einops`, Hugging Face token (optional) |
| `grounding_sam` | Logos, signatures, complex graphics | Install GroundingDINO + SAM weights (see `INSTALL_GROUNDING_SAM.md`) |

**Fallbacks** – use `--fallback-detector` to provide a list. ClearPixAi will try the primary detector followed by fallbacks until one produces a mask.

## Quick Start

```bash
# 1. Install core dependencies
uv sync

# 2. (Optional) Install Florence-2 dependencies are already included.

# 3. (Optional) Install GroundingDINO + SAM
uv pip install git+https://github.com/IDEA-Research/GroundingDINO.git \
              git+https://github.com/facebookresearch/segment-anything.git
```

Download weights to `~/.cache/clearpixai/weights` or use `--weights-dir` to override. See `INSTALL_GROUNDING_SAM.md` for details.

## CLI Usage

All commands can be executed with either `uv run clearpixai ...` or `uv run python run.py ...` (the latter keeps backward compatibility).

```bash
# Fast mode (OpenCV + EasyOCR)
uv run clearpixai -i tests/watermark.jpg -o outputs/clean.jpg

# Quality mode with Stable Diffusion
uv run clearpixai -i tests/watermark.jpg -o outputs/clean.jpg --quality

# Florence-2 detector with custom prompt (falls back to EasyOCR)
uv run clearpixai -i tests/watermark.jpg -o outputs/clean.jpg \
  --detector florence2 --fallback-detector easyocr \
  --prompt "watermark; logo; signature"

# GroundingDINO + SAM with Stable Diffusion
uv run clearpixai -i tests/watermark.jpg -o outputs/clean.jpg \
  --quality --detector grounding_sam --prompt "logo. watermark."

# Save generated mask and force CPU execution
uv run clearpixai -i tests/watermark.jpg -o outputs/clean.jpg \
  --save-mask --cpu
```

### Common Flags

- `--mode {fast,quality}` – choose OpenCV or Stable Diffusion inpainting
- `--detector {easyocr,florence2,grounding_sam}` – primary detector
- `--fallback-detector ...` – optional cascade of fallback detectors
- `--prompt` – text prompt for Florence-2 and GroundingDINO
- `--weights-dir` – directory containing model checkpoints (`~/.cache/clearpixai/weights` by default)
- `--hf-token` – Hugging Face token for Florence-2 (can also be provided via env vars)
- `--gpu` / `--cpu` / `--device` – device selection helpers
- `--save-mask` – store the union mask next to the output image
- `--verbose` – enable debug logging

Run `uv run clearpixai --help` for the full reference.

## Workflow Overview

```
Input Image
   │
   ├─ Detection (EasyOCR / Florence-2 / GroundingDINO+SAM)
   │       ↓
   ├─ Mask builder (expand + dilate + Gaussian blur)
   │       ↓
   └─ Inpainting
           ├─ Fast mode → OpenCV Telea
           └─ Quality mode → Stable Diffusion crop → sample → stitch
```

Each stage can be reused independently via the Python API (`clearpixai.detection`, `clearpixai.mask`, `clearpixai.inpaint`).

## Model Assets

| Model | Purpose | Where to place |
|-------|---------|----------------|
| `groundingdino_swint_ogc.pth` | GroundingDINO weights | `~/.cache/clearpixai/weights` (default) |
| `sam_vit_h_4b8939.pth` | Segment Anything | same as above |
| Stable Diffusion 2 Inpainting | Quality mode | Downloaded automatically by `diffusers` |

See `INSTALL_GROUNDING_SAM.md` for helper commands.

## Architecture

- `clearpixai/cli.py` – argument parsing & logging configuration
- `clearpixai/pipeline.py` – orchestration, detector fallback logic, mask + inpainting flow
- `clearpixai/detection/` – detector implementations (EasyOCR, Florence-2, GroundingDINO+SAM)
- `clearpixai/mask.py` – mask aggregation (expansion, dilation, blur)
- `clearpixai/inpaint/` – OpenCV and Stable Diffusion utilities

The `clearpixai` package can be imported directly for scripting:

```python
from pathlib import Path
from clearpixai.pipeline import PipelineConfig, remove_watermark

config = PipelineConfig()
config.mode = "quality"
config.detection.detector = "florence2"
config.detection.fallback_detectors = ("easyocr",)

remove_watermark(Path("photo.jpg"), Path("photo_clean.jpg"), config)
```

## Troubleshooting

- **No detections** – add `--fallback-detector easyocr`, tweak `--prompt`, or lower `--box-threshold`
- **Florence-2 errors** – ensure `timm` and `einops` are installed and pass `--hf-token` if the model requires authentication
- **GroundingDINO errors** – double-check weights and installation steps in `INSTALL_GROUNDING_SAM.md`
- **CUDA OOM** – use `--mode fast` or `--cpu`, or reduce image dimensions before processing

## License

MIT License – free for personal and commercial use.
