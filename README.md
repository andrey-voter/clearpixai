# ClearPixAI

ClearPixAI is a watermark removal tool built as a two-stage pipeline:
- **Detection**: a segmentation model predicts a watermark mask
- **Inpainting**: a diffusion inpainting backend restores the masked area

This repository focuses on the **training pipeline for the watermark detector**.

## Overview

- **Task**: binary segmentation (watermark vs background)
- **Output**: a mask used by the watermark removal pipeline
- **Training**: config-driven (`configs/train_config.yaml`)

## Metrics (targets)

| Metric | Target | Minimum acceptable |
|--------|--------|-------------------|
| **IoU** | ≥ 0.90 | ≥ 0.80 |
| **Dice (F1)** | ≥ 0.90 | ≥ 0.80 |
| **Precision** | ≥ 0.85 | ≥ 0.75 |
| **Recall** | ≥ 0.85 | ≥ 0.75 |

Quality gate (recommended): **IoU ≥ 0.80**.

## Data description

The dataset is made of **paired images** (watermarked + clean). Masks are generated automatically by computing the difference between the pair.

### Directory layout

```text
data/
├── clean/
│   ├── image001.jpg
│   └── ...
└── watermarked/
    ├── image001_v1.jpg
    └── ...
```

Validate your pair dataset before training:

```bash
uv run python clearpixai/training/detector/validate_data.py --data-dir /path/to/data
```

## Run instructions

For a quick walk-through, see `QUICKSTART.md`. Below is the short version.

### Install

```bash
uv sync
```

or:

```bash
pip install -r requirements.txt
```

### Configure training

Edit `configs/train_config.yaml` and set:
- `data.data_dir`: your dataset path
- `data.batch_size`: reduce if you hit OOM

### Train

```bash
uv run python clearpixai/training/detector/train_from_config.py \
  --config configs/train_config.yaml \
  --verbose
```

Optional (monitor training):

```bash
tensorboard --logdir checkpoints
```

### Validate

```bash
uv run python clearpixai/training/detector/validate.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --data-dir /path/to/validation/data \
  --output validation_metrics.json
```

### Export

```bash
uv run python clearpixai/training/detector/export_model.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --output-dir exported_models/my_model \
  --model-name "my-watermark-detector-v1"
```

### Run inference (CLI)

```bash
uv run clearpixai -i input.jpg -o output_clean.jpg
```

Use your exported weights:

```bash
uv run clearpixai -i input.jpg -o output_clean.jpg \
  --segmentation-weights exported_models/my_model/pytorch_model.pth
```

## Testing

Install dev dependencies (includes `pytest`):

```bash
make install-dev
```

Run tests:

```bash
make test
```

More commands and test categories: `TESTING.md`.