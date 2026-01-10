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

## Data Versioning with DVC

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage large datasets and model files outside of Git.

### Physical Storage

- **Datasets**: Stored in DVC cache (`.dvc/cache/`) and remote storage
  - Training data: `clearpixai/training/detector/data/train/`
  - Validation data: `clearpixai/training/detector/data/val/`
- **Models**: Stored in DVC cache and remote storage
  - Exported models: `exported_models/`

### Getting Started with DVC

After cloning the repository:

```bash
cd ClearPixAi
dvc pull
dvc repro
```

This will:
1. Download all datasets and models from remote storage
2. Run the complete pipeline (prepare → train → evaluate)

### DVC Pipeline

The project includes a DVC pipeline with three stages:

1. **prepare**: Validates and prepares the dataset
2. **train**: Trains the watermark detection model
3. **evaluate**: Validates the trained model and exports it

Run individual stages:

```bash
dvc repro prepare    # Prepare data only
dvc repro train      # Train model (requires prepare)
dvc repro evaluate   # Evaluate model (requires train)
```

### Experiment Plan

The DVC pipeline allows you to:
- Track different versions of datasets and models
- Reproduce any experiment with `dvc repro`
- Compare metrics across experiments using `dvc metrics show`
- Switch between dataset/model versions using Git tags and `dvc checkout`

## Experiment Tracking with MLflow

This project uses [MLflow](https://mlflow.org/) to track training experiments, parameters, metrics, and artifacts.

### MLflow Setup

MLflow is automatically configured to use **local storage** (`./mlruns/` directory). Each training run creates a new experiment run with:

- **Parameters**: All training configuration (learning rate, batch size, encoder name, etc.)
- **Metrics**: Training and validation metrics (loss, IoU) logged during training
- **Artifacts**:
  - Training configuration file (`configs/train_config.yaml`)
  - DVC lock file (`dvc.lock`) for reproducibility
  - Best model checkpoint
  - TensorBoard logs
  - Saved training config copy

### Viewing Results

After training, you can view experiment results using the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open your browser to `http://localhost:5000` to:
- Compare different runs
- View parameters and metrics
- Download artifacts
- Search and filter experiments

### MLflow Storage Location

All MLflow data is stored locally in:
- **Tracking URI**: `file:./mlruns`
- **Experiment name**: `clearpixai_watermark_detector` (or as specified in config)

Each run is automatically tracked when you execute:

```bash
uv run python clearpixai/training/detector/train_from_config.py \
  --config configs/train_config.yaml
```

The MLflow run ID is displayed in the training logs, and you can use it to reference specific runs.

### What Gets Logged

- **Hyperparameters**: All training parameters from config file
- **Metrics**: `train_loss`, `val_loss`, `train_iou`, `val_iou`, `best_val_iou`, `total_epochs`
- **Artifacts**: Config files, checkpoints, TensorBoard logs, DVC lock file
- **Model**: PyTorch Lightning model (via autolog)

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

**Note**: If using DVC, also install DVC:

```bash
pip install dvc
# or
uv add dvc
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

View MLflow experiments:

```bash
mlflow ui --backend-store-uri file:./mlruns
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