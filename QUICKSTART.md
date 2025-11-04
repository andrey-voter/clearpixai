# ClearPixAI - Quick Start Guide

This guide will get you from zero to trained model in under 10 minutes.

## 🚀 Prerequisites

- Python 3.10+
- GPU with 8GB+ VRAM (recommended) or CPU
- `uv` package manager installed (or use `pip`)

## 📦 Step 1: Install Dependencies (1 minute)

```bash
# Clone repository (if not already done)
git clone <your-repo-url>
cd clearpixai

# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

## 📊 Step 2: Prepare Your Dataset (2 minutes)

### Option A: Use Existing Dataset

If you already have a dataset, organize it as:

```
data/
├── clean/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── watermarked/
    ├── image001_v1.jpg
    ├── image001_v2.jpg
    └── ...
```

Or flat structure:
```
data/
├── image001.jpg           # Watermarked
├── image001_clean.jpg     # Clean
└── ...
```

### Option B: Download Kaggle Dataset

See https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset/data

## ✅ Step 3: Validate Dataset (1 minute)

```bash
# Validate your dataset
uv run python clearpixai/training/detector/validate_data.py \
    --data-dir /path/to/your/data

# Expected output:
# ✅ Dataset validation PASSED
# Dataset is ready for training with N image pairs
```

## ⚙️ Step 4: Configure Training (1 minute)

Edit `configs/train_config.yaml`:

```yaml
# Update these key settings
random_seed: 42

data:
  data_dir: "/path/to/your/data"  # ← UPDATE THIS
  batch_size: 8                    # Reduce to 4 if OOM
  image_size: 512

training:
  learning_rate: 0.0001
  max_epochs: 50                   # Start with 50 epochs

pretrained:
  use_pretrained: true             # Recommended for fast training
```

## 🏋️ Step 5: Train Model (3-60 minutes depending on dataset size)

```bash
# Start training
uv run python clearpixai/training/detector/train_from_config.py \
    --config configs/train_config.yaml \
    --verbose

# Monitor with TensorBoard (in separate terminal)
tensorboard --logdir checkpoints
# Open http://localhost:6006 in browser
```

**Expected Training Time**:
- Small dataset (10-50 pairs): 3-10 minutes
- Medium dataset (50-200 pairs): 10-30 minutes
- Large dataset (200+ pairs): 30-60+ minutes

## 📊 Step 6: Validate Model (1 minute)

```bash
# After training completes, validate the best checkpoint
uv run python clearpixai/training/detector/validate.py \
    --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
    --data-dir /path/to/validation/data \
    --output validation_metrics.json

# Check metrics
cat validation_metrics.json
```

**Quality Gates**:
- ✅ IoU ≥ 0.80: Production ready
- ⚠️ IoU < 0.80: Need more data or training

## 📤 Step 7: Export Model (1 minute)

```bash
# Export to HuggingFace format
uv run python clearpixai/training/detector/export_model.py \
    --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
    --output-dir exported_models/my_model_v1 \
    --model-name "my-watermark-detector-v1"
```

## 🎯 Step 8: Use Your Model (1 minute)

```bash
# Test watermark removal with your trained model
uv run clearpixai \
    -i test_image.jpg \
    -o output_clean.jpg \
    --segmentation-weights exported_models/my_model_v1/pytorch_model.pth
```

## 🎉 That's It!

You now have a trained, validated, and exported watermark detection model!

## 🆘 Troubleshooting

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
data:
  batch_size: 4  # or even 2
```

### Poor Results (IoU < 0.70)

1. Check if watermarks are similar to training data
2. Collect more diverse training samples
3. Train for more epochs
4. Try `loss_fn: "combined"` in config