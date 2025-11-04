# ClearPixAI - MLOps Project Documentation

> **Note**: This document provides comprehensive MLOps-focused documentation for the ClearPixAI project, including target metrics, dataset details, reproducibility guidelines, and production-ready training pipeline.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Target Metrics](#target-metrics)
3. [Dataset Description](#dataset-description)
4. [Experiment Plan](#experiment-plan)
5. [Reproducible Training](#reproducible-training)
6. [Model Validation](#model-validation)
7. [Production Deployment](#production-deployment)
8. [Project Structure](#project-structure)

---

## 🎯 Project Overview

### Business Goal

**ClearPixAI** is an AI-powered watermark removal system designed to automatically detect and remove watermarks from images with high quality and reliability. The system serves photographers, content creators, and digital archivists who need to restore watermarked images to their original state.

### Technical Approach

The system uses a two-stage pipeline:
1. **Watermark Detection**: Segmentation-based detection using a U-Net model with MiT-B5 encoder
2. **Inpainting**: SDXL/SD-based diffusion inpainting to restore image content

This MLOps project focuses on the **watermark detection model** training pipeline.

### Key Features

- ✅ Production-ready training pipeline with full reproducibility
- ✅ Configuration-based training for easy experimentation
- ✅ Comprehensive logging and monitoring
- ✅ Data validation and quality checks
- ✅ Model versioning and export to HuggingFace format
- ✅ Automated validation and metric reporting

---

## 🎯 Target Metrics

### Model Quality Metrics (ML Metrics)

| Metric | Target | Minimum Acceptable | Business Impact |
|--------|--------|-------------------|-----------------|
| **IoU (Intersection over Union)** | ≥ 0.90 | ≥ 0.80 | Core quality metric for segmentation accuracy |
| **Dice Coefficient (F1)** | ≥ 0.90 | ≥ 0.80 | Balance between precision and recall |
| **Precision** | ≥ 0.85 | ≥ 0.75 | Minimize false positives (removing non-watermark areas) |
| **Recall** | ≥ 0.85 | ≥ 0.75 | Minimize false negatives (missing watermark areas) |

**Quality Assessment**:
- IoU ≥ 0.90: **EXCELLENT** ✨ - Production ready
- IoU ≥ 0.80: **GOOD** ✓ - Acceptable for production
- IoU ≥ 0.70: **ACCEPTABLE** ~ - Needs improvement
- IoU < 0.70: **NEEDS IMPROVEMENT** ⚠ - Not production ready

### Production Service Metrics (Technical SLA)

| Metric | Target | Critical Threshold | Monitoring |
|--------|--------|-------------------|-----------|
| **Average Response Time** | ≤ 200 ms | ≤ 500 ms | Per-request latency (detection only) |
| **95th Percentile Latency** | ≤ 300 ms | ≤ 1000 ms | Tail latency monitoring |
| **Failed Request Rate** | ≤ 0.1% | ≤ 1% | Error rate tracking |
| **GPU Memory Usage** | ≤ 4 GB | ≤ 8 GB | Resource utilization |
| **CPU Usage (per request)** | ≤ 50% | ≤ 80% | CPU efficiency |
| **Model Load Time** | ≤ 5 sec | ≤ 10 sec | Cold start performance |

### Business Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **User Satisfaction** | ≥ 4.0/5.0 | Average user rating for watermark removal quality |
| **Processing Success Rate** | ≥ 95% | Percentage of images successfully processed |
| **False Detection Rate** | ≤ 5% | Images with watermarks detected when none present |

---

## 📊 Dataset Description

### Dataset Overview

**Name**: Custom Watermark Detection Dataset  
**Task**: Binary segmentation (watermark vs. background)  
**Format**: Image pairs (watermarked + clean)

### Dataset Structure

The dataset expects paired images:

```
data/
├── clean/                 # Clean images without watermarks
│   ├── clean-0000.jpg
│   ├── clean-0001.jpg
│   └── ...
└── watermarked/          # Images with watermarks
    ├── clean-0000_v1.jpg  # Can have multiple versions
    ├── clean-0000_v2.jpg
    ├── clean-0001_v1.jpg
    └── ...
```

Alternative flat structure:
```
data/
├── image0.jpg            # Watermarked
├── image0_clean.jpg      # Clean
├── image1.jpg
├── image1_clean.jpg
└── ...
```

### Data Preprocessing

1. **Automatic Mask Generation**: Masks are automatically created by computing the difference between watermarked and clean images
2. **Image Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Resizing**: Images resized to 512x512 during training (configurable)
4. **Augmentation**: Comprehensive augmentation pipeline including:
   - Geometric: flips, rotations, shifts, scales
   - Color: brightness, contrast, hue, saturation
   - Noise: Gaussian noise, blur, motion blur

### Data Validation

Before training, validate your dataset:

```bash
# Validate dataset structure and quality
uv run python clearpixai/training/detector/validate_data.py \
    --data-dir /path/to/data
```

This checks:
- ✅ File formats and integrity
- ✅ Image dimensions and properties
- ✅ Corrupted files detection
- ✅ Basic statistics (dimensions, file sizes)
- ✅ Image pair matching

### Recommended Dataset Size

| Dataset Size | Use Case | Expected Quality |
|--------------|----------|------------------|
| 10-50 pairs | Quick prototyping, finetuning | Basic functionality |
| 50-200 pairs | Small-scale production | Good performance |
| 200-1000 pairs | Production deployment | High quality |
| 1000+ pairs | Large-scale production | Excellent performance |

### Data Sources

**For training your own model, you can use**:

1. **Kaggle Watermark Dataset**: Large-scale dataset with diverse watermarks
   - See https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset/data
   
2. **Custom Dataset**: Create your own paired images
   - Collect images with watermarks
   - Create clean versions (or vice versa)
   - Ensure consistent image quality

   You can download my dataset with this link https://disk.360.yandex.ru/d/AEUb2BnCMXGWLw

---

### Experiment Tracking

All experiments are logged to TensorBoard:

```bash
# View experiment results
tensorboard --logdir checkpoints/watermark_detection
```

**Tracked Metrics**:
- Training/Validation Loss
- IoU, Dice, Precision, Recall
- Learning Rate
- Sample Predictions

---

## 🔄 Reproducible Training

### Prerequisites

**System Requirements**:
- Python 3.10+
- CUDA-capable GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM

**Install Dependencies**:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 1: Prepare Dataset

Make your own, download my or from kaggle, then:

```bash
# 1. Validate your dataset
uv run python clearpixai/training/detector/validate_data.py \
    --data-dir /path/to/your/data

# Expected output:
# ✅ Dataset validation PASSED
# Dataset is ready for training with N image pairs
```

### Step 2: Configure Training

Edit `configs/train_config.yaml` or create your own config:

```yaml
# Key parameters to adjust
random_seed: 42              # For reproducibility
data:
  data_dir: "path/to/data"   # Your data directory
  batch_size: 8              # Adjust based on GPU memory
  image_size: 512            # Target image size

model:
  encoder_name: "mit_b5"     # Model architecture
  loss_fn: "combined"        # Loss function

training:
  learning_rate: 0.0001      # Learning rate
  max_epochs: 100            # Maximum epochs

pretrained:
  use_pretrained: true       # Use transfer learning
  weights_path: "clearpixai/detection/best_watermark_model_mit_b5_best.pth"
```

### Step 3: Train Model

**Basic Training**:

```bash
# Train with config file
uv run python clearpixai/training/detector/train_from_config.py \
    --config configs/train_config.yaml \
    --verbose
```

**With Command-Line Overrides**:

```bash
# Override specific parameters
uv run python clearpixai/training/detector/train_from_config.py \
    --config configs/train_config.yaml \
    --data-dir /path/to/data \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --max-epochs 50 \
    --gpu 0 \
    --verbose
```

**Training on Specific GPU**:

```bash
# Use GPU 5
uv run python clearpixai/training/detector/train_from_config.py \
    --config configs/train_config.yaml \
    --gpu 5
```

### Step 4: Monitor Training

```bash
# In a separate terminal, launch TensorBoard
tensorboard --logdir checkpoints

# Open browser to http://localhost:6006
```

**What to Monitor**:
- Training/validation loss should decrease
- IoU/Dice should increase
- Check for overfitting (val_loss increases while train_loss decreases)

### Step 5: Validate Model

```bash
# Validate best checkpoint
uv run python clearpixai/training/detector/validate.py \
    --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
    --data-dir /path/to/validation/data \
    --output validation_results.json
```

**Expected Output**:

```
Validation Results
================================================================================
IoU (Intersection over Union): 0.8750
Dice Coefficient (F1 Score):   0.9333
Precision:                      0.9100
Recall:                         0.9200
Accuracy:                       0.9850

Model Quality: GOOD ✓
IoU Score: 0.8750 (Target: ≥ 0.80)
================================================================================
```

### Step 6: Export Model

```bash
# Export to HuggingFace format
uv run python clearpixai/training/detector/export_model.py \
    --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
    --output-dir exported_models/my_model \
    --model-name "clearpixai-watermark-detector-v1" \
    --description "Finetuned on custom dataset"
```

**Exported Files**:
- `pytorch_model.pth` - PyTorch state dict
- `model.safetensors` - SafeTensors format
- `config.json` - Model configuration
- `hyperparameters.json` - Training hyperparameters
- `README.md` - Model card

### Reproducibility Checklist

- ✅ **Fixed Random Seed**: Set in config (`random_seed: 42`)
- ✅ **Deterministic Operations**: Enabled in trainer
- ✅ **Version Pinning**: All dependencies pinned in `requirements.txt`
- ✅ **Configuration Saved**: Training config saved with checkpoints
- ✅ **Environment Documentation**: System requirements documented
- ✅ **Data Versioning**: Dataset structure and validation documented

### Quick Start (One Command)

```bash
# Complete training pipeline
uv run python clearpixai/training/detector/train_from_config.py \
    --config configs/train_config.yaml \
    --data-dir clearpixai/training/detector/data/train \
    --verbose
```

---

## ✅ Model Validation

### Validation Pipeline

```bash
# 1. Validate data quality
uv run python clearpixai/training/detector/validate_data.py \
    --data-dir /path/to/val/data

# 2. Run model validation
uv run python clearpixai/training/detector/validate.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --data-dir /path/to/val/data \
    --output metrics.json

# 3. Check metrics
cat metrics.json
```

### Validation Metrics

The validation script outputs:

**Console Output**:
- IoU, Dice, Precision, Recall, Accuracy
- Confusion matrix (TP, FP, FN, TN)
- Per-batch statistics (mean, std, min, max, median)
- Quality assessment (EXCELLENT/GOOD/ACCEPTABLE/NEEDS IMPROVEMENT)

**JSON Output** (`metrics.json`):
```json
{
  "aggregate_metrics": {
    "iou": 0.875,
    "dice": 0.933,
    "precision": 0.91,
    "recall": 0.92,
    "accuracy": 0.985
  },
  "per_batch_statistics": {
    "mean_iou": 0.875,
    "std_iou": 0.045,
    "min_iou": 0.750,
    "max_iou": 0.950
  },
  "quality_assessment": {
    "quality": "GOOD ✓",
    "meets_target": true
  }
}
```

### Quality Gates

Before deploying to production, ensure:

1. ✅ **IoU ≥ 0.80** (minimum acceptable)
2. ✅ **Precision ≥ 0.75** (minimize false positives)
3. ✅ **Recall ≥ 0.75** (minimize false negatives)
4. ✅ **Validation on diverse test set** (different watermark types)
5. ✅ **Stable performance** (low std across batches)

---

## 🚀 Production Deployment

### Model Deployment Steps

1. **Export Model**:
```bash
uv run python clearpixai/training/detector/export_model.py \
    --checkpoint best_model.ckpt \
    --output-dir production_models/v1.0
```

2. **Test Inference**:
```bash
# Test with sample image
uv run clearpixai -i test_image.jpg -o output.jpg
```

3. **Performance Benchmarking**:
```bash
# Measure latency and throughput
# (Add benchmarking script if needed)
```

4. **Deploy to Production**:
   - Copy model to production server
   - Update model path in production config
   - Run integration tests
   - Monitor metrics

### Model Serving

**Using ClearPixAI CLI**:

```bash
# Basic usage
uv run clearpixai -i input.jpg -o output.jpg

# With custom model
uv run clearpixai -i input.jpg -o output.jpg \
    --segmentation-weights production_models/v1.0/pytorch_model.pth
```

**Using Python API**:

```python
from pathlib import Path
from clearpixai.pipeline import PipelineConfig, remove_watermark

config = PipelineConfig()
config.segmentation.weights = Path("production_models/v1.0/pytorch_model.pth")
config.segmentation.threshold = 0.5

remove_watermark(
    input_path=Path("input.jpg"),
    output_path=Path("output.jpg"),
    config=config
)
```

### Monitoring in Production

**Key Metrics to Track**:

1. **Latency**: Response time per request
2. **Throughput**: Requests per second
3. **Error Rate**: Failed requests / total requests
4. **Resource Usage**: GPU/CPU/Memory utilization
5. **Quality Metrics**: User feedback, manual QA sampling

**Recommended Tools**:
- Prometheus + Grafana for metrics
- ELK Stack for logging
- TensorBoard for model monitoring

---

## 📁 Project Structure

```
clearpixai/
├── configs/                          # Configuration files
│   └── train_config.yaml            # Main training configuration
│
├── clearpixai/                       # Main package
│   ├── training/                     # Training pipeline
│   │   ├── config.py                # Configuration management
│   │   └── detector/                # Detector training
│   │       ├── train.py             # Original training script
│   │       ├── train_from_config.py # Config-based training
│   │       ├── validate.py          # Model validation
│   │       ├── validate_data.py     # Data validation
│   │       ├── export_model.py      # Model export
│   │       ├── model.py             # PyTorch Lightning model
│   │       └── dataset.py           # Dataset loader
│   │
│   ├── detection/                   # Detection module
│   │   ├── segmentation.py         # Segmentation detector
│   │   └── best_watermark_model_mit_b5_best.pth  # Pretrained weights
│   │
│   ├── inpaint/                     # Inpainting backends
│   ├── pipeline.py                  # Main pipeline
│   └── cli.py                       # Command-line interface
│
├── requirements.txt                  # Pinned dependencies
├── pyproject.toml                   # Project metadata
├── README.md                        # User documentation
└── README_MLOPS.md                  # This file (MLOps documentation)
```

### Key Files

- **`configs/train_config.yaml`**: Central configuration for training
- **`requirements.txt`**: Pinned dependencies for reproducibility
- **`clearpixai/training/config.py`**: Configuration management
- **`clearpixai/training/detector/train_from_config.py`**: Production training script
- **`clearpixai/training/detector/validate.py`**: Model validation
- **`clearpixai/training/detector/validate_data.py`**: Data quality checks

---


