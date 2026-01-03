# ClearPixAI - Quick Start

This guide takes you from a fresh environment to a trained and exported watermark detection model.

## Prerequisites

- Python 3.10+
- GPU with 8GB+ VRAM (recommended) or CPU
- `uv` installed (recommended) or `pip`

## Install

If you are working from a fresh clone:

```bash
git clone <your-repo-url>
cd clearpixai
```

Install dependencies:

```bash
# Recommended
uv sync

# Alternative
pip install -r requirements.txt
```

## Prepare your dataset

ClearPixAI expects paired images (watermarked + clean). 

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

### Option C: example dataset

If you want a ready-made dataset, see the Kaggle dataset: [Large Scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset/data) or use the one I made: [my own dataset](https://disk.360.yandex.ru/d/AEUb2BnCMXGWLw).

## Validate your dataset

Before training, validate that the dataset is discoverable and pairs correctly:

```bash
uv run python clearpixai/training/detector/validate_data.py \
  --data-dir /path/to/your/data
```

## Configure training

Edit `configs/train_config.yaml` and set at minimum:

- `data.data_dir`: path to your dataset
- `data.batch_size`: reduce if you hit OOM

Example configuration:

```yaml
random_seed: 42

data:
  data_dir: "/path/to/your/data"
  batch_size: 8
  image_size: 512

training:
  learning_rate: 0.0001
  max_epochs: 50

pretrained:
  use_pretrained: true
```

## Train

```bash
uv run python clearpixai/training/detector/train_from_config.py \
  --config configs/train_config.yaml \
  --verbose
```

Optional: monitor training with TensorBoard:

```bash
tensorboard --logdir checkpoints
```

Then open `http://localhost:6006` in your browser.

Typical training time (highly dataset-dependent):

- Small dataset (10-50 pairs): 3-10 minutes
- Medium dataset (50-200 pairs): 10-30 minutes
- Large dataset (200+ pairs): 30-60+ minutes

## Validate

After training completes, validate a checkpoint and save metrics:

```bash
uv run python clearpixai/training/detector/validate.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --data-dir /path/to/validation/data \
  --output validation_metrics.json

cat validation_metrics.json
```

Quality gate (recommended): **IoU ≥ 0.80**.

## Export

```bash
uv run python clearpixai/training/detector/export_model.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --output-dir exported_models/my_model_v1 \
  --model-name "my-watermark-detector-v1"
```

## Run inference (CLI)

Use your exported segmentation weights:

```bash
uv run clearpixai \
  -i test_image.jpg \
  -o test_image_cleaned.jpg \
  --segmentation-weights exported_models/my_model_v1/pytorch_model.pth
```

## Troubleshooting

### Out of memory (OOM)

Reduce batch size in `configs/train_config.yaml`:

```yaml
data:
  batch_size: 4
```

### Poor results (IoU < 0.70)

- Confirm that your training data contains similar watermark styles to your target images.
- Increase dataset diversity and size.
- Train for more epochs.
- Consider trying `loss_fn: "combined"` in the training config.