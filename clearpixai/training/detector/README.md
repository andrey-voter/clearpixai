# Watermark Detector Training

Training pipeline for watermark detection model based on the [Diffusion-Dynamics/watermark-segmentation](https://github.com/Diffusion-Dynamics/watermark-segmentation) approach.

## Features

- **PyTorch Lightning**: Modern training framework with built-in features
- **Segmentation Models PyTorch**: State-of-the-art encoder architectures (MIT-B5, ResNet, EfficientNet, etc.)
- **Automatic Mask Generation**: Creates masks from watermarked/clean image pairs
- **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- **Checkpointing**: Automatic saving of best models
- **TensorBoard Logging**: Training metrics visualization
- **Early Stopping**: Prevents overfitting

## Dataset Structure

Place your training data in the `data/train/` directory with the following structure:

```
data/train/
├── image0.jpg          # Watermarked image
├── image0 clean.jpg    # Clean version (no watermark)
├── image1.jpg
├── image1 clean.jpg
└── ...
```

The dataset will automatically:
1. Find image pairs (watermarked and clean)
2. Generate binary masks from the difference
3. Apply augmentations

## Training

### Basic Usage

```bash
python -m clearpixai.training.detector.train \
    --data-dir clearpixai/training/detector/data/train \
    --output-dir checkpoints
```

### Advanced Options

```bash
python -m clearpixai.training.detector.train \
    --data-dir clearpixai/training/detector/data/train \
    --output-dir checkpoints \
    --encoder-name mit_b5 \
    --encoder-weights imagenet \
    --image-size 512 \
    --batch-size 8 \
    --max-epochs 100 \
    --learning-rate 1e-4 \
    --loss-fn combined \
    --val-split 0.2 \
    --accelerator gpu
```

### Resume Training

```bash
python -m clearpixai.training.detector.train \
    --data-dir clearpixai/training/detector/data/train \
    --resume checkpoints/watermark-last.ckpt
```

## Parameters

- `--data-dir`: Directory containing training data (required)
- `--output-dir`: Directory to save checkpoints (default: `checkpoints`)
- `--encoder-name`: Encoder architecture (default: `mit_b5`)
  - Options: `mit_b5`, `resnet50`, `efficientnet-b5`, etc.
- `--encoder-weights`: Pretrained weights (default: `imagenet`)
- `--image-size`: Target image size (default: 512)
- `--batch-size`: Batch size (default: 8)
- `--num-workers`: Data loading workers (default: 4)
- `--max-epochs`: Maximum training epochs (default: 100)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--loss-fn`: Loss function (default: `combined`)
  - Options: `dice`, `bce`, `combined`
- `--val-split`: Validation split ratio (default: 0.2)
- `--resume`: Path to checkpoint to resume from
- `--accelerator`: Device to use (default: `auto`)
  - Options: `auto`, `gpu`, `cpu`

## Model Architecture

The training uses a U-Net architecture with various encoder backbones:

- **Encoder**: Pretrained on ImageNet (MIT-B5, ResNet, EfficientNet, etc.)
- **Decoder**: U-Net decoder with skip connections
- **Output**: Binary mask (watermark vs. background)

## Loss Functions

- **Dice Loss**: Measures overlap between prediction and ground truth
- **BCE Loss**: Binary cross-entropy loss
- **Combined**: Combination of Dice + BCE for better convergence

## Augmentations

Training augmentations include:
- Random horizontal/vertical flips
- Random 90° rotations
- Shift/scale/rotate
- Elastic transforms
- Grid/optical distortions
- Gaussian noise/blur
- Brightness/contrast adjustments

## Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir checkpoints/watermark_detection
```

Metrics tracked:
- Training/validation loss
- Training/validation IoU
- Learning rate

## Checkpoints

The training saves:
- **Top 3 models**: Based on validation IoU
- **Last checkpoint**: For resuming training
- **Best model**: Highest validation IoU

Checkpoint naming: `watermark-{epoch:02d}-{val_iou:.4f}.ckpt`

## Converting to Inference Model

After training, convert the PyTorch Lightning checkpoint to a standard PyTorch model:

```python
from clearpixai.training.detector.model import WatermarkDetectionModel

# Load checkpoint
model = WatermarkDetectionModel.load_from_checkpoint("checkpoints/best.ckpt")

# Save as standard PyTorch model
torch.save(model.model.state_dict(), "best_watermark_model.pth")
```

## Requirements

Install dependencies:

```bash
pip install pytorch-lightning segmentation-models-pytorch albumentations
```

## Tips for Better Results

1. **More Data**: The model will perform better with more training pairs
2. **Diverse Watermarks**: Include various watermark types, sizes, and positions
3. **Image Quality**: Use high-quality images for better mask generation
4. **Augmentation**: Adjust augmentation strength based on your dataset
5. **Learning Rate**: Start with 1e-4 and adjust if needed
6. **Batch Size**: Increase if you have enough GPU memory

## Current Limitations

With only 2 training pairs, the model will:
- Overfit quickly
- Not generalize well to new watermarks
- Require more data for production use

Consider:
- Collecting more data
- Using synthetic watermarks
- Data augmentation techniques
- Transfer learning from pretrained models

