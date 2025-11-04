"""Export trained model to HuggingFace-compatible format.

This script exports a trained PyTorch Lightning checkpoint to a format
compatible with HuggingFace Hub, following MLOps best practices.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clearpixai.training.detector.model import WatermarkDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """Export model to HuggingFace-compatible format.
    
    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        output_dir: Output directory for exported model
        model_name: Optional model name for metadata
        description: Optional model description
        metadata: Optional additional metadata
    """
    logger.info("="*80)
    logger.info("ClearPixAI Model Export to HuggingFace Format")
    logger.info("="*80)
    
    # Validate checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    try:
        model = WatermarkDetectionModel.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Extract model hyperparameters
    hparams = model.hparams
    logger.info("\nModel Hyperparameters:")
    for key, value in hparams.items():
        logger.info(f"  {key}: {value}")
    
    # Save model state dict (PyTorch format)
    model_path = output_dir / "pytorch_model.pth"
    logger.info(f"\nSaving PyTorch model to: {model_path}")
    torch.save(model.model.state_dict(), model_path)
    logger.info("✓ PyTorch model saved")
    
    # Save as safetensors (HuggingFace preferred format)
    try:
        from safetensors.torch import save_file
        safetensors_path = output_dir / "model.safetensors"
        logger.info(f"Saving safetensors to: {safetensors_path}")
        save_file(model.model.state_dict(), safetensors_path)
        logger.info("✓ Safetensors model saved")
    except ImportError:
        logger.warning("safetensors not available, skipping safetensors export")
    
    # Create config.json for HuggingFace
    config = {
        "model_type": "segmentation",
        "architecture": "unet",
        "encoder_name": hparams.get('encoder_name', 'mit_b5'),
        "encoder_weights": hparams.get('encoder_weights', 'imagenet'),
        "in_channels": hparams.get('in_channels', 3),
        "classes": hparams.get('classes', 1),
        "task": "watermark_detection",
        "framework": "pytorch",
        "library": "segmentation_models_pytorch",
    }
    
    # Add custom metadata
    if model_name:
        config["model_name"] = model_name
    if description:
        config["description"] = description
    if metadata:
        config["metadata"] = metadata
    
    config_path = output_dir / "config.json"
    logger.info(f"\nSaving config to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("✓ Config saved")
    
    # Save hyperparameters
    hparams_path = output_dir / "hyperparameters.json"
    logger.info(f"Saving hyperparameters to: {hparams_path}")
    with open(hparams_path, 'w') as f:
        # Convert hparams to dict and make JSON serializable
        hparams_dict = dict(hparams)
        json.dump(hparams_dict, f, indent=2, default=str)
    logger.info("✓ Hyperparameters saved")
    
    # Create README for model card
    readme_content = f"""# {model_name or 'ClearPixAI Watermark Detection Model'}

{description or 'Watermark detection model using segmentation.'}

## Model Details

- **Architecture**: U-Net with {hparams.get('encoder_name', 'mit_b5')} encoder
- **Task**: Binary segmentation for watermark detection
- **Framework**: PyTorch
- **Library**: segmentation_models_pytorch
- **Input**: RGB images (3 channels)
- **Output**: Binary mask (1 channel)

## Hyperparameters

```json
{json.dumps(dict(hparams), indent=2, default=str)}
```

## Usage

### Loading the Model

```python
import torch
import segmentation_models_pytorch as smp

# Load model
model = smp.Unet(
    encoder_name="{hparams.get('encoder_name', 'mit_b5')}",
    encoder_weights=None,  # Load custom weights
    in_channels=3,
    classes=1,
)

# Load weights
state_dict = torch.load("pytorch_model.pth")
model.load_state_dict(state_dict)
model.eval()
```

### Inference

```python
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Prepare image
image = Image.open("input.jpg").convert("RGB")
image = np.array(image)

# Preprocessing
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

augmented = transform(image=image)
image_tensor = augmented['image'].unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(image_tensor)
    mask = torch.sigmoid(logits) > 0.5
```

## Training

This model was trained using the ClearPixAI training pipeline.

See the [ClearPixAI repository](https://github.com/yourusername/clearpixai) for training details.

## License

MIT License
"""
    
    readme_path = output_dir / "README.md"
    logger.info(f"Creating model card: {readme_path}")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    logger.info("✓ Model card created")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Export Complete!")
    logger.info("="*80)
    logger.info(f"Model exported to: {output_dir}")
    logger.info("\nExported files:")
    logger.info(f"  ✓ pytorch_model.pth      - PyTorch state dict")
    if (output_dir / "model.safetensors").exists():
        logger.info(f"  ✓ model.safetensors      - SafeTensors format")
    logger.info(f"  ✓ config.json            - Model configuration")
    logger.info(f"  ✓ hyperparameters.json   - Training hyperparameters")
    logger.info(f"  ✓ README.md              - Model card")
    logger.info("="*80 + "\n")
    
    logger.info("To use this model:")
    logger.info("  1. Load with PyTorch: torch.load('pytorch_model.pth')")
    logger.info("  2. Or upload to HuggingFace Hub for easy sharing")
    logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export model to HuggingFace format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="clearpixai-watermark-detector",
        help="Model name for metadata",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Watermark detection model trained with ClearPixAI",
        help="Model description",
    )
    
    args = parser.parse_args()
    
    try:
        export_to_huggingface(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            model_name=args.model_name,
            description=args.description,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

