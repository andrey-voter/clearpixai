#!/usr/bin/env python3
"""Quick test to verify YOLO dataset loader works correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clearpixai.training.detector.yolo_dataset import (
    YOLOWatermarkDataset,
    get_training_augmentation,
    get_validation_augmentation
)


def test_dataset():
    """Test the YOLO dataset loader."""
    
    data_dir = Path("clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset")
    
    print("\n" + "="*60)
    print("Testing YOLO Watermark Dataset Loader")
    print("="*60)
    
    # Test training dataset
    print("\n1. Loading Training Dataset...")
    train_dataset = YOLOWatermarkDataset(
        data_dir=data_dir,
        image_size=512,
        transform=get_training_augmentation(512),
        split="train",
    )
    print(f"✅ Training dataset loaded: {len(train_dataset):,} samples")
    
    # Test validation dataset
    print("\n2. Loading Validation Dataset...")
    val_dataset = YOLOWatermarkDataset(
        data_dir=data_dir,
        image_size=512,
        transform=get_validation_augmentation(512),
        split="val",
    )
    print(f"✅ Validation dataset loaded: {len(val_dataset):,} samples")
    
    # Test loading a sample
    print("\n3. Loading Sample from Training Set...")
    image, mask = train_dataset[0]
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Mask shape: {mask.shape}")
    print(f"✅ Image type: {image.dtype}")
    print(f"✅ Mask type: {mask.dtype}")
    print(f"✅ Mask has watermark pixels: {mask.sum().item() > 0}")
    
    # Test multiple samples
    print("\n4. Testing Multiple Samples...")
    samples_to_test = min(5, len(train_dataset))
    for i in range(samples_to_test):
        image, mask = train_dataset[i]
        has_watermark = mask.sum().item() > 0
        status = "✅" if has_watermark else "⚠️"
        print(f"  Sample {i}: Image {image.shape}, Mask {mask.shape}, Has watermark: {status}")
    
    print("\n" + "="*60)
    print("✅ All Tests Passed!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"  - Training samples: {len(train_dataset):,}")
    print(f"  - Validation samples: {len(val_dataset):,}")
    print(f"  - Image size: {image.shape}")
    print(f"  - Mask size: {mask.shape}")
    print("\n🚀 Dataset is ready for training!")
    print("\nRun this to start training:")
    print("  uv run python train_kaggle.py \\")
    print("      --data-dir clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset \\")
    print("      --pretrained-weights clearpixai/detection/best_watermark_model_mit_b5_best.pth")
    print()


if __name__ == "__main__":
    try:
        test_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

