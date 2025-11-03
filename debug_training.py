#!/usr/bin/env python3
"""Debug script to test a single training batch."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from clearpixai.training.detector.yolo_dataset import (
    YOLOWatermarkDataset,
    get_training_augmentation,
)
from clearpixai.training.detector.model import WatermarkDetectionModel
from torch.utils.data import DataLoader


def debug_batch():
    """Test loading and processing a single batch."""
    
    print("=" * 60)
    print("Debug: Testing Single Batch")
    print("=" * 60)
    
    # Create dataset
    data_dir = Path("clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset")
    dataset = YOLOWatermarkDataset(
        data_dir=data_dir,
        image_size=512,
        transform=get_training_augmentation(512),
        split="train",
    )
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Get one batch
    print("\n1. Loading batch...")
    batch = next(iter(loader))
    images, masks = batch
    
    print(f"✅ Images shape: {images.shape}")
    print(f"✅ Images dtype: {images.dtype}")
    print(f"✅ Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"✅ Masks shape: {masks.shape}")
    print(f"✅ Masks dtype: {masks.dtype}")
    print(f"✅ Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"✅ Masks unique values: {masks.unique()}")
    
    # Create model
    print("\n2. Creating model...")
    model = WatermarkDetectionModel(
        encoder_name="mit_b5",
        encoder_weights="imagenet",
        learning_rate=1e-4,
        loss_fn="combined",
        pretrained_checkpoint="clearpixai/detection/best_watermark_model_mit_b5_best.pth",
    )
    model.eval()
    
    print(f"✅ Model created")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    with torch.no_grad():
        logits = model(images)
    
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Output dtype: {logits.dtype}")
    print(f"✅ Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Check shapes match
    print("\n4. Checking shape compatibility...")
    print(f"   Logits: {logits.shape}")
    print(f"   Masks: {masks.shape}")
    
    # Masks should be (B, H, W) but logits are (B, 1, H, W)
    # Need to add channel dimension to masks or squeeze logits
    if logits.shape[1] == 1:
        print("   ⚠️  Logits have channel dim, masks don't")
        print(f"   Need to: masks.unsqueeze(1) -> {(masks.shape[0], 1, masks.shape[1], masks.shape[2])}")
    
    # Test loss calculation
    print("\n5. Testing loss calculation...")
    model.train()
    try:
        # Manually compute loss to see what happens
        masks_with_channel = masks.unsqueeze(1)  # Add channel dimension
        print(f"   Masks with channel: {masks_with_channel.shape}")
        
        # Dice loss
        from segmentation_models_pytorch.losses import DiceLoss
        dice_loss_fn = DiceLoss(mode="binary", from_logits=True)
        dice_loss = dice_loss_fn(logits, masks_with_channel)
        print(f"✅ Dice loss: {dice_loss.item():.4f}")
        
        # BCE loss
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(logits, masks_with_channel)
        print(f"✅ BCE loss: {bce_loss.item():.4f}")
        
        # Combined
        combined_loss = 0.5 * dice_loss + 0.5 * bce_loss
        print(f"✅ Combined loss: {combined_loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Error during loss calculation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training step
    print("\n6. Testing training_step...")
    try:
        batch = (images, masks)
        loss = model.training_step(batch, 0)
        print(f"✅ Training step loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Error during training_step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = debug_batch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

