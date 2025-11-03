#!/usr/bin/env python3
"""Simple script to train the watermark detector.

This is a convenience script that wraps the training module.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clearpixai.training.detector.train import train


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train watermark detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="clearpixai/training/detector/data/train",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="mit_b5",
        help="Encoder architecture name",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pretrained weights for encoder",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Target image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (reduced for small datasets)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="combined",
        choices=["dice", "bce", "combined"],
        help="Loss function",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Lightning checkpoint to resume training from",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="clearpixai/detection/best_watermark_model_mit_b5_best.pth",
        help="Path to pretrained model weights (.pth) for finetuning",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch (ignore pretrained weights)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        print(f"\nPlease create the directory and add image pairs:")
        print(f"  {args.data_dir}/")
        print(f"    ├── image0.jpg")
        print(f"    ├── image0 clean.jpg")
        print(f"    ├── image1.jpg")
        print(f"    └── image1 clean.jpg")
        sys.exit(1)
    
    # Check for image files
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    if not image_files:
        print(f"Error: No images found in '{args.data_dir}'!")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in '{args.data_dir}'")
    
    # Check if pretrained weights exist
    pretrained_path = None if args.from_scratch else args.pretrained_weights
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\n🎯 FINETUNING MODE")
        print(f"   Starting from: {pretrained_path}")
    elif pretrained_path:
        print(f"\n⚠️  Warning: Pretrained weights not found at {pretrained_path}")
        print(f"   Training from scratch instead")
        pretrained_path = None
    else:
        print(f"\n🔨 TRAINING FROM SCRATCH")
    
    print(f"\nTraining configuration:")
    print(f"  Encoder: {args.encoder_name}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Loss function: {args.loss_fn}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Accelerator: {args.accelerator}")
    print()
    
    # Train
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        loss_fn=args.loss_fn,
        val_split=args.val_split,
        resume_from_checkpoint=args.resume,
        pretrained_weights=pretrained_path,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()

