#!/usr/bin/env python3
"""Convenience script to train on Kaggle watermark dataset.

This script trains the watermark detector using the Large-scale Common
Watermark Dataset from Kaggle (YOLO format).
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clearpixai.training.detector.train_yolo import train_yolo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train watermark detector on Kaggle dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to extracted Kaggle dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints_kaggle",
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
        default=6,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (None = use all). E.g., 10000 for first 10k images",
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        print(f"\nPlease:")
        print(f"  1. Download the Kaggle dataset from:")
        print(f"     https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset")
        print(f"  2. Extract it to a directory")
        print(f"  3. Run this script with --data-dir pointing to that directory")
        sys.exit(1)
    
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
    
    print(f"\nDataset: {args.data_dir}")
    print(f"\nTraining configuration:")
    print(f"  Encoder: {args.encoder_name}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Loss function: {args.loss_fn}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Accelerator: {args.accelerator}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples} (subset training)")
    else:
        print(f"  Max samples: All available")
    print()
    
    # Train
    train_yolo(
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
        resume_from_checkpoint=args.resume,
        pretrained_weights=pretrained_path,
        accelerator=args.accelerator,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

