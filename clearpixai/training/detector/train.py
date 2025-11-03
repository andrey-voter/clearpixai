"""Training script for watermark detection model."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import WatermarkDataset, get_training_augmentation, get_validation_augmentation
from .model import WatermarkDetectionModel


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    encoder_name: str = "mit_b5",
    encoder_weights: str = "imagenet",
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    loss_fn: str = "combined",
    val_split: float = 0.2,
    resume_from_checkpoint: str = None,
    pretrained_weights: str = None,
    accelerator: str = "auto",
):
    """Train watermark detection model.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints
        encoder_name: Encoder architecture name
        encoder_weights: Pretrained weights for encoder
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate
        loss_fn: Loss function ('dice', 'bce', or 'combined')
        val_split: Validation split ratio
        resume_from_checkpoint: Path to Lightning checkpoint to resume from
        pretrained_weights: Path to pretrained model weights (.pth) for finetuning
        accelerator: Device to use ('auto', 'gpu', 'cpu')
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    full_dataset = WatermarkDataset(
        data_dir=Path(data_dir),
        image_size=image_size,
        transform=get_training_augmentation(image_size),
        create_masks=True,
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = get_validation_augmentation(image_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Create model
    model = WatermarkDetectionModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        pretrained_checkpoint=pretrained_weights,
    )
    
    if pretrained_weights:
        print(f"\n🎯 Finetuning from pretrained checkpoint: {pretrained_weights}")
    else:
        print(f"\n🔨 Training from scratch with {encoder_weights} initialization")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="watermark-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        patience=15,
        mode="max",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="watermark_detection",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train watermark detection model")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
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
        default=8,
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
        default=100,
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
        default=None,
        help="Path to pretrained model weights (.pth) for finetuning",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
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
        pretrained_weights=args.pretrained_weights,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()

