"""Watermark detection model using PyTorch Lightning."""

from typing import Optional, List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WatermarkDetectionModel(pl.LightningModule):
    """PyTorch Lightning module for watermark detection.
    
    Based on the Diffusion-Dynamics watermark-segmentation approach.
    """
    
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        learning_rate: float = 1e-4,
        loss_fn: str = "dice",
        pretrained_checkpoint: Optional[str] = None,
    ):
        """Initialize model.
        
        Args:
            encoder_name: Encoder architecture name
            encoder_weights: Pretrained weights for encoder
            in_channels: Number of input channels
            classes: Number of output classes
            learning_rate: Learning rate for optimizer
            loss_fn: Loss function name ('dice', 'bce', or 'combined')
            pretrained_checkpoint: Path to pretrained model weights (.pth file)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create segmentation model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self._load_pretrained_weights(pretrained_checkpoint)
        
        # Loss function
        if loss_fn == "dice":
            self.loss_fn = smp.losses.DiceLoss(mode="binary")
        elif loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_fn == "combined":
            # Combination of Dice and BCE
            self.dice_loss = smp.losses.DiceLoss(mode="binary")
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        self.loss_name = loss_fn
        
        # Metrics
        self.train_iou = []
        self.val_iou = []
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean up state dict (remove Lightning prefixes)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Skip metadata keys
            if key in ["std", "mean"]:
                continue
            
            # Remove common prefixes
            new_key = key
            if key.startswith("model."):
                new_key = key[len("model."):]
            elif key.startswith("net."):
                new_key = key[len("net."):]
            
            cleaned_state_dict[new_key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(
            cleaned_state_dict, strict=False
        )
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info("Successfully loaded pretrained weights!")
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def shared_step(self, batch, stage: str):
        """Shared step for training and validation.
        
        Args:
            batch: Batch data
            stage: 'train' or 'val'
            
        Returns:
            Loss value
        """
        images, masks = batch
        
        # Ensure masks have correct shape [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        if self.loss_name == "combined":
            loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)
        else:
            loss = self.loss_fn(logits, masks)
        
        # Calculate IoU metric
        pr_masks = torch.sigmoid(logits)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pr_masks,
            masks.long(),
            mode="binary",
            threshold=0.5,
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_iou", iou_score, prog_bar=True, on_step=False, on_epoch=True)
        
        if stage == "train":
            self.train_iou.append(iou_score)
        else:
            self.val_iou.append(iou_score)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_iou:
            avg_iou = torch.stack(self.train_iou).mean()
            self.log("train_epoch_iou", avg_iou)
            self.train_iou.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_iou:
            avg_iou = torch.stack(self.val_iou).mean()
            self.log("val_epoch_iou", avg_iou)
            self.val_iou.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou",
            },
        }

