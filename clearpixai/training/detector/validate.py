"""Validation script for trained watermark detection model.

This script evaluates a trained model on a validation/test dataset
and outputs comprehensive metrics to console and JSON file.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clearpixai.training.detector.dataset import (
    WatermarkDataset,
    get_validation_augmentation,
)
from clearpixai.training.detector.model import WatermarkDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive segmentation metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth masks
        threshold: Threshold for binarization
    
    Returns:
        Dictionary of metrics
    """
    # Apply sigmoid if predictions are logits
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    
    # Calculate statistics
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_binary.long(),
        targets.long(),
        mode="binary",
        threshold=threshold,
    )
    
    # Calculate metrics
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'tp': tp.sum().item(),
        'fp': fp.sum().item(),
        'fn': fn.sum().item(),
        'tn': tn.sum().item(),
    }


def validate_model(
    model: WatermarkDetectionModel,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Validate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Validation data loader
        device: Device to use for inference
    
    Returns:
        Tuple of (aggregate_metrics, per_sample_metrics)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    per_sample_metrics = []
    
    logger.info(f"Running validation on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Ensure masks have correct shape [B, 1, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass
            logits = model(images)
            predictions = torch.sigmoid(logits)
            
            # Calculate per-batch metrics
            batch_metrics = calculate_metrics(logits, masks)
            per_sample_metrics.append(batch_metrics)
            
            # Store for aggregate metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
    
    # Calculate aggregate metrics on all data
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    aggregate_metrics = calculate_metrics(all_predictions, all_targets)
    
    return aggregate_metrics, per_sample_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate trained watermark detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing validation data",
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
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save metrics JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mask binarization",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to validate (for debugging)",
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ClearPixAI Model Validation")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("")
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Validate data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Load model
    logger.info("Loading model from checkpoint...")
    try:
        model = WatermarkDetectionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=args.device,
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create validation dataset
    logger.info("\nLoading validation dataset...")
    try:
        val_dataset = WatermarkDataset(
            data_dir=data_dir,
            image_size=args.image_size,
            transform=get_validation_augmentation(args.image_size),
            create_masks=True,
            max_samples=args.max_samples,
        )
        logger.info(f"✓ Loaded {len(val_dataset)} validation samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Run validation
    logger.info("\nRunning validation...")
    aggregate_metrics, per_sample_metrics = validate_model(
        model=model,
        dataloader=val_loader,
        device=args.device,
    )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("Validation Results")
    logger.info("="*80)
    logger.info(f"IoU (Intersection over Union): {aggregate_metrics['iou']:.4f}")
    logger.info(f"Dice Coefficient (F1 Score):   {aggregate_metrics['dice']:.4f}")
    logger.info(f"Precision:                      {aggregate_metrics['precision']:.4f}")
    logger.info(f"Recall:                         {aggregate_metrics['recall']:.4f}")
    logger.info(f"Accuracy:                       {aggregate_metrics['accuracy']:.4f}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"  True Positives:  {aggregate_metrics['tp']:,}")
    logger.info(f"  False Positives: {aggregate_metrics['fp']:,}")
    logger.info(f"  False Negatives: {aggregate_metrics['fn']:,}")
    logger.info(f"  True Negatives:  {aggregate_metrics['tn']:,}")
    logger.info("="*80)
    
    # Calculate per-batch statistics
    batch_ious = [m['iou'] for m in per_sample_metrics]
    logger.info("\nPer-Batch Statistics:")
    logger.info(f"  Mean IoU:   {np.mean(batch_ious):.4f}")
    logger.info(f"  Std IoU:    {np.std(batch_ious):.4f}")
    logger.info(f"  Min IoU:    {np.min(batch_ious):.4f}")
    logger.info(f"  Max IoU:    {np.max(batch_ious):.4f}")
    logger.info(f"  Median IoU: {np.median(batch_ious):.4f}")
    
    # Quality assessment
    logger.info("\n" + "="*80)
    logger.info("Quality Assessment")
    logger.info("="*80)
    
    iou = aggregate_metrics['iou']
    if iou >= 0.90:
        quality = "EXCELLENT ✨"
    elif iou >= 0.80:
        quality = "GOOD ✓"
    elif iou >= 0.70:
        quality = "ACCEPTABLE ~"
    else:
        quality = "NEEDS IMPROVEMENT ⚠"
    
    logger.info(f"Model Quality: {quality}")
    logger.info(f"IoU Score: {iou:.4f} (Target: ≥ 0.80)")
    logger.info("="*80 + "\n")
    
    # Save metrics to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'checkpoint': str(args.checkpoint),
            'data_dir': str(args.data_dir),
            'num_samples': len(val_dataset),
            'image_size': args.image_size,
            'threshold': args.threshold,
            'aggregate_metrics': aggregate_metrics,
            'per_batch_statistics': {
                'mean_iou': float(np.mean(batch_ious)),
                'std_iou': float(np.std(batch_ious)),
                'min_iou': float(np.min(batch_ious)),
                'max_iou': float(np.max(batch_ious)),
                'median_iou': float(np.median(batch_ious)),
            },
            'quality_assessment': {
                'quality': quality,
                'meets_target': iou >= 0.80,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Metrics saved to: {output_path}")
    
    return aggregate_metrics


if __name__ == "__main__":
    main()

