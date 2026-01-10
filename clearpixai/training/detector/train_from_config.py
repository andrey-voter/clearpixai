"""Config-based training script for watermark detection model.

This script loads all parameters from a YAML config file, ensuring
reproducibility and ease of experimentation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clearpixai.training.config import load_config
from clearpixai.training.detector.dataset import (
    WatermarkDataset,
    get_training_augmentation,
    get_validation_augmentation,
)
from clearpixai.training.detector.model import WatermarkDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional path to log file
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )


def set_random_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seed: {seed}")
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic mode for better reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_from_config(config_path: str, verbose: bool = False, **overrides):
    """Train watermark detection model from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose logging
        **overrides: Configuration overrides as keyword arguments
    """
    # Setup logging
    setup_logging(verbose=verbose)
    
    logger.info("="*80)
    logger.info("ClearPixAI Watermark Detection Training")
    logger.info("="*80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path, overrides=overrides)
    
    # Setup MLflow tracking (local storage)
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = config.get('project.name', 'clearpixai_watermark_detector')
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {experiment_name}")
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_random_seed(seed)
    
    # Log configuration summary
    logger.info("\nTraining Configuration:")
    logger.info(f"  Project: {config.get('project.name', 'N/A')}")
    logger.info(f"  Version: {config.get('project.version', 'N/A')}")
    logger.info(f"  Random Seed: {seed}")
    logger.info(f"  Data Directory: {config.get('data.data_dir')}")
    logger.info(f"  Image Size: {config.get('data.image_size')}")
    logger.info(f"  Batch Size: {config.get('data.batch_size')}")
    logger.info(f"  Validation Split: {config.get('data.val_split')}")
    logger.info(f"  Encoder: {config.get('model.encoder_name')}")
    logger.info(f"  Learning Rate: {config.get('training.learning_rate')}")
    logger.info(f"  Max Epochs: {config.get('training.max_epochs')}")
    logger.info(f"  Loss Function: {config.get('model.loss_fn')}")
    
    # Create output directory
    output_dir = Path(config.get('output.checkpoint_dir', 'checkpoints'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Checkpoint Directory: {output_dir}")
    
    # Save configuration to output directory
    config_save_path = output_dir / "training_config.yaml"
    config.save(config_save_path)
    logger.info(f"  Saved config to: {config_save_path}")
    
    logger.info("")
    
    # Start MLflow run
    with mlflow.start_run():
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        # Enable PyTorch Lightning autologging
        mlflow.pytorch.autolog(log_models=True, log_every_n_epoch=1)
        
        # Log all configuration parameters
        logger.info("\nLogging parameters to MLflow...")
        mlflow_params = {
            'random_seed': seed,
            'data_dir': str(config.get('data.data_dir')),
            'image_size': config.get('data.image_size', 512),
            'batch_size': config.get('data.batch_size', 8),
            'num_workers': config.get('data.num_workers', 4),
            'val_split': config.get('data.val_split', 0.2),
            'encoder_name': config.get('model.encoder_name', 'mit_b5'),
            'encoder_weights': config.get('model.encoder_weights', 'imagenet'),
            'in_channels': config.get('model.in_channels', 3),
            'classes': config.get('model.classes', 1),
            'loss_fn': config.get('model.loss_fn', 'combined'),
            'learning_rate': config.get('training.learning_rate', 1e-4),
            'max_epochs': config.get('training.max_epochs', 100),
            'accelerator': config.get('hardware.accelerator', 'auto'),
            'devices': str(config.get('hardware.devices', 1)),
            'precision': config.get('hardware.precision', 32),
        }
        
        # Add pretrained info
        use_pretrained = config.get('pretrained.use_pretrained', True)
        mlflow_params['use_pretrained'] = use_pretrained
        if use_pretrained:
            mlflow_params['pretrained_weights'] = str(config.get('pretrained.weights_path', ''))
        
        # Log parameters
        mlflow.log_params(mlflow_params)
        logger.info("  ✓ Parameters logged")
        
        # Log config file as artifact
        mlflow.log_artifact(str(config_path), artifact_path="config")
        logger.info(f"  ✓ Config file logged: {config_path}")
        
        # Log dvc.lock if exists
        dvc_lock_path = Path("dvc.lock")
        if dvc_lock_path.exists():
            mlflow.log_artifact(str(dvc_lock_path), artifact_path="dvc")
            logger.info(f"  ✓ DVC lock file logged: {dvc_lock_path}")
        
        # Create dataset
        logger.info("Loading dataset...")
        data_dir = Path(config.get('data.data_dir'))
        image_size = config.get('data.image_size', 512)
        max_samples = config.get('data.max_samples')
        
        try:
            full_dataset = WatermarkDataset(
                data_dir=data_dir,
                image_size=image_size,
                transform=get_training_augmentation(image_size),
                create_masks=True,
                max_samples=max_samples,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        logger.info(f"Total samples: {len(full_dataset)}")
        
        # Split into train and validation
        val_split = config.get('data.val_split', 0.2)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        logger.info(f"Train samples: {train_size}")
        logger.info(f"Validation samples: {val_size}")
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = get_validation_augmentation(image_size)
        
        # Create data loaders
        batch_size = config.get('data.batch_size', 8)
        num_workers = config.get('data.num_workers', 4)
        pin_memory = config.get('data.pin_memory', True)
        
        logger.info(f"Creating data loaders (batch_size={batch_size}, num_workers={num_workers})...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Create model
        logger.info("\nInitializing model...")
        
        use_pretrained = config.get('pretrained.use_pretrained', True)
        pretrained_weights = config.get('pretrained.weights_path') if use_pretrained else None
        
        if pretrained_weights and Path(pretrained_weights).exists():
            logger.info(f"🎯 Finetuning from pretrained checkpoint: {pretrained_weights}")
        elif pretrained_weights:
            logger.warning(f"Pretrained weights not found: {pretrained_weights}")
            logger.info("Training from scratch with ImageNet initialization")
            pretrained_weights = None
        else:
            logger.info(f"🔨 Training from scratch with {config.get('model.encoder_weights', 'imagenet')} initialization")
        
        model = WatermarkDetectionModel(
            encoder_name=config.get('model.encoder_name', 'mit_b5'),
            encoder_weights=config.get('model.encoder_weights', 'imagenet'),
            learning_rate=config.get('training.learning_rate', 1e-4),
            loss_fn=config.get('model.loss_fn', 'combined'),
            pretrained_checkpoint=pretrained_weights,
        )
        
        # Setup callbacks
        logger.info("\nSetting up training callbacks...")
        
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=config.get('training.checkpoint.filename', 'watermark-{epoch:02d}-{val_iou:.4f}'),
            monitor=config.get('training.checkpoint.monitor', 'val_iou'),
            mode=config.get('training.checkpoint.mode', 'max'),
            save_top_k=config.get('training.checkpoint.save_top_k', 3),
            save_last=config.get('training.checkpoint.save_last', True),
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        logger.info("  ✓ Model checkpointing enabled")
        
        # Early stopping
        if config.get('training.early_stopping.enabled', True):
            early_stop_callback = EarlyStopping(
                monitor=config.get('training.early_stopping.monitor', 'val_iou'),
                patience=config.get('training.early_stopping.patience', 15),
                mode=config.get('training.early_stopping.mode', 'max'),
                verbose=True,
            )
            callbacks.append(early_stop_callback)
            logger.info(f"  ✓ Early stopping enabled (patience={config.get('training.early_stopping.patience', 15)})")
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        logger.info("  ✓ Learning rate monitoring enabled")
        
        # Logger
        logger.info("\nSetting up TensorBoard logging...")
        tb_logger = TensorBoardLogger(
            save_dir=config.get('output.tensorboard.save_dir', output_dir),
            name=config.get('output.tensorboard.name', 'watermark_detection'),
        )
        logger.info(f"  TensorBoard logs: {tb_logger.log_dir}")
        
        # Determine hardware configuration
        accelerator = config.get('hardware.accelerator', 'auto')
        devices = config.get('hardware.devices', 1)
        gpu_id = config.get('hardware.gpu_id')
        
        if gpu_id is not None:
            devices = [gpu_id]
            logger.info(f"  Hardware: Using GPU {gpu_id}")
        else:
            logger.info(f"  Hardware: {accelerator}, devices={devices}")
        
        # Create trainer
        logger.info("\nCreating PyTorch Lightning Trainer...")
        trainer = pl.Trainer(
            max_epochs=config.get('training.max_epochs', 100),
            accelerator=accelerator,
            devices=devices,
            precision=config.get('hardware.precision', 32),
            callbacks=callbacks,
            logger=tb_logger,
            log_every_n_steps=config.get('output.log_every_n_steps', 10),
            val_check_interval=config.get('output.val_check_interval', 1.0),
            deterministic=True,  # For reproducibility
        )
        
        # Train
        logger.info("\n" + "="*80)
        logger.info("Starting training...")
        logger.info("="*80 + "\n")
        
        try:
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            mlflow.log_param("status", "interrupted")
        except Exception as e:
            logger.error(f"\nTraining failed with error: {e}")
            mlflow.log_param("status", "failed")
            raise
        
        # Training summary
        logger.info("\n" + "="*80)
        logger.info("Training Complete!")
        logger.info("="*80)
        logger.info(f"Best model saved to: {checkpoint_callback.best_model_path}")
        logger.info(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")
        logger.info(f"Total epochs: {trainer.current_epoch + 1}")
        logger.info(f"TensorBoard logs: {tb_logger.log_dir}")
        
        # Log final metrics
        if checkpoint_callback.best_model_score is not None:
            mlflow.log_metric("best_val_iou", float(checkpoint_callback.best_model_score))
        mlflow.log_metric("total_epochs", trainer.current_epoch + 1)
        
        # Log best model checkpoint as artifact
        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(str(checkpoint_callback.best_model_path), artifact_path="checkpoints")
            logger.info(f"  ✓ Best model checkpoint logged: {checkpoint_callback.best_model_path}")
        
        # Log training config copy as artifact
        mlflow.log_artifact(str(config_save_path), artifact_path="config")
        
        # Log TensorBoard logs directory if exists
        if Path(tb_logger.log_dir).exists():
            mlflow.log_artifacts(str(tb_logger.log_dir), artifact_path="tensorboard")
            logger.info(f"  ✓ TensorBoard logs logged: {tb_logger.log_dir}")
        
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"MLflow UI: mlflow ui --backend-store-uri file:./mlruns")
        logger.info("="*80 + "\n")
        
        return checkpoint_callback.best_model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train watermark detection model from configuration file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    
    # Allow command-line overrides of common parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Specific GPU ID to use (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config",
    )
    
    args = parser.parse_args()
    
    # Build overrides dictionary
    overrides = {}
    if args.data_dir:
        overrides['data.data_dir'] = args.data_dir
    if args.batch_size:
        overrides['data.batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.max_epochs:
        overrides['training.max_epochs'] = args.max_epochs
    if args.gpu is not None:
        overrides['hardware.gpu_id'] = args.gpu
    if args.output_dir:
        overrides['output.checkpoint_dir'] = args.output_dir
    
    # Train
    train_from_config(
        config_path=args.config,
        verbose=args.verbose,
        **overrides
    )


if __name__ == "__main__":
    main()

