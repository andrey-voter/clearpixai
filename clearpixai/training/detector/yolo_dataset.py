"""Dataset loader for YOLO format watermark dataset (Kaggle).

This dataset has:
- images/train/ and images/val/ - images with watermarks
- labels/train/ and labels/val/ - YOLO format bounding boxes

YOLO format: class_id center_x center_y width height (all normalized 0-1)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class YOLOWatermarkDataset(Dataset):
    """Dataset for YOLO format watermark detection.
    
    The Kaggle dataset has images with watermarks and corresponding .txt files
    with bounding box annotations in YOLO format.
    
    Structure:
        images/train/ - training images
        labels/train/ - YOLO format annotations (.txt)
        images/val/ - validation images
        labels/val/ - validation annotations
    """
    
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 512,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Root directory (should contain images/ and labels/)
            image_size: Target image size
            transform: Albumentations transforms
            split: Dataset split ('train' or 'val')
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.split = split
        self.max_samples = max_samples
        
        # Paths to images and labels
        self.images_dir = self.data_dir / "images" / split
        self.labels_dir = self.data_dir / "labels" / split
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")
        
        # Find all image files
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Limit to max_samples if specified
        total_available = len(self.image_files)
        if self.max_samples is not None and self.max_samples < total_available:
            self.image_files = self.image_files[:self.max_samples]
            print(f"Using {len(self.image_files)} of {total_available} images from {self.images_dir}")
        else:
            print(f"Found {len(self.image_files)} images in {self.images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def _parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> np.ndarray:
        """Parse YOLO format label file and create binary mask.
        
        Args:
            label_path: Path to .txt label file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Binary mask (H, W) with 1 where watermarks are, 0 elsewhere
        """
        # Create empty mask
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        
        # Check if label file exists
        if not label_path.exists():
            # No watermark in this image
            return mask
        
        # Read YOLO annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Parse each bounding box
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # YOLO format: class_id center_x center_y width height (all normalized)
            class_id, center_x, center_y, box_width, box_height = map(float, parts)
            
            # Convert normalized coordinates to pixel coordinates
            center_x_px = center_x * img_width
            center_y_px = center_y * img_height
            box_width_px = box_width * img_width
            box_height_px = box_height * img_height
            
            # Calculate bounding box corners
            x1 = int(center_x_px - box_width_px / 2)
            y1 = int(center_y_px - box_height_px / 2)
            x2 = int(center_x_px + box_width_px / 2)
            y2 = int(center_y_px + box_height_px / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # Fill the bounding box region in the mask
            mask[y1:y2, x1:x2] = 1.0
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get image and mask pair.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, mask)
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Get corresponding label file
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # Create mask from YOLO annotations
        mask = self._parse_yolo_label(label_path, img_width, img_height)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def get_training_augmentation(image_size: int = 512) -> A.Compose:
    """Get augmentation pipeline for training.
    
    Based on Diffusion-Dynamics notebook augmentations.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=0.0625,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: int = 512) -> A.Compose:
    """Get augmentation pipeline for validation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

