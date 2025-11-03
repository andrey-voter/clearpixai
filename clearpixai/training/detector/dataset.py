"""Dataset for watermark detection training."""

import os
from pathlib import Path
from typing import Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class WatermarkDataset(Dataset):
    """Dataset for watermark detection.
    
    Expects pairs of images:
    - Original image with watermark
    - Clean image without watermark (mask will be generated from difference)
    """
    
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 512,
        transform: Optional[A.Compose] = None,
        create_masks: bool = True,
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing image pairs
            image_size: Target image size
            transform: Albumentations transforms
            create_masks: Whether to create masks from clean/watermarked pairs
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.create_masks = create_masks
        self.max_samples = max_samples
        
        # Find all image pairs
        self.image_pairs = self._find_image_pairs()
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No image pairs found in {data_dir}")
        
        # Limit to max_samples if specified
        total_available = len(self.image_pairs)
        if self.max_samples is not None and self.max_samples < total_available:
            self.image_pairs = self.image_pairs[:self.max_samples]
            print(f"Using {len(self.image_pairs)} of {total_available} image pairs")
    
    def _find_image_pairs(self):
        """Find pairs of watermarked and clean images."""
        pairs = []
        
        # Look for patterns like:
        # - image0.jpg / image0 clean.jpg
        # - image1.jpg / image1 clean.jpg
        for file in sorted(self.data_dir.glob("*.jpg")):
            if "clean" not in file.name.lower():
                # This is a watermarked image
                # Look for corresponding clean image
                base_name = file.stem
                clean_candidates = [
                    self.data_dir / f"{base_name} clean.jpg",
                    self.data_dir / f"{base_name}_clean.jpg",
                    self.data_dir / f"{base_name}-clean.jpg",
                ]
                
                for clean_path in clean_candidates:
                    if clean_path.exists():
                        pairs.append({
                            'watermarked': file,
                            'clean': clean_path
                        })
                        break
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def _create_mask(self, watermarked: np.ndarray, clean: np.ndarray) -> np.ndarray:
        """Create binary mask from watermarked and clean image pair.
        
        Args:
            watermarked: Image with watermark
            clean: Clean image
            
        Returns:
            Binary mask where watermark was present
        """
        # Convert to grayscale
        if len(watermarked.shape) == 3:
            watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_RGB2GRAY)
            clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        else:
            watermarked_gray = watermarked
            clean_gray = clean
        
        # Calculate absolute difference
        diff = cv2.absdiff(watermarked_gray, clean_gray)
        
        # Threshold to create binary mask
        # Use adaptive threshold to handle varying lighting
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get image and mask pair.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, mask)
        """
        pair = self.image_pairs[idx]
        
        # Load images
        watermarked = cv2.imread(str(pair['watermarked']))
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)
        
        clean = cv2.imread(str(pair['clean']))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        
        # Create mask from difference
        if self.create_masks:
            mask = self._create_mask(watermarked, clean)
        else:
            # If masks are provided separately, load them
            mask_path = pair.get('mask')
            if mask_path and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0
            else:
                raise ValueError(f"No mask found for {pair['watermarked']}")
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=watermarked, mask=mask)
            watermarked = augmented['image']
            mask = augmented['mask']
        
        return watermarked, mask


def get_training_augmentation(image_size: int = 512) -> A.Compose:
    """Get augmentation pipeline for training.
    
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
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
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

