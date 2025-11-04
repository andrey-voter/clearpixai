"""Dataset for watermark detection training.

This module provides a PyTorch Dataset implementation for watermark detection,
supporting automatic mask generation from watermarked/clean image pairs.
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
            error_msg = (
                f"No image pairs found in {data_dir}\n\n"
                f"Expected structures:\n\n"
                f"Structure 1 (Subdirectories):\n"
                f"  {data_dir}/\n"
                f"  ├── clean/\n"
                f"  │   ├── clean-0000.JPG\n"
                f"  │   └── ...\n"
                f"  └── watermarked/\n"
                f"      ├── clean-0000_v1.JPG\n"
                f"      ├── clean-0000_v2.JPG\n"
                f"      └── ...\n\n"
                f"Structure 2 (Flat directory):\n"
                f"  - watermark-0000.jpg / clean-0000.jpg\n"
                f"  - image0.jpg / image0 clean.jpg\n"
                f"  - photo_watermarked.jpg / photo_clean.jpg"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Found {len(self.image_pairs)} image pairs in {data_dir}")
        # Show first few pairs for verification
        for i, pair in enumerate(self.image_pairs[:3]):
            logger.debug(f"  Pair {i+1}: {pair['watermarked'].name} → {pair['clean'].name}")
        if len(self.image_pairs) > 3:
            logger.info(f"  ... and {len(self.image_pairs) - 3} more pairs")
        
        # Limit to max_samples if specified
        total_available = len(self.image_pairs)
        if self.max_samples is not None and self.max_samples < total_available:
            self.image_pairs = self.image_pairs[:self.max_samples]
            logger.warning(
                f"Using {len(self.image_pairs)} of {total_available} image pairs "
                f"(max_samples={self.max_samples})"
            )
    
    def _find_image_pairs(self):
        """Find pairs of watermarked and clean images.
        
        Supports multiple dataset structures:
        
        Structure 1: Separate clean/ and watermarked/ subdirectories
            data/
            ├── clean/
            │   ├── clean-0000.JPG
            │   ├── clean-0001.JPG
            │   └── ...
            └── watermarked/
                ├── clean-0000_v1.JPG
                ├── clean-0000_v2.JPG  (multiple versions per clean image)
                └── ...
        
        Structure 2: Flat directory with naming patterns
            data/
            ├── watermark-0000.jpg / clean-0000.jpg
            ├── image0.jpg / image0 clean.jpg
            └── ...
        """
        pairs = []
        
        # Check for Structure 1: clean/ and watermarked/ subdirectories
        clean_dir = self.data_dir / "clean"
        watermarked_dir = self.data_dir / "watermarked"
        
        if clean_dir.exists() and watermarked_dir.exists():
            logger.debug("Found clean/ and watermarked/ subdirectories")
            
            # Find all clean images
            clean_images = {}
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for clean_file in sorted(clean_dir.glob(ext)):
                    base_name = clean_file.stem
                    clean_images[base_name] = clean_file
            
            logger.debug(f"Found {len(clean_images)} clean images")
            
            # Find all watermarked images and match to clean images
            watermarked_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for watermarked_file in sorted(watermarked_dir.glob(ext)):
                    # Extract base name (remove _vX suffix)
                    name = watermarked_file.stem
                    
                    # Try to match: clean-0000_v1 -> clean-0000
                    # Pattern: {base}_v{number}
                    if '_v' in name:
                        base_name = name.rsplit('_v', 1)[0]
                    else:
                        # Fallback: use the full name
                        base_name = name
                    
                    # Find corresponding clean image
                    if base_name in clean_images:
                        pairs.append({
                            'watermarked': watermarked_file,
                            'clean': clean_images[base_name]
                        })
                        watermarked_count += 1
            
            logger.debug(f"Found {watermarked_count} watermarked images")
            logger.debug(f"Created {len(pairs)} training pairs")
            
            if len(pairs) > 0:
                return pairs
            else:
                logger.warning("No pairs found in subdirectories, falling back to flat structure")
        
        # Structure 2: Flat directory with various naming patterns
        seen_watermarked = set()
        
        # Pattern A: Look for watermark-XXXX.jpg / clean-XXXX.jpg pairs
        for watermark_file in sorted(self.data_dir.glob("watermark-*.jpg")):
            suffix = watermark_file.stem.replace("watermark-", "")
            clean_file = self.data_dir / f"clean-{suffix}.jpg"
            if clean_file.exists():
                pairs.append({
                    'watermarked': watermark_file,
                    'clean': clean_file
                })
                seen_watermarked.add(watermark_file)
        
        # Pattern B & C: Look for any other image with corresponding clean version
        for file in sorted(self.data_dir.glob("*.jpg")):
            # Skip if already processed or if it's a clean image
            if file in seen_watermarked or "clean" in file.name.lower():
                continue
            
            # This is a watermarked image, look for corresponding clean image
            base_name = file.stem
            
            # Check multiple naming patterns
            clean_candidates = [
                # Pattern: image0 clean.jpg, image0_clean.jpg, image0-clean.jpg
                self.data_dir / f"{base_name} clean.jpg",
                self.data_dir / f"{base_name}_clean.jpg",
                self.data_dir / f"{base_name}-clean.jpg",
                # Pattern: replace "watermark" or "watermarked" with "clean"
                self.data_dir / f"{base_name.replace('watermark', 'clean')}.jpg",
                self.data_dir / f"{base_name.replace('watermarked', 'clean')}.jpg",
            ]
            
            # Also try .png extension
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                clean_candidates.extend([
                    self.data_dir / f"{base_name} clean{ext}",
                    self.data_dir / f"{base_name}_clean{ext}",
                    self.data_dir / f"{base_name}-clean{ext}",
                ])
            
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
        # Ensure images have the same dimensions
        if watermarked.shape != clean.shape:
            # Resize clean to match watermarked dimensions
            h, w = watermarked.shape[:2]
            clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_LANCZOS4)
            logger.debug(f"Resized clean image to match watermarked image size ({w}x{h})")
        
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

