"""Data validation script to check dataset quality and correctness.

This script validates the input data before training:
- Checks file formats and types
- Validates image dimensions and properties
- Checks for corrupted files
- Generates basic statistics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def validate_image(image_path: Path) -> Tuple[bool, Dict]:
    """Validate a single image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (is_valid, info_dict)
    """
    info = {
        'path': str(image_path),
        'valid': False,
        'error': None,
        'width': None,
        'height': None,
        'channels': None,
        'format': None,
        'size_bytes': None,
    }
    
    try:
        # Check file exists and has size
        if not image_path.exists():
            info['error'] = "File not found"
            return False, info
        
        file_size = image_path.stat().st_size
        info['size_bytes'] = file_size
        
        if file_size == 0:
            info['error'] = "Empty file"
            return False, info
        
        # Try to load with PIL
        with Image.open(image_path) as img:
            info['width'] = img.width
            info['height'] = img.height
            info['format'] = img.format
            info['channels'] = len(img.getbands())
        
        # Try to load with OpenCV as additional validation
        img_array = cv2.imread(str(image_path))
        if img_array is None:
            info['error'] = "Cannot load with OpenCV"
            return False, info
        
        # Check for reasonable dimensions
        if info['width'] < 32 or info['height'] < 32:
            info['error'] = f"Image too small: {info['width']}x{info['height']}"
            return False, info
        
        if info['width'] > 10000 or info['height'] > 10000:
            info['error'] = f"Image too large: {info['width']}x{info['height']}"
            return False, info
        
        info['valid'] = True
        return True, info
        
    except Exception as e:
        info['error'] = str(e)
        return False, info


def find_image_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Find image pairs in dataset.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        List of (watermarked_path, clean_path) tuples
    """
    pairs = []
    
    # Check for Structure 1: clean/ and watermarked/ subdirectories
    clean_dir = data_dir / "clean"
    watermarked_dir = data_dir / "watermarked"
    
    if clean_dir.exists() and watermarked_dir.exists():
        logger.info("Found clean/ and watermarked/ subdirectories")
        
        # Find all clean images
        clean_images = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for clean_file in sorted(clean_dir.glob(ext)):
                base_name = clean_file.stem
                clean_images[base_name] = clean_file
        
        # Find all watermarked images and match to clean images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for watermarked_file in sorted(watermarked_dir.glob(ext)):
                name = watermarked_file.stem
                
                # Try to match: clean-0000_v1 -> clean-0000
                if '_v' in name:
                    base_name = name.rsplit('_v', 1)[0]
                else:
                    base_name = name
                
                if base_name in clean_images:
                    pairs.append((watermarked_file, clean_images[base_name]))
    
    # Structure 2: Flat directory
    if len(pairs) == 0:
        logger.info("Checking flat directory structure")
        
        seen_watermarked = set()
        
        # Pattern: watermark-XXXX.jpg / clean-XXXX.jpg pairs
        for watermark_file in sorted(data_dir.glob("watermark-*.jpg")):
            suffix = watermark_file.stem.replace("watermark-", "")
            clean_file = data_dir / f"clean-{suffix}.jpg"
            if clean_file.exists():
                pairs.append((watermark_file, clean_file))
                seen_watermarked.add(watermark_file)
        
        # Look for other patterns
        for file in sorted(data_dir.glob("*.jpg")):
            if file in seen_watermarked or "clean" in file.name.lower():
                continue
            
            base_name = file.stem
            
            # Check multiple naming patterns
            clean_candidates = [
                data_dir / f"{base_name} clean.jpg",
                data_dir / f"{base_name}_clean.jpg",
                data_dir / f"{base_name}-clean.jpg",
            ]
            
            for clean_path in clean_candidates:
                if clean_path.exists():
                    pairs.append((file, clean_path))
                    break
    
    return pairs


def validate_dataset(data_dir: Path) -> Dict:
    """Validate entire dataset.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        Dictionary with validation results and statistics
    """
    logger.info(f"Validating dataset in: {data_dir}")
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return {'valid': False, 'error': 'Directory not found'}
    
    # Find image pairs
    logger.info("Finding image pairs...")
    pairs = find_image_pairs(data_dir)
    
    if len(pairs) == 0:
        logger.error("No image pairs found!")
        return {
            'valid': False,
            'error': 'No image pairs found',
            'num_pairs': 0,
        }
    
    logger.info(f"Found {len(pairs)} image pairs")
    
    # Validate each image
    logger.info("Validating images...")
    
    valid_pairs = []
    invalid_images = []
    
    all_widths = []
    all_heights = []
    all_sizes = []
    
    for watermarked_path, clean_path in tqdm(pairs, desc="Validating"):
        # Validate watermarked image
        wm_valid, wm_info = validate_image(watermarked_path)
        if not wm_valid:
            invalid_images.append(wm_info)
            logger.warning(f"Invalid watermarked image: {watermarked_path.name} - {wm_info['error']}")
            continue
        
        # Validate clean image
        clean_valid, clean_info = validate_image(clean_path)
        if not clean_valid:
            invalid_images.append(clean_info)
            logger.warning(f"Invalid clean image: {clean_path.name} - {clean_info['error']}")
            continue
        
        # Check if dimensions match
        if wm_info['width'] != clean_info['width'] or wm_info['height'] != clean_info['height']:
            logger.warning(
                f"Dimension mismatch: {watermarked_path.name} "
                f"({wm_info['width']}x{wm_info['height']}) != "
                f"{clean_path.name} ({clean_info['width']}x{clean_info['height']})"
            )
            # Still valid, will be resized during training
        
        valid_pairs.append((watermarked_path, clean_path, wm_info, clean_info))
        all_widths.append(wm_info['width'])
        all_heights.append(wm_info['height'])
        all_sizes.append(wm_info['size_bytes'])
    
    # Calculate statistics
    results = {
        'valid': True,
        'total_pairs': len(pairs),
        'valid_pairs': len(valid_pairs),
        'invalid_images': len(invalid_images),
        'statistics': {
            'width': {
                'min': int(np.min(all_widths)) if all_widths else 0,
                'max': int(np.max(all_widths)) if all_widths else 0,
                'mean': float(np.mean(all_widths)) if all_widths else 0,
                'median': float(np.median(all_widths)) if all_widths else 0,
            },
            'height': {
                'min': int(np.min(all_heights)) if all_heights else 0,
                'max': int(np.max(all_heights)) if all_heights else 0,
                'mean': float(np.mean(all_heights)) if all_heights else 0,
                'median': float(np.median(all_heights)) if all_heights else 0,
            },
            'file_size_mb': {
                'min': float(np.min(all_sizes)) / 1024 / 1024 if all_sizes else 0,
                'max': float(np.max(all_sizes)) / 1024 / 1024 if all_sizes else 0,
                'mean': float(np.mean(all_sizes)) / 1024 / 1024 if all_sizes else 0,
            }
        },
        'invalid_image_details': invalid_images,
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate watermark detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*80)
    logger.info("ClearPixAI Dataset Validation")
    logger.info("="*80)
    logger.info("")
    
    # Validate dataset
    results = validate_dataset(Path(args.data_dir))
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("Validation Results")
    logger.info("="*80)
    
    if not results['valid']:
        logger.error(f"❌ Dataset validation FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info(f"Total image pairs found: {results['total_pairs']}")
    logger.info(f"Valid image pairs: {results['valid_pairs']}")
    logger.info(f"Invalid images: {results['invalid_images']}")
    
    if results['valid_pairs'] == 0:
        logger.error("❌ No valid image pairs found!")
        sys.exit(1)
    
    if results['invalid_images'] > 0:
        logger.warning(f"⚠️  Found {results['invalid_images']} invalid images")
    
    logger.info("\nImage Statistics:")
    stats = results['statistics']
    logger.info(f"  Width:  {stats['width']['min']} - {stats['width']['max']} "
                f"(mean: {stats['width']['mean']:.0f}, median: {stats['width']['median']:.0f})")
    logger.info(f"  Height: {stats['height']['min']} - {stats['height']['max']} "
                f"(mean: {stats['height']['mean']:.0f}, median: {stats['height']['median']:.0f})")
    logger.info(f"  Size:   {stats['file_size_mb']['min']:.2f} - {stats['file_size_mb']['max']:.2f} MB "
                f"(mean: {stats['file_size_mb']['mean']:.2f} MB)")
    
    logger.info("\n" + "="*80)
    
    if results['valid_pairs'] >= 10:
        logger.info("✅ Dataset validation PASSED")
        logger.info(f"   Dataset is ready for training with {results['valid_pairs']} image pairs")
    else:
        logger.warning(f"⚠️  Dataset has only {results['valid_pairs']} valid pairs")
        logger.warning("   Recommended: At least 10 image pairs for meaningful training")
    
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

