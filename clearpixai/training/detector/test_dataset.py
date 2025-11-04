"""Test script to verify dataset is working correctly."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .dataset import WatermarkDataset, get_training_augmentation


def visualize_samples(data_dir: str, num_samples: int = 3, save_dir: str = None):
    """Visualize dataset samples with masks.
    
    Args:
        data_dir: Directory containing training data
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations (optional)
    """
    print(f"\n{'='*60}")
    print(f"Testing WatermarkDataset")
    print(f"{'='*60}\n")
    
    # Create dataset without augmentation for visualization
    dataset = WatermarkDataset(
        data_dir=Path(data_dir),
        image_size=512,
        transform=None,  # No augmentation for testing
        create_masks=True,
    )
    
    print(f"\n✅ Dataset initialized successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    # Create save directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Saving visualizations to: {save_dir}")
    
    # Visualize samples
    num_samples = min(num_samples, len(dataset))
    
    for idx in range(num_samples):
        print(f"\n📊 Visualizing sample {idx + 1}/{num_samples}...")
        
        # Get sample
        watermarked, mask = dataset[idx]
        pair = dataset.image_pairs[idx]
        
        # Load clean image for comparison
        clean = cv2.imread(str(pair['clean']))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        
        # Resize for visualization
        h, w = watermarked.shape[:2]
        if h != 512 or w != 512:
            watermarked_vis = cv2.resize(watermarked, (512, 512))
            clean_vis = cv2.resize(clean, (512, 512))
            mask_vis = cv2.resize(mask, (512, 512))
        else:
            watermarked_vis = watermarked
            clean_vis = clean
            mask_vis = mask
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Watermarked image
        axes[0].imshow(watermarked_vis)
        axes[0].set_title(f"Watermarked\n{pair['watermarked'].name}", fontsize=10)
        axes[0].axis('off')
        
        # Clean image
        axes[1].imshow(clean_vis)
        axes[1].set_title(f"Clean\n{pair['clean'].name}", fontsize=10)
        axes[1].axis('off')
        
        # Generated mask
        axes[2].imshow(mask_vis, cmap='gray')
        axes[2].set_title(f"Generated Mask\n(white = watermark)", fontsize=10)
        axes[2].axis('off')
        
        # Overlay
        overlay = watermarked_vis.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = mask_vis * 255  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay\n(red = detected watermark)", fontsize=10)
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_dir:
            output_path = save_dir / f"sample_{idx:03d}.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"   ✅ Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print mask statistics
        mask_coverage = mask.mean() * 100
        print(f"   Mask coverage: {mask_coverage:.2f}% of image")
        if mask_coverage < 0.1:
            print(f"   ⚠️  Warning: Very small mask coverage! Check if images are different.")
        elif mask_coverage > 50:
            print(f"   ⚠️  Warning: Large mask coverage! Check if images are aligned.")
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset test completed successfully!")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test watermark dataset")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (if not specified, displays interactively)",
    )
    
    args = parser.parse_args()
    
    try:
        visualize_samples(
            data_dir=args.data_dir,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()

