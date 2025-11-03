#!/usr/bin/env python3
"""Export PyTorch Lightning checkpoint to .pth file for inference."""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clearpixai.training.detector.model import WatermarkDetectionModel


def export_checkpoint(checkpoint_path: str, output_path: str = None):
    """Export checkpoint to .pth file.
    
    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Output .pth file path (auto-generated if None)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load Lightning checkpoint
        model = WatermarkDetectionModel.load_from_checkpoint(str(checkpoint_path))
        print("✅ Checkpoint loaded successfully")
        
        # Auto-generate output path if not provided
        if output_path is None:
            output_path = checkpoint_path.stem + ".pth"
        
        output_path = Path(output_path)
        
        # Export model state dict
        print(f"\n💾 Exporting to: {output_path}")
        torch.save(model.model.state_dict(), str(output_path))
        
        print(f"\n✅ Export complete!")
        print(f"\n📋 Next steps:")
        print(f"\n1. Test your model:")
        print(f"   uv run clearpixai --save-mask \\")
        print(f"       -i tests/image0.jpg \\")
        print(f"       -o outputs/result.jpg \\")
        print(f"       --segmentation-weights {output_path} \\")
        print(f"       --gpu 3")
        print(f"\n2. Use as default (optional):")
        print(f"   # Backup original")
        print(f"   cp clearpixai/detection/best_watermark_model_mit_b5_best.pth \\")
        print(f"      clearpixai/detection/best_watermark_model_mit_b5_best.pth.backup")
        print(f"   # Replace with your model")
        print(f"   cp {output_path} \\")
        print(f"      clearpixai/detection/best_watermark_model_mit_b5_best.pth")
        print()
        
    except Exception as e:
        print(f"\n❌ Error exporting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch Lightning checkpoint to .pth file for inference"
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to .ckpt checkpoint file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .pth file path (default: <checkpoint_name>.pth)",
    )
    
    args = parser.parse_args()
    
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

