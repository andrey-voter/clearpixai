"""Convert PyTorch Lightning checkpoint to .pth format for ClearPixAI."""

import argparse
from pathlib import Path

import torch

from .model import WatermarkDetectionModel


def convert_checkpoint(checkpoint_path: str, output_path: str = None):
    """Convert Lightning checkpoint to .pth format.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt)
        output_path: Path to save .pth file (optional, auto-generated if not provided)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the Lightning checkpoint
    model = WatermarkDetectionModel.load_from_checkpoint(str(checkpoint_path))
    
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.pth"
    else:
        output_path = Path(output_path)
    
    # Extract the underlying model state dict
    state_dict = model.model.state_dict()
    
    # Save to .pth format
    torch.save(state_dict, output_path)
    
    print(f"✅ Saved model to: {output_path}")
    print(f"\nYou can now use it with ClearPixAI:")
    print(f"\n  uv run clearpixai \\")
    print(f"      --save-mask \\")
    print(f"      -i tests/image2.jpg \\")
    print(f"      -o outputs/cleaned.jpg \\")
    print(f"      --segmentation-weights {output_path} \\")
    print(f"      --gpu 0")
    
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoint to .pth format"
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for .pth file (default: same name as checkpoint)",
    )
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint(args.checkpoint, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

