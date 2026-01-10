#!/usr/bin/env python3
"""Export PyTorch model to TorchScript format for TorchServe.

This script loads a trained watermark detection model and exports it to TorchScript format (.pt).
The exported model can be used with TorchServe.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import segmentation_models_pytorch as smp


def load_state_dict(weights_path: Path):
    """Load state dict from checkpoint file."""
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    
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
    
    return cleaned_state_dict


def export_to_torchscript(
    weights_path: str,
    config_path: str,
    output_path: str = "model.pt",
    image_size: int = 512,
):
    """Export model to TorchScript format.
    
    Args:
        weights_path: Path to model weights (.pth file)
        config_path: Path to model config.json
        output_path: Output path for TorchScript model
        image_size: Input image size for tracing
    """
    weights_path = Path(weights_path)
    config_path = Path(config_path)
    output_path = Path(output_path)
    
    if not weights_path.exists():
        print(f"Error: Weights not found: {weights_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    encoder_name = config.get("encoder_name", "mit_b5")
    
    print(f"\nLoading model (encoder={encoder_name})...")
    
    # Create model
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # We'll load custom weights
        in_channels=3,
        classes=1,
    )
    
    # Load weights
    state_dict = load_state_dict(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
    
    model.eval()
    
    print(f"Tracing model with input size ({image_size}, {image_size})...")
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        print("Model traced successfully!")
    except Exception as e:
        print(f"Error tracing model: {e}")
        sys.exit(1)
    
    # Save traced model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    print(f"Model saved to: {output_path}")
    
    # Verify the saved model
    try:
        loaded_model = torch.jit.load(str(output_path))
        test_output = loaded_model(dummy_input)
        print(f"Verification successful! Output shape: {test_output.shape}")
    except Exception as e:
        print(f"Warning: Could not verify saved model: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export watermark detection model to TorchScript format"
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to model config.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model.pt",
        help="Output path for TorchScript model (default: model.pt)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size for tracing (default: 512)",
    )
    
    args = parser.parse_args()
    
    export_to_torchscript(
        weights_path=args.weights,
        config_path=args.config,
        output_path=args.output,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()

