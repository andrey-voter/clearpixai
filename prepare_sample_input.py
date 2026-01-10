#!/usr/bin/env python3
"""Prepare sample input JSON file for TorchServe testing.

This script reads an image file and creates a JSON file with base64-encoded image data.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from PIL import Image


def create_sample_input(image_path: str, output_path: str = "sample_input.json"):
    """Create sample input JSON file.
    
    Args:
        image_path: Path to input image
        output_path: Output JSON file path
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load and validate image
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Create JSON
    data = {
        "image": image_base64,
        "original_size": list(image.size)  # [width, height]
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Sample input created: {output_path}")
    print(f"   Image: {image_path} ({image.size[0]}x{image.size[1]})")
    print(f"   Base64 size: {len(image_base64)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create sample input JSON for TorchServe"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sample_input.json",
        help="Output JSON file path (default: sample_input.json)",
    )
    
    args = parser.parse_args()
    create_sample_input(args.image, args.output)

