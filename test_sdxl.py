#!/usr/bin/env python3
"""Test script for SDXL inpainting backend."""

import sys
from pathlib import Path

from clearpixai.pipeline import (
    PipelineConfig,
    DiffusionConfig,
    SegmentationConfig,
    MaskConfig,
    remove_watermark,
)

def main():
    """Run a test of SDXL inpainting."""
    # Find a test image
    test_dir = Path(__file__).parent / "tests"
    test_images = list(test_dir.glob("image*.jpg"))
    
    if not test_images:
        print("No test images found in tests/ directory")
        return 1
    
    test_image = test_images[0]
    output_image = Path("outputs") / f"sdxl_test_{test_image.name}"
    output_image.parent.mkdir(exist_ok=True)
    
    print(f"Testing SDXL inpainting on: {test_image}")
    print(f"Output will be saved to: {output_image}")
    print()
    
    # Configure with SDXL backend
    config = PipelineConfig(
        segmentation=SegmentationConfig(
            device="cuda",  # Will use the GPU set by CUDA_VISIBLE_DEVICES
        ),
        mask=MaskConfig(),
        diffusion=DiffusionConfig(
            backend="sdxl",
            num_inference_steps=30,  # Reduced for faster testing
            guidance_scale=8.0,
            seed=42,
        ),
        save_mask=True,
    )
    
    print("Configuration:")
    print(f"  Backend: {config.diffusion.backend}")
    print(f"  Model: {config.diffusion.model_id or 'auto'}")
    print(f"  Steps: {config.diffusion.num_inference_steps}")
    print(f"  Guidance: {config.diffusion.guidance_scale}")
    print()
    
    try:
        print("Starting watermark removal...")
        result_path = remove_watermark(test_image, output_image, config)
        print(f"\n✓ Success! Result saved to: {result_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

