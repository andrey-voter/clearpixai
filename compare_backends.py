#!/usr/bin/env python3
"""Compare SD 2.0 vs SDXL inpainting quality."""

import sys
import time
from pathlib import Path

from clearpixai.pipeline import (
    PipelineConfig,
    DiffusionConfig,
    SegmentationConfig,
    MaskConfig,
    remove_watermark,
)


def test_backend(backend: str, test_image: Path, output_dir: Path):
    """Test a specific inpainting backend."""
    output_image = output_dir / f"{backend}_{test_image.name}"
    
    print(f"\n{'='*60}")
    print(f"Testing {backend.upper()} backend")
    print(f"{'='*60}")
    
    config = PipelineConfig(
        segmentation=SegmentationConfig(device="cuda"),
        mask=MaskConfig(),
        diffusion=DiffusionConfig(
            backend=backend,
            num_inference_steps=30,  # Same steps for fair comparison
            seed=42,  # Same seed for reproducibility
        ),
        save_mask=True,
    )
    
    print(f"Input: {test_image}")
    print(f"Output: {output_image}")
    print(f"Steps: {config.diffusion.num_inference_steps}")
    print()
    
    try:
        start_time = time.time()
        result_path = remove_watermark(test_image, output_image, config)
        elapsed = time.time() - start_time
        
        print(f"✓ Success!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Result: {result_path}")
        return True, elapsed
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Compare all backends."""
    # Find a test image
    test_dir = Path(__file__).parent / "tests"
    test_images = list(test_dir.glob("image*.jpg"))
    
    if not test_images:
        print("No test images found in tests/ directory")
        return 1
    
    test_image = test_images[0]
    output_dir = Path("outputs") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("INPAINTING BACKEND COMPARISON")
    print("="*60)
    print(f"Test image: {test_image}")
    print(f"Output directory: {output_dir}")
    
    # Test both backends
    backends = ["sd", "sdxl"]
    results = {}
    
    for backend in backends:
        success, elapsed = test_backend(backend, test_image, output_dir)
        results[backend] = {"success": success, "time": elapsed}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for backend, result in results.items():
        status = "✓" if result["success"] else "✗"
        time_str = f"{result['time']:.2f}s" if result["success"] else "N/A"
        print(f"{status} {backend.upper():8s} - Time: {time_str}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Compare the output images to see the quality difference!")
    
    return 0 if all(r["success"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

