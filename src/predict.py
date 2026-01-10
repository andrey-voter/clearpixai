"""Docker entry point for ClearPixAI inference.

This script loads the model and performs watermark removal on input images.
It accepts command-line arguments --input_path and --output_path.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import clearpixai
sys.path.insert(0, str(Path(__file__).parent.parent))

from clearpixai.pipeline import PipelineConfig, remove_watermark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Docker container."""
    parser = argparse.ArgumentParser(
        description="ClearPixAI Docker inference - Remove watermarks from images"
    )
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to input image file (supports: jpg, jpeg, png, etc.)",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to save output cleaned image",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to segmentation model weights (default: uses built-in model)",
    )
    parser.add_argument(
        "--segmentation-weights",
        "-w",
        type=str,
        default=None,
        dest="model_path",  # Alias for model_path
        help="Path to segmentation model weights (alias for --model_path)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Validate input file exists
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration
    config = PipelineConfig()
    
    # Override model path if provided
    if args.model_path:
        config.segmentation.weights = Path(args.model_path)
        logger.info("Using custom model: %s", args.model_path)
    else:
        # Use default model path (relative to clearpixai package)
        default_model = Path(__file__).parent.parent / "clearpixai" / "detection" / "best_watermark_model_mit_b5_best.pth"
        config.segmentation.weights = default_model
        logger.info("Using default model: %s", default_model)

    # Validate model exists
    if not config.segmentation.weights.exists():
        logger.error("Model weights not found: %s", config.segmentation.weights)
        sys.exit(1)

    try:
        logger.info("Processing image: %s -> %s", input_path, output_path)
        remove_watermark(input_path, output_path, config)
        logger.info("Successfully processed image. Output saved to: %s", output_path)
    except Exception as e:
        logger.error("Error during processing: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

