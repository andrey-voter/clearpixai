"""Command line interface for ClearPixAi."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DiffusionConfig,
    MaskConfig,
    PipelineConfig,
    SegmentationConfig,
    remove_watermark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ClearPixAi - AI-powered watermark removal using segmentation and diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Basic usage (uses SDXL backend by default for best quality)
  clearpixai -i input.jpg -o output.jpg

  # Use legacy Stable Diffusion 2.0 (faster but lower quality)
  clearpixai -i input.jpg -o output.jpg --diffusion-backend sd

  # With custom weights
  clearpixai -i input.jpg -o output.jpg --segmentation-weights /path/to/model.pth

  # Override specific parameters
  clearpixai -i input.jpg -o output.jpg --threshold 0.3 --diffusion-steps 100

  # Use specific GPU
  clearpixai -i input.jpg -o output.jpg --gpu 2

Note: All default values are defined in pipeline.py config classes.
      CLI arguments only override when explicitly provided.
        """,
    )

    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    
    # Optional weights override
    parser.add_argument(
        "--segmentation-weights",
        "-w",
        default=None,
        help="Path to segmentation model checkpoint (default: clearpixai/detection/best_watermark_model_mit_b5_best.pth)",
    )
    # Segmentation options
    seg_group = parser.add_argument_group("Segmentation options")
    seg_group.add_argument(
        "--segmentation-encoder",
        default=None,
        help="Encoder backbone name",
    )
    seg_group.add_argument(
        "--segmentation-encoder-weights",
        default=None,
        help="Pretrained encoder weights identifier (e.g., imagenet)",
    )
    seg_group.add_argument(
        "--segmentation-threshold",
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for mask binarization",
    )
    seg_group.add_argument(
        "--segmentation-image-size",
        type=int,
        default=None,
        help="Optional square resize dimension before inference",
    )

    # Mask processing options
    mask_group = parser.add_argument_group("Mask processing options")
    mask_group.add_argument(
        "--mask-expand",
        type=float,
        default=None,
        help="Mask expansion ratio",
    )
    mask_group.add_argument(
        "--mask-dilate",
        type=int,
        default=None,
        help="Mask dilation kernel size in pixels",
    )
    mask_group.add_argument(
        "--mask-blur",
        type=int,
        default=None,
        help="Mask blur radius",
    )

    # Diffusion options
    diffusion_group = parser.add_argument_group("Diffusion inpainting options")
    diffusion_group.add_argument(
        "--diffusion-backend",
        choices=["sd", "sdxl"],
        default=None,
        help="Inpainting backend: 'sd' (Stable Diffusion 2.0) or 'sdxl' (SDXL, better quality, default)",
    )
    diffusion_group.add_argument(
        "--diffusion-model",
        default=None,
        help="Diffusion model ID (overrides default for chosen backend)",
    )
    diffusion_group.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Number of inference steps",
    )
    diffusion_group.add_argument(
        "--diffusion-guidance",
        type=float,
        default=None,
        help="Guidance scale",
    )
    diffusion_group.add_argument(
        "--diffusion-strength",
        type=float,
        default=None,
        help="Diffusion strength",
    )
    diffusion_group.add_argument(
        "--diffusion-prompt",
        default=None,
        help="Custom positive prompt for inpainting",
    )
    diffusion_group.add_argument(
        "--diffusion-negative-prompt",
        default=None,
        help="Custom negative prompt for inpainting",
    )
    diffusion_group.add_argument(
        "--blend-with-original",
        type=float,
        default=None,
        help="Blend ratio with original (0.0-1.0)",
    )

    # General options
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Computation device",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES (e.g., '0' or '0,1')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save the generated mask alongside the output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    )


def build_config(args: argparse.Namespace) -> PipelineConfig:
    # Start with defaults from dataclasses
    segmentation = SegmentationConfig()
    mask = MaskConfig()
    diffusion = DiffusionConfig()
    
    # Override only if explicitly provided
    if args.segmentation_weights is not None:
        segmentation.weights = Path(args.segmentation_weights)
    if args.segmentation_encoder is not None:
        segmentation.encoder = args.segmentation_encoder
    if args.segmentation_encoder_weights is not None:
        segmentation.encoder_weights = args.segmentation_encoder_weights
    if args.segmentation_threshold is not None:
        segmentation.threshold = args.segmentation_threshold
    if args.segmentation_image_size is not None:
        segmentation.image_size = args.segmentation_image_size
    if args.device is not None:
        segmentation.device = args.device
    if args.seed is not None:
        segmentation.seed = args.seed

    if args.mask_expand is not None:
        mask.expand_ratio = args.mask_expand
    if args.mask_dilate is not None:
        mask.dilate = args.mask_dilate
    if args.mask_blur is not None:
        mask.blur_radius = args.mask_blur

    if args.diffusion_backend is not None:
        diffusion.backend = args.diffusion_backend
    if args.diffusion_model is not None:
        diffusion.model_id = args.diffusion_model
    if args.diffusion_steps is not None:
        diffusion.num_inference_steps = args.diffusion_steps
    if args.diffusion_guidance is not None:
        diffusion.guidance_scale = args.diffusion_guidance
    if args.diffusion_strength is not None:
        diffusion.strength = args.diffusion_strength
    if args.blend_with_original is not None:
        diffusion.blend_with_original = args.blend_with_original
    if args.diffusion_prompt is not None:
        diffusion.prompt = args.diffusion_prompt
    if args.diffusion_negative_prompt is not None:
        diffusion.negative_prompt = args.diffusion_negative_prompt
    if args.seed is not None:
        diffusion.seed = args.seed

    config = PipelineConfig(
        segmentation=segmentation,
        mask=mask,
        diffusion=diffusion,
        save_mask=args.save_mask,
        seed=args.seed,
    )

    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    configure_logging(args.verbose)

    config = build_config(args)
    remove_watermark(Path(args.input), Path(args.output), config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
