"""Command line interface for ClearPixAi."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DetectionConfig,
    DiffusionConfig,
    MaskConfig,
    PipelineConfig,
    remove_watermark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ClearPixAi - AI-powered watermark removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Fast mode (OpenCV) with EasyOCR
  clearpixai -i input.jpg -o output.jpg

  # Quality mode with GroundingDINO + SAM detection
  clearpixai -i input.jpg -o output.jpg --quality --detector grounding_sam

  # Florence-2 detection with custom prompt
  clearpixai -i input.jpg -o output.jpg --detector florence2 --prompt "watermark; logo"
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")

    parser.add_argument(
        "--mode",
        choices=["fast", "quality"],
        default="fast",
        help="Inpainting mode (fast=OpenCV, quality=Stable Diffusion)",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Shortcut to set --mode quality",
    )

    parser.add_argument(
        "--detector",
        choices=["easyocr", "grounding_sam", "florence2", "segmentation"],
        default="easyocr",
        help="Primary detector to use",
    )
    parser.add_argument(
        "--fallback-detector",
        action="append",
        dest="fallback_detectors",
        default=[],
        help="Specify fallback detectors (can be provided multiple times)",
    )
    parser.add_argument(
        "--grounding-sam",
        action="store_true",
        help="Alias for --detector grounding_sam",
    )
    parser.add_argument(
        "--florence2",
        action="store_true",
        help="Alias for --detector florence2",
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Alias for --detector segmentation",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for detectors")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="EasyOCR languages (space separated)",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Directory to load/store detector weights",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES before execution",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (overrides --device)",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for Florence-2")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed applied to detection and diffusion unless overridden",
    )
    parser.add_argument(
        "--detection-seed",
        type=int,
        default=None,
        help="Seed for detector models (overrides --seed)",
    )
    parser.add_argument(
        "--diffusion-seed",
        type=int,
        default=None,
        help="Seed for Stable Diffusion inpainting (overrides --seed)",
    )
    parser.add_argument(
        "--blend-with-original",
        action="store_true",
        default=False,
        help="Blend diffusion output with the original crop to soften artifacts",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="GroundingDINO box threshold",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save the generated mask alongside the output",
    )
    parser.add_argument(
        "--segmentation-weights",
        type=str,
        default=None,
        help="Path to Diffusion Dynamics watermark-segmentation checkpoint",
    )
    parser.add_argument(
        "--segmentation-encoder",
        type=str,
        default="mit_b5",
        help="Encoder backbone name for segmentation-models-pytorch (e.g., mit_b5)",
    )
    parser.add_argument(
        "--segmentation-encoder-weights",
        type=str,
        default=None,
        help="Pretrained encoder weights identifier (e.g., imagenet)",
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for segmentation mask binarisation",
    )
    parser.add_argument(
        "--segmentation-image-size",
        type=int,
        default=None,
        help="Optional square resize dimension before segmentation inference",
    )
    parser.add_argument(
        "--segmentation-dilate",
        type=int,
        default=0,
        help="Post-process mask dilation kernel size (pixels)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.grounding_sam:
        args.detector = "grounding_sam"
    if args.florence2:
        args.detector = "florence2"
    if args.segmentation:
        args.detector = "segmentation"
    if args.quality:
        args.mode = "quality"
    if args.cpu:
        args.device = "cpu"

    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    )


def build_config(args: argparse.Namespace) -> PipelineConfig:
    detection = DetectionConfig()
    detection.detector = args.detector
    if args.fallback_detectors:
        detection.fallback_detectors = tuple(args.fallback_detectors)
    detection.prompt = args.prompt
    detection.weights_dir = Path(args.weights_dir)
    detection.device = args.device
    detection.hf_token = args.hf_token
    if args.languages:
        detection.easyocr_languages = tuple(args.languages)
    detection.box_threshold = args.box_threshold
    detection.text_threshold = args.text_threshold
    if args.segmentation_weights:
        detection.segmentation_weights = Path(args.segmentation_weights)
    detection.segmentation_encoder = args.segmentation_encoder
    detection.segmentation_encoder_weights = args.segmentation_encoder_weights
    detection.segmentation_threshold = args.segmentation_threshold
    detection.segmentation_image_size = args.segmentation_image_size
    detection.segmentation_dilate = args.segmentation_dilate

    mask = MaskConfig()

    diffusion = DiffusionConfig()
    diffusion.seed = args.diffusion_seed if args.diffusion_seed is not None else args.seed
    if args.blend_with_original:
        diffusion.blend_with_original = 0.25

    config = PipelineConfig(
        detection=detection,
        mask=mask,
        diffusion=diffusion,
        mode=args.mode,
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


