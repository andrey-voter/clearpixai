#!/bin/bash
# Example usage script for ClearPixAi

echo "=== ClearPixAi - Watermark Removal Example ==="
echo ""

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_image> [mode]"
    echo ""
    echo "Modes:"
    echo "  fast    - Fast mode (EasyOCR + OpenCV) [default]"
    echo "  quality - Quality mode (Florence-2 + Stable Diffusion)"
    echo ""
    echo "Examples:"
    echo "  $0 image.jpg"
    echo "  $0 image.jpg quality"
    exit 1
fi

INPUT_FILE="$1"
MODE="${2:-fast}"
OUTPUT_FILE="${INPUT_FILE%.*}_cleaned.${INPUT_FILE##*.}"

echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Mode:   $MODE"
echo ""

# Run watermark removal
if [ "$MODE" = "quality" ]; then
    uv run python run.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --quality --save-mask
else
    uv run python run.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --save-mask
fi

echo ""
echo "✓ Done! Check $OUTPUT_FILE for the result"

