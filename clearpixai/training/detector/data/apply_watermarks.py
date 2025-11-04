#!/usr/bin/env python3
"""
Apply all watermarks to photos in a directory with improvements.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

# Configuration
CLEAN_DIR = "clean"
WATERMARKED_DIR = "watermarked"
WATERMARKS_DIR = "watermarks"
BLACK_BG_DIR = os.path.join(WATERMARKS_DIR, "black_background")
WHITE_BG_DIR = os.path.join(WATERMARKS_DIR, "white_background")

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def get_image_files(directory):
    """Get all image files from a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    return [f for f in path.iterdir() 
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

def remove_background(watermark, bg_type='white'):
    """
    Remove background from watermark based on background type.
    
    Args:
        watermark: PIL Image object
        bg_type: 'white' for white/light backgrounds or 'black' for black/dark backgrounds
    """
    wm_data = watermark.getdata()
    new_data = []
    
    if bg_type == 'white':
        # Remove white/light backgrounds (for logos on white background)
        for item in wm_data:
            # If pixel is white or near-white (RGB > 240), make it transparent
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
    else:  # black background
        # Remove black/dark backgrounds (for logos on black background)
        for item in wm_data:
            # If pixel is black or near-black (RGB < 15), make it transparent
            if item[0] < 15 and item[1] < 15 and item[2] < 15:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append(item)
    
    watermark.putdata(new_data)
    return watermark

def apply_watermark(image_path, watermark_path, bg_type, output_path, position=None, opacity=0.8):
    """
    Apply a watermark to an image, with optional text watermark.
    
    Args:
        image_path: Path to the original image
        watermark_path: Path to the watermark image
        bg_type: 'white' or 'black' - type of background to remove
        output_path: Path to save the watermarked image
        position: Where to place watermark ('bottom-right', 'bottom-left', 'top-right', 'top-left', 'center')
        opacity: Watermark opacity (0.0 to 1.0)
    """
    # Open images
    base_image = Image.open(image_path).convert('RGBA')
    watermark = Image.open(watermark_path).convert('RGBA')
    
    # Remove background from watermark
    watermark = remove_background(watermark, bg_type)
    
    # Calculate watermark size (random 30-100% of image width)
    img_width, img_height = base_image.size
    scale = random.uniform(0.3, 1.0)
    wm_width = int(img_width * scale)
    wm_aspect = watermark.size[1] / watermark.size[0]
    wm_height = int(wm_width * wm_aspect)
    
    # Resize watermark
    watermark = watermark.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
    
    # Adjust watermark opacity
    alpha = watermark.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    watermark.putalpha(alpha)
    
    # Choose random position if not specified
    if position is None:
        position = random.choice(['bottom-right', 'bottom-left', 'top-right', 'top-left', 'center'])
    
    # Calculate position
    margin = 20
    if position == 'bottom-right':
        pos = (img_width - wm_width - margin, img_height - wm_height - margin)
    elif position == 'bottom-left':
        pos = (margin, img_height - wm_height - margin)
    elif position == 'top-right':
        pos = (img_width - wm_width - margin, margin)
    elif position == 'top-left':
        pos = (margin, margin)
    elif position == 'center':
        pos = ((img_width - wm_width) // 2, (img_height - wm_height) // 2)
    else:
        pos = (img_width - wm_width - margin, img_height - wm_height - margin)
    
    # Create a transparent layer for watermark
    transparent = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    transparent.paste(watermark, pos, watermark)
    
    # Composite main watermark
    watermarked = Image.alpha_composite(base_image, transparent)
    
    # With 50% probability, add text watermark
    if random.random() < 0.5:
        text = str(random.randint(1000, 9999))
        font = ImageFont.load_default(size=40)
        draw = ImageDraw.Draw(Image.new('RGBA', (1,1)))  # Dummy for size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        padding = 10
        text_width = (text_bbox[2] - text_bbox[0]) + 2 * padding
        text_height = (text_bbox[3] - text_bbox[1]) + 2 * padding
        
        wm_text = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 255))  # White bg
        draw = ImageDraw.Draw(wm_text)
        draw.text((padding, padding), text, font=font, fill=(0, 0, 0, 255))  # Black text
        
        # Adjust opacity
        alpha_text = wm_text.split()[3]
        alpha_text = alpha_text.point(lambda p: int(p * opacity))
        wm_text.putalpha(alpha_text)
        
        # Position: bottom-right
        pos_text = (img_width - text_width - margin, img_height - text_height - margin)
        
        # Create another transparent layer
        transparent_text = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
        transparent_text.paste(wm_text, pos_text, wm_text)
        
        # Composite text
        watermarked = Image.alpha_composite(watermarked, transparent_text)
    
    # Convert back to RGB if original was RGB
    original_mode = Image.open(image_path).mode
    if original_mode == 'RGB':
        watermarked = watermarked.convert('RGB')
    
    # Save with original quality
    watermarked.save(output_path, quality=95, optimize=True)

def main():
    """Main function to process all images."""
    # Create output directory if it doesn't exist
    os.makedirs(WATERMARKED_DIR, exist_ok=True)
    
    # Get all images
    clean_images = get_image_files(CLEAN_DIR)
    
    # Get watermarks from both directories
    black_bg_watermarks = [(f, 'black') for f in get_image_files(BLACK_BG_DIR)]
    white_bg_watermarks = [(f, 'white') for f in get_image_files(WHITE_BG_DIR)]
    
    # Combine all watermarks
    all_watermarks = black_bg_watermarks + white_bg_watermarks
    
    if not clean_images:
        print(f"No images found in '{CLEAN_DIR}' directory.")
        return
    
    if not all_watermarks:
        print(f"No watermarks found in '{BLACK_BG_DIR}' or '{WHITE_BG_DIR}' directories.")
        return
    
    print(f"Found {len(clean_images)} images.")
    print(f"Found {len(black_bg_watermarks)} watermarks with black background.")
    print(f"Found {len(white_bg_watermarks)} watermarks with white background.")
    print("Processing images...\n")
    
    # Process each image with all watermarks
    for i, image_path in enumerate(clean_images, 1):
        for v, (watermark_path, bg_type) in enumerate(all_watermarks, 1):
            # Create output path with version
            output_path = Path(WATERMARKED_DIR) / f"{image_path.stem}_v{v}{image_path.suffix}"
            
            try:
                # Apply watermark with random position
                apply_watermark(image_path, watermark_path, bg_type, output_path)
                bg_folder = "black_background" if bg_type == 'black' else "white_background"
                print(f"[{i}/{len(clean_images)}] v{v} ✓ {image_path.name} (watermark: {bg_folder}/{watermark_path.name})")
            except Exception as e:
                print(f"[{i}/{len(clean_images)}] v{v} ✗ {image_path.name} - Error: {e}")
    
    print(f"\nDone! Watermarked images saved to '{WATERMARKED_DIR}' directory.")

if __name__ == "__main__":
    main()