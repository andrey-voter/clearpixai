#!/usr/bin/env python3
"""
ClearPixAi - AI-powered watermark removal tool
Based on ComfyUI workflow: https://comfyui.org/en/ai-powered-watermark-removal-workflow

Implements the complete workflow:
1. Detection: GroundingDINO + SAM or EasyOCR
2. Inpainting: Crop → KSampler (SD) → Stitch
"""
import argparse
import torch
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')


def detect_watermarks_easyocr(image, reader):
    """
    Use EasyOCR to detect text/watermarks (Simple method).
    """
    print("🔍 Detecting text/watermarks with EasyOCR...")
    img_array = np.array(image)
    results = reader.readtext(img_array)
    
    boxes = []
    detected_texts = []
    for detection in results:
        bbox, text, conf = detection
        boxes.append(bbox)
        detected_texts.append(f"{text} (conf: {conf:.2f})")
    
    print(f"✓ Found {len(boxes)} text regions:")
    for text in detected_texts:
        print(f"  - {text}")
    
    return boxes


def detect_watermarks_grounding_sam(image, text_prompt="watermark. logo. text. signature.", device="cuda"):
    """
    Use GroundingDINO + SAM for watermark detection (ComfyUI workflow Module 2).
    Better for detecting logos and image-based watermarks.
    """
    print("🔍 Detecting watermarks with GroundingDINO + SAM...")
    print(f"   Prompt: {text_prompt}")
    
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        from segment_anything import sam_model_registry, SamPredictor
        import supervision as sv
        
        # Load GroundingDINO
        print("   Loading GroundingDINO...")
        # Note: Model will be auto-downloaded on first run
        grounding_dino_model = load_model(
            model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="weights/groundingdino_swint_ogc.pth"
        )
        
        # Convert PIL image to format GroundingDINO expects
        img_array = np.array(image)
        
        # Detect with GroundingDINO
        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=img_array,
            caption=text_prompt,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )
        
        print(f"✓ GroundingDINO found {len(boxes)} regions")
        
        if len(boxes) == 0:
            return []
        
        # Load SAM for precise segmentation
        print("   Loading SAM for precise segmentation...")
        sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        
        # Set image for SAM
        sam_predictor.set_image(img_array)
        
        # Convert boxes to SAM format and get masks
        h, w = img_array.shape[:2]
        boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
        
        masks_list = []
        for box in boxes_xyxy:
            box_np = box.cpu().numpy()
            masks, _, _ = sam_predictor.predict(
                box=box_np,
                multimask_output=False
            )
            masks_list.append(masks[0])
        
        print(f"✓ SAM generated {len(masks_list)} precise masks")
        
        # Convert masks to bounding boxes format
        result_boxes = []
        for mask in masks_list:
            # Get bounding box from mask
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                result_boxes.append([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ])
        
        return result_boxes
        
    except Exception as e:
        print(f"⚠️  GroundingDINO+SAM failed: {e}")
        print("   Falling back to EasyOCR...")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        return detect_watermarks_easyocr(image, reader)


def create_mask_from_boxes(image_size, boxes, expand_ratio=0.15):
    """
    Create a binary mask from bounding boxes.
    """
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for box in boxes:
        # Get bounding rectangle
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Expand the box
        w = x_max - x_min
        h = y_max - y_min
        x_min = max(0, x_min - w * expand_ratio)
        y_min = max(0, y_min - h * expand_ratio)
        x_max = min(width, x_max + w * expand_ratio)
        y_max = min(height, y_max + h * expand_ratio)
        
        # Draw filled rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
    
    return mask


def grow_mask_with_blur(mask, expand_pixels=10, blur_radius=5):
    """
    Grow mask edges and blur for natural blending (ComfyUI workflow technique).
    """
    # Convert to numpy
    mask_np = np.array(mask)
    
    # Dilate to expand
    kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Convert back to PIL and blur
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask_pil


def inpaint_crop_stitch(image, mask, device="cuda", model_id="stabilityai/stable-diffusion-2-inpainting"):
    """
    Proper inpainting workflow (ComfyUI style):
    1. InpaintCrop: Crop the watermark region
    2. KSampler: Apply SD inpainting on cropped region
    3. InpaintStitch: Stitch back to original image
    """
    print("🎨 Applying Stable Diffusion inpainting (Crop → Sample → Stitch)...")
    
    from diffusers import AutoPipelineForInpainting
    
    # Find bounding box of mask
    mask_np = np.array(mask)
    coords = np.argwhere(mask_np > 0)
    
    if len(coords) == 0:
        print("   No mask found, returning original image")
        return image
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add padding for context
    padding = 32
    h, w = image.size[1], image.size[0]
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(h, y_max + padding)
    x_max = min(w, x_max + padding)
    
    print(f"   📐 Crop region: [{x_min}:{x_max}, {y_min}:{y_max}]")
    
    # Step 1: InpaintCrop - Crop the region
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
    
    print(f"   ✂️  Cropped size: {cropped_image.size}")
    
    # Step 2: KSampler - Load and apply SD inpainting
    print(f"   🎨 Loading model: {model_id}")
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing(1)
    else:
        pipe = pipe.to(device)
    
    # Inpaint the cropped region
    print(f"   🔬 Running KSampler (SD inpainting)...")
    prompt = "clean surface, natural background, seamless, high quality, detailed"
    negative_prompt = "watermark, text, logo, signature, writing, letters, words, blurry, distorted"
    
    inpainted_crop = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=cropped_image,
        mask_image=cropped_mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.99,
    ).images[0]
    
    # Step 3: InpaintStitch - Stitch back to original
    print(f"   🧩 Stitching back to original image...")
    result = image.copy()
    result.paste(inpainted_crop, (x_min, y_min))
    
    print(f"   ✓ Inpainting complete!")
    
    return result


def inpaint_opencv(image, mask):
    """
    Fast OpenCV inpainting (simple method).
    """
    print("🎨 Applying OpenCV inpainting...")
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask)
    
    result = cv2.inpaint(img_cv, mask_cv, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    return result_pil


def remove_watermark(
    input_path, 
    output_path, 
    mode="fast",
    detection_method="easyocr",
    device=None,
    save_mask=False,
    text_prompt=None
):
    """
    Remove watermark from image following ComfyUI workflow.
    
    Args:
        input_path: Path to input image
        output_path: Path to save cleaned image
        mode: 'fast' (OpenCV) or 'quality' (SD with crop+stitch)
        detection_method: 'easyocr' or 'grounding_sam'
        device: 'cuda' or 'cpu'
        save_mask: Save detection mask
        text_prompt: Text prompt for GroundingDINO (if using grounding_sam)
    """
    print(f"\n{'='*60}")
    print(f"🖼️  ClearPixAi - Watermark Removal")
    print(f"{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"Detection: {detection_method.upper()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Load image
    print("📂 Loading image...")
    image = Image.open(input_path).convert("RGB")
    print(f"   Size: {image.size[0]}x{image.size[1]}")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Step 1: Detect watermarks
    print()
    if detection_method == "grounding_sam":
        if text_prompt is None:
            text_prompt = "watermark. logo. text. signature."
        boxes = detect_watermarks_grounding_sam(image, text_prompt=text_prompt, device=device)
    else:  # easyocr
        import easyocr
        print("🔧 Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        boxes = detect_watermarks_easyocr(image, reader)
    
    if not boxes:
        print("\n⚠️  No watermarks detected!")
        print("💾 Saving original image...")
        image.save(output_path, quality=95)
        print(f"\n✅ Done! (No watermarks detected)")
        return
    
    # Step 2: Create mask
    print()
    print("🎭 Creating detection mask...")
    mask = create_mask_from_boxes(image.size, boxes)
    
    # Grow and blur mask for natural blending (ComfyUI technique)
    print("   Growing mask with blur for natural blending...")
    mask = grow_mask_with_blur(mask, expand_pixels=10, blur_radius=5)
    
    if save_mask:
        mask_path = output_path.rsplit('.', 1)[0] + '_mask.png'
        mask.save(mask_path)
        print(f"   Saved mask: {mask_path}")
    
    # Step 3: Inpaint
    print()
    if mode == "quality":
        # Use proper Crop → Sample → Stitch workflow
        result = inpaint_crop_stitch(image, mask, device=device)
    else:  # fast mode
        result = inpaint_opencv(image, mask)
    
    # Step 4: Save
    print()
    print(f"💾 Saving result...")
    result.save(output_path, quality=95)
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Check: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ClearPixAi - AI-powered watermark removal (ComfyUI workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode with EasyOCR
  %(prog)s -i image.jpg -o clean.jpg
  
  # Quality mode with GroundingDINO + SAM
  %(prog)s -i image.jpg -o clean.jpg --quality --grounding-sam
  
  # Custom prompt for GroundingDINO
  %(prog)s -i image.jpg -o clean.jpg --quality --grounding-sam --prompt "logo. watermark."
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    
    parser.add_argument(
        "--mode",
        choices=["fast", "quality"],
        default="fast",
        help="'fast' (OpenCV) or 'quality' (SD Crop+Stitch)"
    )
    
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use quality mode (SD with Crop→Sample→Stitch)"
    )
    
    parser.add_argument(
        "--detection",
        choices=["easyocr", "grounding_sam"],
        default="easyocr",
        help="Detection method: 'easyocr' (text) or 'grounding_sam' (logos/images)"
    )
    
    parser.add_argument(
        "--grounding-sam",
        action="store_true",
        help="Use GroundingDINO + SAM for detection (better for logos)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for GroundingDINO (e.g., 'watermark. logo. text.')"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Select specific GPU (e.g., '0' or '6')"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode"
    )
    
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save detection mask for debugging"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    mode = "quality" if args.quality else args.mode
    
    # Determine detection method
    detection = "grounding_sam" if args.grounding_sam else args.detection
    
    # Determine device
    if args.cpu:
        device = "cpu"
    elif args.device:
        device = args.device
    else:
        device = None
    
    # Set GPU if specified
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"🎮 Using GPU: {args.gpu}")
    
    remove_watermark(
        input_path=args.input,
        output_path=args.output,
        mode=mode,
        detection_method=detection,
        device=device,
        save_mask=args.save_mask,
        text_prompt=args.prompt
    )


if __name__ == "__main__":
    main()
