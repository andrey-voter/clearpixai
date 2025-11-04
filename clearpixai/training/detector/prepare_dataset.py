"""Helper script to prepare dataset from zip file or organize existing data."""

import argparse
import shutil
import zipfile
from pathlib import Path


def extract_and_organize(zip_path: str, output_dir: str = None):
    """Extract zip file and organize into clean/ and watermarked/ structure.
    
    Args:
        zip_path: Path to zip file
        output_dir: Output directory (default: same as zip file location)
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    if output_dir is None:
        output_dir = zip_path.parent / "extracted_data"
    else:
        output_dir = Path(output_dir)
    
    print(f"📦 Extracting: {zip_path}")
    print(f"📂 Output directory: {output_dir}")
    
    # Create output directories
    clean_dir = output_dir / "clean"
    watermarked_dir = output_dir / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract zip
    temp_extract = output_dir / "_temp_extract"
    temp_extract.mkdir(exist_ok=True)
    
    print(f"\n⏳ Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
    
    # Find and organize images
    print(f"\n📋 Organizing images...")
    
    clean_count = 0
    watermarked_count = 0
    
    # Recursively find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = []
    for ext in image_extensions:
        all_images.extend(temp_extract.rglob(f'*{ext}'))
    
    for img_file in sorted(all_images):
        name = img_file.stem
        
        # Determine if it's clean or watermarked based on naming
        if '_v' in name:
            # This is a watermarked version (e.g., clean-0000_v1)
            shutil.copy2(img_file, watermarked_dir / img_file.name)
            watermarked_count += 1
        else:
            # This is a clean image
            shutil.copy2(img_file, clean_dir / img_file.name)
            clean_count += 1
    
    # Clean up temp directory
    print(f"\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_extract)
    
    print(f"\n✅ Done!")
    print(f"   Clean images: {clean_count}")
    print(f"   Watermarked images: {watermarked_count}")
    print(f"   Training pairs: {watermarked_count}")
    print(f"\n📁 Dataset ready at: {output_dir}")
    print(f"\nNext step:")
    print(f"  uv run python -m clearpixai.training.detector.test_dataset \\")
    print(f"      --data-dir {output_dir} \\")
    print(f"      --save-dir test_visualizations")


def organize_existing(input_dir: str, output_dir: str = None):
    """Organize existing images into clean/ and watermarked/ structure.
    
    Args:
        input_dir: Directory containing images to organize
        output_dir: Output directory (default: input_dir + "_organized")
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_organized"
    else:
        output_dir = Path(output_dir)
    
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    # Create output directories
    clean_dir = output_dir / "clean"
    watermarked_dir = output_dir / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Organizing images...")
    
    clean_count = 0
    watermarked_count = 0
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    for img_file in sorted(input_dir.iterdir()):
        if img_file.suffix not in image_extensions:
            continue
        
        name = img_file.stem
        
        # Determine if it's clean or watermarked based on naming
        if '_v' in name:
            # This is a watermarked version
            shutil.copy2(img_file, watermarked_dir / img_file.name)
            watermarked_count += 1
        else:
            # This is a clean image
            shutil.copy2(img_file, clean_dir / img_file.name)
            clean_count += 1
    
    print(f"\n✅ Done!")
    print(f"   Clean images: {clean_count}")
    print(f"   Watermarked images: {watermarked_count}")
    print(f"   Training pairs: {watermarked_count}")
    print(f"\n📁 Dataset ready at: {output_dir}")
    print(f"\nNext step:")
    print(f"  uv run python -m clearpixai.training.detector.test_dataset \\")
    print(f"      --data-dir {output_dir} \\")
    print(f"      --save-dir test_visualizations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract zip file
  python -m clearpixai.training.detector.prepare_dataset \\
      --zip data.zip \\
      --output my_dataset

  # Organize existing directory
  python -m clearpixai.training.detector.prepare_dataset \\
      --input existing_images/ \\
      --output organized_dataset
        """
    )
    
    parser.add_argument(
        "--zip",
        type=str,
        help="Path to zip file to extract",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to existing directory to organize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    
    args = parser.parse_args()
    
    if not args.zip and not args.input:
        parser.error("Either --zip or --input must be specified")
    
    if args.zip and args.input:
        parser.error("Specify either --zip or --input, not both")
    
    try:
        if args.zip:
            extract_and_organize(args.zip, args.output)
        else:
            organize_existing(args.input, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

