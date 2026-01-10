#!/usr/bin/env python3
"""Create TorchServe model archive (.mar file).

This script packages the model, handler, and config files into a TorchServe archive.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def create_archive(
    model_path: str,
    handler_path: str,
    config_path: str,
    model_name: str = "mymodel",
    output_dir: str = "model-store",
    extra_files: list = None,
):
    """Create TorchServe model archive.
    
    Args:
        model_path: Path to TorchScript model (.pt file)
        handler_path: Path to handler.py
        config_path: Path to config.json
        model_name: Name of the model
        output_dir: Output directory for .mar file
        extra_files: List of extra files to include
    """
    model_path = Path(model_path)
    handler_path = Path(handler_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if not handler_path.exists():
        print(f"Error: Handler file not found: {handler_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Find torch-model-archiver command
    # Try PATH first, then check venv/bin directory
    archiver_cmd = shutil.which("torch-model-archiver")
    
    if archiver_cmd is None:
        # Try to find in venv/bin (common location for uv venv)
        venv_bin = Path(sys.prefix) / "bin" / "torch-model-archiver"
        if venv_bin.exists():
            archiver_cmd = str(venv_bin)
        else:
            print("Error: torch-model-archiver not found!")
            print("Install it with: uv add torch-model-archiver")
            print("or: pip install torchserve")
            sys.exit(1)
    
    archiver_cmd = [archiver_cmd]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare extra files list
    extra_files_list = [str(config_path)]
    if extra_files:
        for f in extra_files:
            if Path(f).exists():
                extra_files_list.append(str(f))
    
    # Build command
    cmd = archiver_cmd + [
        "--model-name", model_name,
        "--version", "1.0",
        "--serialized-file", str(model_path),
        "--handler", str(handler_path),
        "--extra-files", ",".join(extra_files_list),
        "--export-path", str(output_dir),
        "--force",  # Overwrite existing archive
    ]
    
    print(f"\nCreating TorchServe archive...")
    print(f"   Model: {model_path}")
    print(f"   Handler: {handler_path}")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_dir}/{model_name}.mar")
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run archiver
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating archive:")
        print(result.stderr)
        sys.exit(1)
    
    mar_path = output_dir / f"{model_name}.mar"
    if mar_path.exists():
        print(f"Archive created successfully: {mar_path}")
        print(f"   Size: {mar_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"Error: Archive file not found at {mar_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create TorchServe model archive (.mar file)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to TorchScript model (.pt file)",
    )
    parser.add_argument(
        "--handler",
        type=str,
        default="torchserve_handler.py",
        help="Path to handler.py (default: torchserve_handler.py)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to model config.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mymodel",
        help="Name of the model (default: mymodel)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="model-store",
        help="Output directory for .mar file (default: model-store)",
    )
    parser.add_argument(
        "--extra-files",
        type=str,
        nargs="*",
        help="Additional files to include in archive",
    )
    
    args = parser.parse_args()
    
    create_archive(
        model_path=args.model,
        handler_path=args.handler,
        config_path=args.config,
        model_name=args.model_name,
        output_dir=args.output_dir,
        extra_files=args.extra_files,
    )


if __name__ == "__main__":
    main()

