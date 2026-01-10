#!/usr/bin/env python3
"""Prepare data stage for DVC pipeline.

This script validates the dataset and creates a flag file and metrics report.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clearpixai.training.detector.validate_data import validate_dataset

def main():
    """Main entry point."""
    data_dir = Path("clearpixai/training/detector/data/train")
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Validate dataset
    results = validate_dataset(data_dir)
    
    # Save metrics report
    report_path = Path("data_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Validation report saved to: {report_path}")
    
    # Create flag file
    flag_path = Path("data_prepared.flag")
    flag_path.touch()
    print(f"✓ Flag file created: {flag_path}")
    
    if not results.get('valid', False):
        print("❌ Dataset validation failed!")
        sys.exit(1)
    
    print("✅ Data preparation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

