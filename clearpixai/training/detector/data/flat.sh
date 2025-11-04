#!/bin/bash

# Root dataset directory
SRC_DIR="dataset_low"
# Destination directory for all photos
DEST_DIR="dataset_flat"

# Create destination if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all image files (jpg, jpeg, png, etc.) recursively and move them
find "$SRC_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read -r file; do
    base=$(basename "$file")
    # Add a random suffix to avoid name collisions
    newname="${base%.*}_$(uuidgen | cut -c1-8).${base##*.}"
    mv "$file" "$DEST_DIR/$newname"
done

echo "✅ All photos moved to $DEST_DIR"
