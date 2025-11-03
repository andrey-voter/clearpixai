#!/bin/bash
# Check GPU memory usage

echo "===================================================="
echo "GPU Memory Status"
echo "===================================================="
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx name total used free; do
        echo "GPU $idx: $name"
        echo "  Total: ${total} MiB"
        echo "  Used:  ${used} MiB"
        echo "  Free:  ${free} MiB"
        percent=$(( 100 * used / total ))
        echo "  Usage: ${percent}%"
        echo ""
    done
    
    echo "===================================================="
    echo "Processes Using GPU"
    echo "===================================================="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    
    # Check free memory on GPU 0
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    echo "===================================================="
    echo "Recommendation"
    echo "===================================================="
    if [ "$free_mem" -lt 4000 ]; then
        echo "⚠️  WARNING: Only ${free_mem} MiB free on GPU 0"
        echo ""
        echo "Solutions:"
        echo "  1. Kill other GPU processes (if safe)"
        echo "  2. Reduce batch size: --batch-size 2 or --batch-size 1"
        echo "  3. Reduce image size: --image-size 256"
        echo "  4. Use CPU: --accelerator cpu (very slow!)"
        echo ""
        echo "Recommended command:"
        echo "  uv run python train_kaggle.py \\"
        echo "      --data-dir clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset \\"
        echo "      --pretrained-weights clearpixai/detection/best_watermark_model_mit_b5_best.pth \\"
        echo "      --batch-size 2 \\"
        echo "      --image-size 384"
    else
        echo "✅ ${free_mem} MiB available - should be enough!"
        echo ""
        echo "Recommended command:"
        echo "  uv run python train_kaggle.py \\"
        echo "      --data-dir clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset \\"
        echo "      --pretrained-weights clearpixai/detection/best_watermark_model_mit_b5_best.pth"
    fi
    echo "===================================================="
else
    echo "❌ nvidia-smi not found. Are you on a machine with NVIDIA GPU?"
fi

