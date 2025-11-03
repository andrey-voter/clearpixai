#!/bin/bash
# Automatically train on the GPU with most free memory

echo "===================================================="
echo "Finding Best GPU..."
echo "===================================================="
echo ""

# Find GPU with most free memory
best_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $best_gpu)

echo "✅ Selected GPU $best_gpu with ${free_mem} MiB free"
echo ""

# Determine batch size based on available memory
if [ "$free_mem" -gt 12000 ]; then
    batch_size=8
    image_size=512
    echo "📊 Using: batch_size=$batch_size, image_size=$image_size"
elif [ "$free_mem" -gt 8000 ]; then
    batch_size=4
    image_size=512
    echo "📊 Using: batch_size=$batch_size, image_size=$image_size"
elif [ "$free_mem" -gt 4000 ]; then
    batch_size=2
    image_size=384
    echo "📊 Using: batch_size=$batch_size, image_size=$image_size"
else
    batch_size=1
    image_size=256
    echo "⚠️  Low memory! Using: batch_size=$batch_size, image_size=$image_size"
fi

echo ""
echo "===================================================="
echo "Starting Training on GPU $best_gpu"
echo "===================================================="
echo ""

# Set which GPU to use
export CUDA_VISIBLE_DEVICES=$best_gpu

# Run training
uv run python train_kaggle.py \
    --data-dir clearpixai/training/detector/data/kaggle_watermark_data/WatermarkDataset \
    --pretrained-weights clearpixai/detection/best_watermark_model_mit_b5_best.pth \
    --batch-size $batch_size \
    --image-size $image_size \
    --output-dir checkpoints_kaggle \
    "$@"

