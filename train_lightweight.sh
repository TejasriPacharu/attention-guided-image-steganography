#!/bin/bash

# Lightweight Training Script for Memory-Constrained Environments
# This script uses optimized parameters to reduce memory usage

echo "==================================="
echo "Lightweight Attention-Guided Training"
echo "==================================="

# Check available memory
echo "System Memory Info:"
free -h | head -2

# Check GPU memory if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Info:"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Optimized parameters for memory efficiency
DATA_DIR="./dataset"
BATCH_SIZE=4                    # Reduced from 8
IMAGE_SIZE=128                  # Reduced from 256
HIDDEN_CHANNELS=32              # Reduced from 64
NUM_EPOCHS=50                   # Reduced for testing
EMBEDDING_STRATEGY="adaptive"
OUTPUT_DIR="./outputs_lite"
CHECKPOINT_DIR="./checkpoints_lite"
LOG_DIR="./logs_lite"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

echo "Lightweight Training Configuration:"
echo "  Batch Size: $BATCH_SIZE (memory optimized)"
echo "  Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE} (reduced resolution)"
echo "  Hidden Channels: $HIDDEN_CHANNELS (reduced model size)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Strategy: $EMBEDDING_STRATEGY"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --hidden_channels)
            HIDDEN_CHANNELS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Lightweight Training Options:"
            echo "  --batch_size SIZE        Batch size (default: 4)"
            echo "  --image_size SIZE        Image size (default: 128)"
            echo "  --hidden_channels SIZE   Hidden channels (default: 32)"
            echo "  --num_epochs EPOCHS      Number of epochs (default: 50)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting lightweight training..."
echo "==================================="

# Run training with memory-optimized parameters
python train.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --hidden_channels $HIDDEN_CHANNELS \
    --num_epochs $NUM_EPOCHS \
    --embedding_strategy $EMBEDDING_STRATEGY \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --num_workers 2 \
    --gen_lr 2e-4 \
    --disc_lr 2e-4 \
    --cover_loss_weight 1.0 \
    --secret_loss_weight 1.0 \
    --attention_loss_weight 0.05 \
    --perceptual_loss_weight 0.25 \
    --adversarial_loss_weight 0.05 \
    --log_interval 5 \
    --save_interval 10

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Lightweight training completed!"
    echo "==================================="
    echo "Results saved in:"
    echo "  Checkpoints: $CHECKPOINT_DIR"
    echo "  Logs: $LOG_DIR"
    echo ""
    echo "Monitor with: tensorboard --logdir=$LOG_DIR"
else
    echo ""
    echo "Training failed. Try even smaller parameters:"
    echo "  ./train_lightweight.sh --batch_size 2 --image_size 64"
fi
