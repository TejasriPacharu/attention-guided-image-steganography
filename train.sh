#!/bin/bash

# Attention-Guided Image Steganography Training Script
# This script starts the training process on GPU

echo "==================================="
echo "Attention-Guided Steganography Training"
echo "==================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Training will use CPU."
    echo ""
fi

# Set default parameters
DATA_DIR="./dataset"
BATCH_SIZE=8
NUM_EPOCHS=100
EMBEDDING_STRATEGY="adaptive"
OUTPUT_DIR="./outputs"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"

# Create necessary directories
echo "Creating directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' not found!"
    echo "Please create the dataset directory with train/val/test subdirectories."
    exit 1
fi

if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ] || [ ! -d "$DATA_DIR/test" ]; then
    echo "Error: Dataset should contain 'train', 'val', and 'test' subdirectories!"
    echo "Current structure:"
    ls -la $DATA_DIR
    exit 1
fi

# Count images in each split
train_count=$(find $DATA_DIR/train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
val_count=$(find $DATA_DIR/val -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
test_count=$(find $DATA_DIR/test -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)

echo "Dataset Statistics:"
echo "  Train images: $train_count"
echo "  Validation images: $val_count"
echo "  Test images: $test_count"
echo ""

if [ $train_count -eq 0 ]; then
    echo "Error: No images found in training directory!"
    exit 1
fi

if [ $val_count -eq 0 ]; then
    echo "Error: No images found in validation directory!"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --embedding_strategy)
            EMBEDDING_STRATEGY="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch_size SIZE        Batch size (default: 8)"
            echo "  --num_epochs EPOCHS      Number of epochs (default: 100)"
            echo "  --embedding_strategy STR Strategy: adaptive/high_low/low_high (default: adaptive)"
            echo "  --data_dir DIR           Dataset directory (default: ./dataset)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --batch_size 16 --num_epochs 50 --embedding_strategy adaptive"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Embedding Strategy: $EMBEDDING_STRATEGY"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"
echo "  Log Directory: $LOG_DIR"
echo ""

# Start training
echo "Starting training..."
echo "==================================="

python train.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --embedding_strategy $EMBEDDING_STRATEGY \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --num_workers 4 \
    --image_size 256 \
    --gen_lr 1e-4 \
    --disc_lr 1e-4 \
    --cover_loss_weight 1.0 \
    --secret_loss_weight 1.0 \
    --attention_loss_weight 0.1 \
    --perceptual_loss_weight 0.5 \
    --adversarial_loss_weight 0.1 \
    --log_interval 10 \
    --save_interval 5

# Check training result
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Training completed successfully!"
    echo "==================================="
    echo "Results saved in:"
    echo "  Checkpoints: $CHECKPOINT_DIR"
    echo "  Outputs: $OUTPUT_DIR"
    echo "  Logs: $LOG_DIR"
    echo ""
    echo "To monitor training progress:"
    echo "  tensorboard --logdir=$LOG_DIR"
    echo ""
    echo "To evaluate the model:"
    echo "  python evaluate.py --model_path $CHECKPOINT_DIR/best_model.pth --test_data_dir $DATA_DIR/test"
else
    echo ""
    echo "==================================="
    echo "Training failed!"
    echo "==================================="
    echo "Please check the error messages above."
    exit 1
fi
