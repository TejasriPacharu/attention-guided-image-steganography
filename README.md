# Dynamic Embedding using Attention Heatmaps in Image Steganography

This repository contains the implementation of **"Dynamic Embedding using Attention Heatmaps in Image Steganography"**, a novel approach that extends the CAISFormer architecture by incorporating attention heatmaps of both cover and secret images to guide the embedding process.

## Overview

Traditional steganography methods often embed secret information uniformly across the cover image, which can lead to detectable artifacts. Our approach leverages attention mechanisms to identify optimal embedding regions by analyzing both cover and secret images, resulting in:

- **Better concealment** by avoiding perceptually important regions
- **Reduced perceptibility** through intelligent embedding strategies
- **Improved security** against steganalysis attacks
- **Higher quality** secret image recovery

## Architecture

The system consists of three main components:

1. **Attention Heatmap Generator**: Generates comprehensive attention maps for both cover and secret images using multiple attention mechanisms (spatial, channel, edge-based, texture-based)

2. **Embedding Network**: Uses CAISFormer modules (CSAM, CCAM, GLAM) with attention-guided embedding strategies to hide secret information in optimal regions

3. **Extraction Network**: Recovers the secret image using attention-guided extraction with similar architectural components

![Architecture Diagram](docs/architecture.png)

##  Key Features

- **Multiple Attention Mechanisms**: Spatial, channel, edge-based, and texture-based attention
- **Three Embedding Strategies**:
  - `adaptive`: Combines texture complexity and edge information
  - `high_low`: High attention areas in cover, low attention areas in secret
  - `low_high`: Low attention areas in cover, high attention areas in secret
- **CAISFormer Integration**: CSAM, CCAM, and GLAM modules for enhanced feature processing
- **Comprehensive Evaluation**: Security, robustness, and quality metrics
- **Interactive Demo**: Easy-to-use interface for testing different strategies

##  Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
scikit-image>=0.18.0
tensorboard>=2.7.0
tqdm>=4.62.0
lpips>=0.1.4
pytorch-msssim>=0.2.1
```

##  Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd attention-guided-image-steganography
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
```bash
# Option 1: Use existing dataset structure
mkdir -p data/train data/val data/test

# Option 2: Split existing dataset
python utils/dataset.py --data_dir /path/to/images --split
```

##  Quick Start

### Demo Usage

1. **Create Architecture Diagram**:
```bash
python demo.py --mode architecture --output_dir ./results
```

2. **Embed Secret Image**:
```bash
python demo.py --mode embed \
    --cover_image path/to/cover.jpg \
    --secret_image path/to/secret.jpg \
    --strategy adaptive \
    --output_dir ./results
```

3. **Extract Secret Image**:
```bash
python demo.py --mode extract \
    --stego_image path/to/stego.jpg \
    --output_dir ./results
```

4. **Compare Strategies**:
```bash
python demo.py --mode compare \
    --cover_image path/to/cover.jpg \
    --secret_image path/to/secret.jpg \
    --output_dir ./results
```

### Training

```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --num_epochs 100 \
    --embedding_strategy adaptive \
    --output_dir ./outputs \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --test_data_dir ./data/test \
    --output_dir ./evaluation_results
```



##  Configuration

### Training Parameters

- `--embedding_strategy`: Choose from `adaptive`, `high_low`, `low_high`
- `--cover_loss_weight`: Weight for cover image preservation (default: 1.0)
- `--secret_loss_weight`: Weight for secret image reconstruction (default: 1.0)
- `--attention_loss_weight`: Weight for attention consistency (default: 0.1)
- `--adversarial_loss_weight`: Weight for adversarial loss (default: 0.1)

### Model Architecture

- `--input_channels`: Number of input channels (default: 3)
- `--hidden_channels`: Hidden layer channels (default: 64)
- CAISFormer blocks with configurable attention heads

##  Project Structure

```
attention-guided-image-steganography/
├── models/
│   ├── __init__.py
│   ├── attention_heatmap.py          # Attention heatmap generation
│   ├── caisformer.py                 # CAISFormer modules (CSAM, CCAM, GLAM)
│   ├── attention_guided_steganography.py  # Main model
│   └── discriminator.py              # SRNet discriminator
├── utils/
│   ├── __init__.py
│   ├── dataset.py                    # Dataset handling
│   ├── metrics.py                    # Evaluation metrics
│   └── visualization.py             # Visualization utilities
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
├── demo.py                          # Interactive demo
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

## Attention Strategies

### 1. Adaptive Strategy
Combines multiple attention cues:
- Texture complexity from cover image
- Edge information from secret image
- Weighted combination for optimal embedding

### 2. High-Low Strategy
- Embeds in high-attention regions of cover image
- Uses low-attention regions of secret image
- Good for preserving secret image quality

### 3. Low-High Strategy
- Embeds in low-attention regions of cover image
- Uses high-attention regions of secret image
- Better for cover image preservation

## Evaluation Metrics

### Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Security Metrics
- **Detection Rate**: Percentage of stego images detected
- **False Positive Rate**: Percentage of cover images misclassified
- **Security Score**: Overall security assessment

### Robustness Metrics
- **JPEG Compression**: Quality preservation under compression
- **Gaussian Noise**: Resistance to additive noise
- **Gaussian Blur**: Performance under blurring
- **Resize Operations**: Robustness to scaling

##  Visualization Features

- **Attention Heatmaps**: Visualize attention patterns
- **Embedding Maps**: Show where secrets are embedded
- **Strategy Comparison**: Side-by-side comparison of methods
- **Training Curves**: Monitor training progress
- **Architecture Diagrams**: System overview

##  Research Applications

This implementation is designed for research in:
- **Image Steganography**: Novel embedding techniques
- **Attention Mechanisms**: Visual attention in computer vision
- **Adversarial Training**: GAN-based steganography
- **Security Analysis**: Steganalysis resistance


## Acknowledgments

- CAISFormer architecture inspiration
- SRNet for steganalysis evaluation
- PyTorch community for excellent deep learning framework
- Research community for steganography advances
