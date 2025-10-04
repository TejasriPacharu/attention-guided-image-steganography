# Training Visualizations

This directory contains comprehensive visualizations generated during training.

## Directory Structure

```
outputs_cpu/visualizations/
├── epoch_000/
│   ├── comprehensive_analysis.png    # Complete 12-panel analysis
│   ├── embedding_statistics.png      # Statistical analysis
│   └── training_progress.png         # Training-specific progress
├── epoch_001/
│   └── ...
└── epoch_N/
    └── ...
```

## Visualization Types

### 1. Comprehensive Analysis (comprehensive_analysis.png)
- **Row 1**: Cover Image, Secret Image, Cover Attention, Secret Attention
- **Row 2**: Embedding Map, Embedding Regions, Strategy Fusion, Strength Distribution
- **Row 3**: Stego Image, Extracted Secret, Difference Map, Quality Metrics

### 2. Embedding Statistics (embedding_statistics.png)
- **Panel 1**: Histogram of embedding strength distribution
- **Panel 2**: Spatial distribution heatmap
- **Panel 3**: Cumulative distribution function
- **Panel 4**: Region analysis (High/Medium/Low embedding)

### 3. Training Progress (training_progress.png)
- **Row 1**: Attention maps evolution
- **Row 2**: Results and embedding distribution

## Color Coding

### Attention Maps
- 🔥 **Hot colormap**: Black (low attention) → Red → Yellow → White (high attention)

### Embedding Maps  
- 🟢 **Viridis colormap**: Purple (low embedding) → Blue → Green → Yellow (high embedding)

### Strategy Fusion
- 🔴 **Red**: Texture Synthesis dominant
- 🟢 **Green**: Neural Codec dominant
- 🔵 **Blue**: Adversarial Embedding dominant

### Embedding Regions
- 🔴 **Red contours**: High embedding (>0.7)
- 🟡 **Yellow contours**: Medium embedding (0.3-0.7)
- 🔵 **Blue contours**: Low embedding (≤0.3)

## Interpretation Guide

### Good Training Signs
- ✅ Balanced embedding distribution (not too concentrated)
- ✅ High PSNR values (>30dB)
- ✅ Clear attention patterns
- ✅ Minimal difference maps
- ✅ Good secret reconstruction

### Warning Signs
- ⚠️ Very concentrated embedding (all high or all low)
- ⚠️ Low PSNR values (<25dB)
- ⚠️ Noisy attention patterns
- ⚠️ Large difference maps
- ⚠️ Poor secret reconstruction

## Usage

These visualizations are automatically generated during training when using:
- `train_novel_cpu.py` (every epoch)
- `train_novel.py` (every validation interval)

To generate standalone visualizations:
```bash
python demo_visualization.py
```
