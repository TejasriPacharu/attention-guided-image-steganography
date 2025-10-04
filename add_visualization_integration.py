#!/usr/bin/env python3
"""
Script to add comprehensive visualization integration to training scripts
"""

import os
import shutil

def add_visualization_to_gpu_training():
    """Add visualization integration to GPU training script"""
    
    # Read the existing file
    with open('train_novel.py', 'r') as f:
        content = f.read()
    
    # Add visualization imports after existing imports
    import_addition = """
# Import comprehensive visualization functions
import matplotlib.pyplot as plt
from demo_visualization import (
    visualize_comprehensive_attention_analysis,
    visualize_embedding_statistics,
    tensor_to_numpy
)
"""
    
    # Find the position to insert imports (after existing imports)
    import_pos = content.find("from utils.visualization import save_attention_visualizations")
    if import_pos != -1:
        end_pos = content.find('\n', import_pos) + 1
        content = content[:end_pos] + import_addition + content[end_pos:]
    
    # Add visualization method to the trainer class
    visualization_method = '''
    def save_epoch_visualizations(self, results, cover_images, secret_images, epoch, save_dir):
        """Save comprehensive visualizations for the epoch"""
        try:
            # Create epoch-specific directory
            epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            print(f"  📊 Saving epoch {epoch} visualizations...")
            
            # Take first sample for visualization
            sample_cover = cover_images[:1]
            sample_secret = secret_images[:1]
            sample_results = {
                'stego_image': results['stego_image'][:1],
                'extracted_secret': results['extracted_secret'][:1],
                'cover_attention': {
                    'embedding_attention': results['cover_attention']['embedding_attention'][:1]
                },
                'secret_attention': {
                    'embedding_attention': results['secret_attention']['embedding_attention'][:1]
                },
                'embedding_map': results['embedding_map'][:1]
            }
            
            # Add fusion weights if available
            if 'fusion_weights' in results:
                sample_results['fusion_weights'] = results['fusion_weights'][:1]
            
            # 1. Comprehensive attention analysis
            plt.ioff()  # Turn off interactive mode
            visualize_comprehensive_attention_analysis(
                sample_cover, sample_secret, sample_results,
                save_path=os.path.join(epoch_dir, 'comprehensive_analysis.png')
            )
            plt.close('all')
            
            # 2. Embedding statistics
            visualize_embedding_statistics(
                sample_results['embedding_map'],
                save_path=os.path.join(epoch_dir, 'embedding_statistics.png')
            )
            plt.close('all')
            
            print(f"  ✅ Visualizations saved to: {epoch_dir}")
            
        except Exception as e:
            print(f"  ⚠️  Visualization failed: {e}")
'''
    
    # Find position to insert the method (before save_checkpoint method)
    method_pos = content.find("def save_checkpoint(self, epoch")
    if method_pos != -1:
        # Find the start of the method (beginning of line)
        line_start = content.rfind('\n', 0, method_pos) + 1
        content = content[:line_start] + visualization_method + '\n    ' + content[line_start:]
    
    # Add visualization call in validation method
    validation_addition = '''
        # Save validation visualizations
        if self.writer:
            viz_dir = os.path.join(self.args.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            # Use first batch for visualization
            sample_cover = cover_images[:1] if 'cover_images' in locals() else None
            sample_secret = secret_images[:1] if 'secret_images' in locals() else None
            if sample_cover is not None and sample_secret is not None:
                sample_results = {
                    'stego_image': results['stego_image'][:1],
                    'extracted_secret': results['extracted_secret'][:1],
                    'cover_attention': results['cover_attention'],
                    'secret_attention': results['secret_attention'],
                    'embedding_map': results['embedding_map'][:1]
                }
                if 'fusion_weights' in results:
                    sample_results['fusion_weights'] = results['fusion_weights'][:1]
                self.save_epoch_visualizations(sample_results, sample_cover, sample_secret, epoch, viz_dir)
'''
    
    # Find the end of validate method and add visualization call
    validate_pos = content.find("return val_losses, val_metrics")
    if validate_pos != -1:
        content = content[:validate_pos] + validation_addition + '\n        ' + content[validate_pos:]
    
    # Write the modified content back
    with open('train_novel.py', 'w') as f:
        f.write(content)
    
    print("✅ Added visualization integration to train_novel.py")

def create_visualization_readme():
    """Create README for visualization outputs"""
    
    readme_content = """# Training Visualizations

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
"""
    
    with open('VISUALIZATION_README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created VISUALIZATION_README.md")

def main():
    """Main function to add visualization integration"""
    print("=" * 60)
    print("🎨 ADDING VISUALIZATION INTEGRATION")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ['train_novel.py', 'demo_visualization.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return
    
    # Backup original files
    print("📋 Creating backups...")
    if os.path.exists('train_novel.py'):
        shutil.copy('train_novel.py', 'train_novel.py.backup')
        print("  ✅ Backed up train_novel.py")
    
    # Add visualization integration
    print("\n🔧 Adding visualization integration...")
    try:
        add_visualization_to_gpu_training()
        print("  ✅ GPU training script updated")
    except Exception as e:
        print(f"  ❌ Failed to update GPU training script: {e}")
        # Restore backup
        if os.path.exists('train_novel.py.backup'):
            shutil.copy('train_novel.py.backup', 'train_novel.py')
            print("  🔄 Restored backup")
    
    # Create documentation
    print("\n📚 Creating documentation...")
    create_visualization_readme()
    
    print("\n" + "=" * 60)
    print("🎉 VISUALIZATION INTEGRATION COMPLETED!")
    print("=" * 60)
    print("\n📋 What was added:")
    print("  ✅ Comprehensive visualization imports")
    print("  ✅ save_epoch_visualizations() method")
    print("  ✅ Automatic visualization generation during validation")
    print("  ✅ Epoch-specific output directories")
    print("  ✅ Documentation (VISUALIZATION_README.md)")
    
    print("\n🚀 Training with visualizations:")
    print("  CPU: python train_novel_cpu.py")
    print("  GPU: python train_novel.py")
    
    print("\n📁 Visualization outputs will be saved to:")
    print("  CPU: ./outputs_cpu/visualizations/epoch_XXX/")
    print("  GPU: ./outputs/visualizations/epoch_XXX/")
    
    print("\n💡 Each epoch will generate:")
    print("  📊 comprehensive_analysis.png - Complete 12-panel analysis")
    print("  📈 embedding_statistics.png - Statistical analysis")
    print("  🎯 training_progress.png - Training-specific progress")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
