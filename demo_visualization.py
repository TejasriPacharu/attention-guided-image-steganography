#!/usr/bin/env python3
"""
Comprehensive Visualization Demo for Attention-Guided Steganography
Shows how embedding maps and attention heatmaps are visualized
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import os
import cv2

def create_sample_data():
    """Create sample data for visualization demo"""
    # Create sample images
    cover_image = torch.randn(1, 3, 64, 64)
    secret_image = torch.randn(1, 3, 64, 64)
    
    # Create sample attention maps
    cover_attention = torch.rand(1, 1, 64, 64)
    secret_attention = torch.rand(1, 1, 64, 64)
    
    # Create sample embedding map
    embedding_map = (cover_attention + secret_attention) / 2
    
    # Create sample results
    results = {
        'stego_image': cover_image + 0.1 * torch.randn_like(cover_image),
        'extracted_secret': secret_image + 0.05 * torch.randn_like(secret_image),
        'cover_attention': {'embedding_attention': cover_attention},
        'secret_attention': {'embedding_attention': secret_attention},
        'embedding_map': embedding_map,
        'fusion_weights': torch.softmax(torch.randn(1, 3, 64, 64), dim=1)
    }
    
    return cover_image, secret_image, results

def tensor_to_numpy(tensor, squeeze=False):
    """Convert tensor to numpy for visualization"""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    
    if squeeze:
        array = np.squeeze(array)
    
    # Handle different tensor formats
    if len(array.shape) == 4:  # [B, C, H, W]
        array = array[0]  # Take first batch
    
    if len(array.shape) == 3 and array.shape[0] in [1, 3]:  # [C, H, W]
        if array.shape[0] == 1:  # Grayscale
            array = array[0]
        else:  # RGB
            array = np.transpose(array, (1, 2, 0))
    
    # Normalize to [0, 1]
    if array.max() > 1.0 or array.min() < 0.0:
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    
    return array

def visualize_embedding_map_basic(embedding_map, save_path=None):
    """
    Basic embedding map visualization
    Shows the spatial distribution of embedding strength
    """
    print("Basic Embedding Map Visualization")
    
    # Convert to numpy
    embed_np = tensor_to_numpy(embedding_map, squeeze=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Raw embedding map
    im1 = axes[0].imshow(embed_np, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Embedding Map\n(Viridis Colormap)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Hot colormap (common for attention)
    im2 = axes[1].imshow(embed_np, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Embedding Map\n(Hot Colormap)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Custom colormap with contours
    im3 = axes[2].imshow(embed_np, cmap='plasma', vmin=0, vmax=1)
    contours = axes[2].contour(embed_np, levels=[0.2, 0.4, 0.6, 0.8], colors='white', linewidths=1)
    axes[2].clabel(contours, inline=True, fontsize=8, colors='white')
    axes[2].set_title('Embedding Map with Contours\n(Plasma Colormap)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    plt.show()

def visualize_attention_overlay(cover_image, embedding_map, save_path=None):
    """
    Visualize embedding map overlayed on cover image
    Shows WHERE the embedding will occur
    """
    print("Attention Overlay Visualization")
    
    # Convert to numpy
    cover_np = tensor_to_numpy(cover_image)
    embed_np = tensor_to_numpy(embedding_map, squeeze=True)
    
    # Create overlays with different alpha values
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    alphas = [0.3, 0.5, 0.7]
    colormaps = ['hot', 'viridis', 'plasma']
    
    for i, (alpha, cmap) in enumerate(zip(alphas, colormaps)):
        # Create colored heatmap
        cmap_obj = plt.get_cmap(cmap)
        attention_colored = cmap_obj(embed_np)[:, :, :3]  # Remove alpha channel
        
        # Overlay on original image
        overlayed = cover_np * (1 - alpha) + attention_colored * alpha
        
        # Top row: Original + overlay
        axes[0, i].imshow(overlayed)
        axes[0, i].set_title(f'Overlay (α={alpha}, {cmap})')
        axes[0, i].axis('off')
        
        # Bottom row: Just the attention map
        im = axes[1, i].imshow(embed_np, cmap=cmap, vmin=0, vmax=1)
        axes[1, i].set_title(f'Embedding Map ({cmap})')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    plt.show()

def visualize_comprehensive_attention_analysis(cover_image, secret_image, results, save_path=None):
    """
    Comprehensive visualization showing all attention components
    """
    print("Comprehensive Attention Analysis")
    
    # Extract data
    cover_img = tensor_to_numpy(cover_image)
    secret_img = tensor_to_numpy(secret_image)
    stego_img = tensor_to_numpy(results['stego_image'])
    extracted_img = tensor_to_numpy(results['extracted_secret'])
    
    cover_attention = tensor_to_numpy(results['cover_attention']['embedding_attention'], squeeze=True)
    secret_attention = tensor_to_numpy(results['secret_attention']['embedding_attention'], squeeze=True)
    embedding_map = tensor_to_numpy(results['embedding_map'], squeeze=True)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Original images and attention maps
    axes[0, 0].imshow(cover_img)
    axes[0, 0].set_title('Cover Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(secret_img)
    axes[0, 1].set_title('Secret Image')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(cover_attention, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Cover Attention')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 3].imshow(secret_attention, cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title('Secret Attention')
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Embedding analysis
    im3 = axes[1, 0].imshow(embedding_map, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Embedding Map')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Embedding map with regions
    embed_regions = axes[1, 1].imshow(embedding_map, cmap='viridis', vmin=0, vmax=1)
    # Add region boundaries
    high_embed = embedding_map > 0.7
    medium_embed = (embedding_map > 0.3) & (embedding_map <= 0.7)
    low_embed = embedding_map <= 0.3
    
    axes[1, 1].contour(high_embed.astype(float), levels=[0.5], colors='red', linewidths=2)
    axes[1, 1].contour(medium_embed.astype(float), levels=[0.5], colors='yellow', linewidths=2)
    axes[1, 1].contour(low_embed.astype(float), levels=[0.5], colors='blue', linewidths=2)
    axes[1, 1].set_title('Embedding Regions\n🔴High 🟡Medium 🔵Low')
    axes[1, 1].axis('off')
    
    # Strategy fusion weights (if available)
    if 'fusion_weights' in results:
        fusion_weights = tensor_to_numpy(results['fusion_weights'])
        
        # Show dominant strategy per pixel
        dominant_strategy = np.argmax(fusion_weights, axis=0)
        strategy_colors = ['red', 'green', 'blue']  # Texture, Codec, Adversarial
        strategy_names = ['Texture', 'Codec', 'Adversarial']
        
        # Create RGB image showing dominant strategy
        strategy_rgb = np.zeros((*dominant_strategy.shape, 3))
        for i in range(3):
            mask = dominant_strategy == i
            if i == 0:  # Red for texture
                strategy_rgb[mask] = [1, 0, 0]
            elif i == 1:  # Green for codec
                strategy_rgb[mask] = [0, 1, 0]
            else:  # Blue for adversarial
                strategy_rgb[mask] = [0, 0, 1]
        
        axes[1, 2].imshow(strategy_rgb)
        axes[1, 2].set_title('Dominant Strategy\n🔴Texture 🟢Codec 🔵Adversarial')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
    
    # Embedding strength histogram
    axes[1, 3].hist(embedding_map.flatten(), bins=50, alpha=0.7, color='viridis')
    axes[1, 3].set_title('Embedding Strength\nDistribution')
    axes[1, 3].set_xlabel('Embedding Strength')
    axes[1, 3].set_ylabel('Pixel Count')
    axes[1, 3].grid(True, alpha=0.3)
    
    # Row 3: Results and analysis
    axes[2, 0].imshow(stego_img)
    axes[2, 0].set_title('Stego Image')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(extracted_img)
    axes[2, 1].set_title('Extracted Secret')
    axes[2, 1].axis('off')
    
    # Difference map
    diff_map = np.abs(cover_img - stego_img)
    if len(diff_map.shape) == 3:
        diff_map = np.mean(diff_map, axis=2)  # Convert to grayscale
    
    im4 = axes[2, 2].imshow(diff_map, cmap='hot')
    axes[2, 2].set_title('Cover-Stego Difference')
    axes[2, 2].axis('off')
    plt.colorbar(im4, ax=axes[2, 2], fraction=0.046, pad=0.04)
    
    # Quality metrics visualization
    if len(cover_img.shape) == 3:
        cover_gray = np.mean(cover_img, axis=2)
        stego_gray = np.mean(stego_img, axis=2)
    else:
        cover_gray = cover_img
        stego_gray = stego_img
    
    # Simple PSNR calculation
    mse = np.mean((cover_gray - stego_gray) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    
    axes[2, 3].text(0.1, 0.8, f'Quality Metrics:', fontsize=12, fontweight='bold')
    axes[2, 3].text(0.1, 0.6, f'PSNR: {psnr:.2f} dB', fontsize=10)
    axes[2, 3].text(0.1, 0.4, f'Max Embed: {embedding_map.max():.3f}', fontsize=10)
    axes[2, 3].text(0.1, 0.2, f'Mean Embed: {embedding_map.mean():.3f}', fontsize=10)
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved to: {save_path}")
    plt.show()

def visualize_embedding_statistics(embedding_map, save_path=None):
    """
    Visualize statistical properties of embedding map
    """
    print("Embedding Map Statistics")
    
    embed_np = tensor_to_numpy(embedding_map, squeeze=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram
    axes[0, 0].hist(embed_np.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Embedding Strength Distribution')
    axes[0, 0].set_xlabel('Embedding Strength')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = embed_np.mean()
    std_val = embed_np.std()
    min_val = embed_np.min()
    max_val = embed_np.max()
    
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
    axes[0, 0].text(0.7, 0.8, stats_text, transform=axes[0, 0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 2D histogram (heatmap of spatial distribution)
    h, w = embed_np.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Flatten coordinates and values
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    values_flat = embed_np.flatten()
    
    # Create 2D histogram
    hist_2d, xedges, yedges = np.histogram2d(x_flat, y_flat, bins=20, weights=values_flat)
    
    im = axes[0, 1].imshow(hist_2d.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Spatial Distribution of Embedding Strength')
    axes[0, 1].set_xlabel('X Coordinate')
    axes[0, 1].set_ylabel('Y Coordinate')
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Cumulative distribution
    sorted_values = np.sort(embed_np.flatten())
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    
    axes[1, 0].plot(sorted_values, cumulative, linewidth=2, color='red')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].set_xlabel('Embedding Strength')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(embed_np, p)
        axes[1, 0].axvline(val, color='gray', linestyle='--', alpha=0.7)
        axes[1, 0].text(val, 0.1, f'{p}%', rotation=90, ha='right')
    
    # 4. Embedding regions analysis
    # Define embedding strength categories
    high_embed = (embed_np > 0.7).sum() / embed_np.size * 100
    medium_embed = ((embed_np > 0.3) & (embed_np <= 0.7)).sum() / embed_np.size * 100
    low_embed = (embed_np <= 0.3).sum() / embed_np.size * 100
    
    categories = ['Low\n(≤0.3)', 'Medium\n(0.3-0.7)', 'High\n(>0.7)']
    percentages = [low_embed, medium_embed, high_embed]
    colors = ['lightblue', 'orange', 'red']
    
    bars = axes[1, 1].bar(categories, percentages, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Embedding Region Distribution')
    axes[1, 1].set_ylabel('Percentage of Pixels (%)')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved to: {save_path}")
    plt.show()

def main():
    """Main visualization demo"""
    print("=" * 60)
    print("EMBEDDING MAP VISUALIZATION DEMO")
    print("=" * 60)
    
    # Create output directory
    output_dir = './visualizations'
    embeddin
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    print("\nCreating sample data...")
    cover_image, secret_image, results = create_sample_data()
    
    # 1. Basic embedding map visualization
    print("\n" + "="*50)
    visualize_embedding_map_basic(
        results['embedding_map'], 
        save_path=os.path.join(output_dir, 'embedding_map_basic.png')
    )
    
    # 2. Attention overlay visualization
    print("\n" + "="*50)
    visualize_attention_overlay(
        cover_image, 
        results['embedding_map'],
        save_path=os.path.join(output_dir, 'attention_overlay.png')
    )
    
    # 3. Comprehensive attention analysis
    print("\n" + "="*50)
    visualize_comprehensive_attention_analysis(
        cover_image, 
        secret_image, 
        results,
        save_path=os.path.join(output_dir, 'comprehensive_analysis.png')
    )
    
    # 4. Embedding statistics
    print("\n" + "="*50)
    visualize_embedding_statistics(
        results['embedding_map'],
        save_path=os.path.join(output_dir, 'embedding_statistics.png')
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMO COMPLETED!")
    print(f"All visualizations saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  ✅ embedding_map_basic.png - Basic embedding map with different colormaps")
    print("  ✅ attention_overlay.png - Embedding map overlayed on cover image")
    print("  ✅ comprehensive_analysis.png - Complete attention analysis")
    print("  ✅ embedding_statistics.png - Statistical analysis of embedding map")
    print("\n These visualizations show:")
    print("  WHERE the secret will be embedded (spatial distribution)")
    print("  HOW STRONG the embedding will be (intensity values)")
    print("  WHAT REGIONS are used for embedding (high/medium/low)")
    print("  STATISTICAL PROPERTIES of the embedding strategy")
    print("=" * 60)

if __name__ == "__main__":
    main()
