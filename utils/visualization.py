import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import seaborn as sns
from matplotlib.patches import Rectangle
import os

def save_attention_visualizations(results, cover_images, secret_images, save_path, num_samples=4):
    """
    Save comprehensive attention visualizations
    
    Args:
        results: Model output dictionary containing attention maps
        cover_images: Cover image tensors [B, C, H, W]
        secret_images: Secret image tensors [B, C, H, W]
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    batch_size = min(cover_images.shape[0], num_samples)
    
    # Create figure with subplots
    fig, axes = plt.subplots(batch_size, 8, figsize=(24, 3 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert tensors to numpy arrays
        cover_img = tensor_to_numpy(cover_images[i])
        secret_img = tensor_to_numpy(secret_images[i])
        stego_img = tensor_to_numpy(results['stego_image'][i])
        extracted_img = tensor_to_numpy(results['extracted_secret'][i])
        
        # Attention maps
        cover_attention = tensor_to_numpy(results['cover_attention']['embedding_attention'][i], squeeze=True)
        secret_attention = tensor_to_numpy(results['secret_attention']['embedding_attention'][i], squeeze=True)
        embedding_map = tensor_to_numpy(results['embedding_map'][i], squeeze=True)
        
        # Plot images and attention maps
        axes[i, 0].imshow(cover_img)
        axes[i, 0].set_title('Cover Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(secret_img)
        axes[i, 1].set_title('Secret Image')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(cover_attention, cmap='hot')
        axes[i, 2].set_title('Cover Attention')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(secret_attention, cmap='hot')
        axes[i, 3].set_title('Secret Attention')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(embedding_map, cmap='viridis')
        axes[i, 4].set_title('Embedding Map')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(stego_img)
        axes[i, 5].set_title('Stego Image')
        axes[i, 5].axis('off')
        
        axes[i, 6].imshow(extracted_img)
        axes[i, 6].set_title('Extracted Secret')
        axes[i, 6].axis('off')
        
        # Difference map
        if 'extraction_map' in results:
            extraction_map = tensor_to_numpy(results['extraction_map'][i], squeeze=True)
            axes[i, 7].imshow(extraction_map, cmap='plasma')
            axes[i, 7].set_title('Extraction Map')
        else:
            diff_map = np.abs(cover_img - stego_img)
            axes[i, 7].imshow(diff_map)
            axes[i, 7].set_title('Difference Map')
        axes[i, 7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_heatmap_overlay(image, attention_map, alpha=0.6, colormap='hot'):
    """
    Create overlay of attention heatmap on original image
    
    Args:
        image: Original image [H, W, C] or [C, H, W]
        attention_map: Attention map [H, W]
        alpha: Transparency of overlay
        colormap: Colormap for attention visualization
        
    Returns:
        Overlayed image
    """
    if isinstance(image, torch.Tensor):
        image = tensor_to_numpy(image)
    if isinstance(attention_map, torch.Tensor):
        attention_map = tensor_to_numpy(attention_map, squeeze=True)
    
    # Normalize attention map
    attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Create colored heatmap
    cmap = plt.get_cmap(colormap)
    attention_colored = cmap(attention_norm)[:, :, :3]  # Remove alpha channel
    
    # Overlay on original image
    overlayed = image * (1 - alpha) + attention_colored * alpha
    
    return np.clip(overlayed, 0, 1)

def create_attention_comparison_grid(cover_images, secret_images, attention_results, strategies, save_path):
    """
    Create comparison grid of different attention strategies
    
    Args:
        cover_images: Cover images tensor
        secret_images: Secret images tensor
        attention_results: Dictionary of results for different strategies
        strategies: List of strategy names
        save_path: Path to save the grid
    """
    num_strategies = len(strategies)
    num_samples = min(4, cover_images.shape[0])
    
    fig, axes = plt.subplots(num_samples, num_strategies + 2, figsize=(4 * (num_strategies + 2), 4 * num_samples))
    
    for i in range(num_samples):
        # Original images
        cover_img = tensor_to_numpy(cover_images[i])
        secret_img = tensor_to_numpy(secret_images[i])
        
        axes[i, 0].imshow(cover_img)
        axes[i, 0].set_title('Cover Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(secret_img)
        axes[i, 1].set_title('Secret Image')
        axes[i, 1].axis('off')
        
        # Strategy results
        for j, strategy in enumerate(strategies):
            if strategy in attention_results:
                embedding_map = tensor_to_numpy(attention_results[strategy]['embedding_map'][i], squeeze=True)
                
                # Create overlay
                overlay = plot_attention_heatmap_overlay(cover_img, embedding_map)
                
                axes[i, j + 2].imshow(overlay)
                axes[i, j + 2].set_title(f'{strategy.title()} Strategy')
                axes[i, j + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(log_data, save_path, metrics=['loss', 'psnr', 'ssim']):
    """
    Plot training curves for various metrics
    
    Args:
        log_data: Dictionary containing training logs
        save_path: Path to save the plot
        metrics: List of metrics to plot
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if f'train_{metric}' in log_data and f'val_{metric}' in log_data:
            epochs = range(len(log_data[f'train_{metric}']))
            
            axes[i].plot(epochs, log_data[f'train_{metric}'], label=f'Train {metric.upper()}', linewidth=2)
            axes[i].plot(epochs, log_data[f'val_{metric}'], label=f'Val {metric.upper()}', linewidth=2)
            
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} vs Epoch')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_attention_statistics(attention_maps, save_path):
    """
    Visualize statistics of attention maps
    
    Args:
        attention_maps: Dictionary of attention maps
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten attention maps for statistics
    cover_att = attention_maps['cover_attention'].flatten()
    secret_att = attention_maps['secret_attention'].flatten()
    embedding_att = attention_maps['embedding_map'].flatten()
    
    # Histograms
    axes[0, 0].hist(cover_att, bins=50, alpha=0.7, label='Cover Attention')
    axes[0, 0].set_title('Cover Attention Distribution')
    axes[0, 0].set_xlabel('Attention Value')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(secret_att, bins=50, alpha=0.7, label='Secret Attention', color='orange')
    axes[0, 1].set_title('Secret Attention Distribution')
    axes[0, 1].set_xlabel('Attention Value')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].hist(embedding_att, bins=50, alpha=0.7, label='Embedding Map', color='green')
    axes[0, 2].set_title('Embedding Map Distribution')
    axes[0, 2].set_xlabel('Attention Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # Box plots
    data_to_plot = [cover_att, secret_att, embedding_att]
    labels = ['Cover', 'Secret', 'Embedding']
    
    axes[1, 0].boxplot(data_to_plot, labels=labels)
    axes[1, 0].set_title('Attention Value Distributions')
    axes[1, 0].set_ylabel('Attention Value')
    
    # Correlation plot
    axes[1, 1].scatter(cover_att[::100], secret_att[::100], alpha=0.5)
    axes[1, 1].set_xlabel('Cover Attention')
    axes[1, 1].set_ylabel('Secret Attention')
    axes[1, 1].set_title('Cover vs Secret Attention')
    
    # Summary statistics
    stats_text = f"""
    Cover Attention:
    Mean: {np.mean(cover_att):.3f}
    Std: {np.std(cover_att):.3f}
    
    Secret Attention:
    Mean: {np.mean(secret_att):.3f}
    Std: {np.std(secret_att):.3f}
    
    Embedding Map:
    Mean: {np.mean(embedding_att):.3f}
    Std: {np.std(embedding_att):.3f}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 2].set_title('Statistics Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_diagram(save_path):
    """
    Create a visual diagram of the attention-guided steganography architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define component positions and sizes
    components = {
        'cover_input': {'pos': (1, 8), 'size': (1.5, 1), 'color': 'lightblue', 'label': 'Cover\nImage'},
        'secret_input': {'pos': (1, 6), 'size': (1.5, 1), 'color': 'lightgreen', 'label': 'Secret\nImage'},
        
        'cover_attention': {'pos': (4, 8), 'size': (2, 1), 'color': 'orange', 'label': 'Cover\nAttention\nGenerator'},
        'secret_attention': {'pos': (4, 6), 'size': (2, 1), 'color': 'orange', 'label': 'Secret\nAttention\nGenerator'},
        
        'attention_fusion': {'pos': (7, 7), 'size': (2, 1), 'color': 'yellow', 'label': 'Attention\nFusion'},
        
        'embedding_net': {'pos': (10, 7), 'size': (2, 1.5), 'color': 'red', 'label': 'Embedding\nNetwork\n(CAISFormer)'},
        
        'stego_output': {'pos': (13, 8), 'size': (1.5, 1), 'color': 'purple', 'label': 'Stego\nImage'},
        
        'extraction_net': {'pos': (10, 4), 'size': (2, 1.5), 'color': 'blue', 'label': 'Extraction\nNetwork'},
        
        'extracted_output': {'pos': (13, 4), 'size': (1.5, 1), 'color': 'lightgreen', 'label': 'Extracted\nSecret'},
        
        'discriminator': {'pos': (13, 1), 'size': (1.5, 1), 'color': 'gray', 'label': 'Discriminator\n(SRNet)'}
    }
    
    # Draw components
    for name, comp in components.items():
        rect = Rectangle(comp['pos'], comp['size'][0], comp['size'][1], 
                        facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2, 
               comp['label'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((2.5, 8.5), (4, 8.5)),  # Cover to cover attention
        ((2.5, 6.5), (4, 6.5)),  # Secret to secret attention
        ((6, 8.5), (7, 7.5)),    # Cover attention to fusion
        ((6, 6.5), (7, 7.5)),    # Secret attention to fusion
        ((9, 7.5), (10, 7.5)),   # Fusion to embedding
        ((12, 7.5), (13, 8.5)),  # Embedding to stego
        ((13.75, 8), (13.75, 5.5)),  # Stego to extraction (curved)
        ((13, 4.75), (12, 4.75)), # Back to extraction
        ((12, 4.75), (13, 4.5)), # Extraction to output
        ((13.75, 8), (13.75, 2)), # Stego to discriminator
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title and labels
    ax.set_title('Attention-Guided Image Steganography Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Set axis properties
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def tensor_to_numpy(tensor, squeeze=False):
    """
    Convert tensor to numpy array for visualization
    
    Args:
        tensor: Input tensor
        squeeze: Whether to squeeze single dimensions
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    
    if squeeze:
        array = np.squeeze(array)
    
    # Handle different tensor formats
    if len(array.shape) == 3 and array.shape[0] in [1, 3]:
        # CHW to HWC
        array = np.transpose(array, (1, 2, 0))
        if array.shape[2] == 1:
            array = np.squeeze(array, axis=2)
    
    # Ensure values are in [0, 1] range
    array = np.clip(array, 0, 1)
    
    return array

def save_comparison_results(results_dict, save_dir):
    """
    Save comprehensive comparison results
    
    Args:
        results_dict: Dictionary containing results for different methods
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create comparison table
    metrics_table = []
    methods = list(results_dict.keys())
    
    for method in methods:
        if 'metrics' in results_dict[method]:
            metrics = results_dict[method]['metrics']
            row = [method, 
                   f"{metrics.get('cover_psnr', 0):.2f}",
                   f"{metrics.get('secret_psnr', 0):.2f}", 
                   f"{metrics.get('cover_ssim', 0):.3f}",
                   f"{metrics.get('secret_ssim', 0):.3f}"]
            metrics_table.append(row)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=metrics_table,
                    colLabels=['Method', 'Cover PSNR', 'Secret PSNR', 'Cover SSIM', 'Secret SSIM'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(methods) + 1):
        for j in range(5):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Performance Comparison of Different Methods', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_analysis_report(attention_data, save_path):
    """
    Create comprehensive attention analysis report
    
    Args:
        attention_data: Dictionary containing attention analysis data
        save_path: Path to save the report
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Attention map examples
    ax1 = fig.add_subplot(gs[0, :2])
    if 'sample_images' in attention_data:
        # Show sample attention overlays
        sample_img = attention_data['sample_images'][0]
        sample_att = attention_data['sample_attention'][0]
        overlay = plot_attention_heatmap_overlay(sample_img, sample_att)
        ax1.imshow(overlay)
        ax1.set_title('Sample Attention Overlay', fontsize=14, fontweight='bold')
        ax1.axis('off')
    
    # Attention statistics
    ax2 = fig.add_subplot(gs[0, 2:])
    if 'attention_stats' in attention_data:
        stats = attention_data['attention_stats']
        strategies = list(stats.keys())
        mean_values = [stats[s]['mean'] for s in strategies]
        std_values = [stats[s]['std'] for s in strategies]
        
        x = np.arange(len(strategies))
        ax2.bar(x, mean_values, yerr=std_values, capsize=5)
        ax2.set_xlabel('Attention Strategy')
        ax2.set_ylabel('Mean Attention Value')
        ax2.set_title('Attention Statistics by Strategy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45)
    
    # Performance metrics
    ax3 = fig.add_subplot(gs[1, :2])
    if 'performance_metrics' in attention_data:
        metrics = attention_data['performance_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        bars = ax3.bar(metric_names, metric_values, color=colors)
        ax3.set_title('Performance Metrics')
        ax3.set_ylabel('Metric Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Attention distribution
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'attention_distribution' in attention_data:
        dist_data = attention_data['attention_distribution']
        ax4.hist(dist_data, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Attention Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Overall Attention Distribution')
    
    # Strategy comparison
    ax5 = fig.add_subplot(gs[2, :])
    if 'strategy_comparison' in attention_data:
        comp_data = attention_data['strategy_comparison']
        strategies = list(comp_data.keys())
        psnr_values = [comp_data[s]['psnr'] for s in strategies]
        ssim_values = [comp_data[s]['ssim'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax5.bar(x - width/2, psnr_values, width, label='PSNR', alpha=0.8)
        ax5.bar(x + width/2, [s*50 for s in ssim_values], width, label='SSIM×50', alpha=0.8)
        
        ax5.set_xlabel('Strategy')
        ax5.set_ylabel('Metric Value')
        ax5.set_title('Strategy Performance Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(strategies)
        ax5.legend()
    
    plt.suptitle('Attention-Guided Steganography Analysis Report', fontsize=18, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Utility function for creating animated attention evolution
def create_attention_animation(attention_sequence, save_path, fps=2):
    """
    Create animation showing attention evolution during training
    
    Args:
        attention_sequence: List of attention maps over training epochs
        save_path: Path to save animation (should end with .gif or .mp4)
        fps: Frames per second
    """
    try:
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame):
            ax.clear()
            attention_map = attention_sequence[frame]
            im = ax.imshow(attention_map, cmap='hot', animated=True)
            ax.set_title(f'Attention Evolution - Epoch {frame + 1}')
            ax.axis('off')
            return [im]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(attention_sequence),
                                     interval=1000//fps, blit=True, repeat=True)
        
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        
        plt.close()
        
    except ImportError:
        print("Animation requires additional dependencies. Skipping animation creation.")
    except Exception as e:
        print(f"Failed to create animation: {e}")
