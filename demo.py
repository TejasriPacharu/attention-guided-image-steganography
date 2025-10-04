import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from models import AttentionGuidedSteganography
from utils.visualization import save_attention_visualizations, create_architecture_diagram
from utils.metrics import compute_metrics

class SteganographyDemo:
    """Interactive demo for attention-guided steganography"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = AttentionGuidedSteganography().to(self.device)
        self.load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.inverse_transform = transforms.ToPILImage()
    
    def load_model(self, model_path):
        """Load pretrained model"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'generator_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        return self.inverse_transform(tensor)
    
    def embed_secret(self, cover_path, secret_path, strategy='adaptive', save_dir='./demo_results'):
        """
        Embed secret image into cover image
        
        Args:
            cover_path: Path to cover image
            secret_path: Path to secret image
            strategy: Embedding strategy ('adaptive', 'high_low', 'low_high')
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load images
        cover_tensor = self.load_image(cover_path)
        secret_tensor = self.load_image(secret_path)
        
        print(f"Embedding secret using {strategy} strategy...")
        
        with torch.no_grad():
            # Embedding
            results = self.model(
                cover_tensor, secret_tensor,
                mode='train',
                embedding_strategy=strategy
            )
            
            # Convert tensors to images
            cover_img = self.tensor_to_image(cover_tensor)
            secret_img = self.tensor_to_image(secret_tensor)
            stego_img = self.tensor_to_image(results['stego_image'])
            extracted_img = self.tensor_to_image(results['extracted_secret'])
            
            # Compute metrics
            metrics = compute_metrics(
                cover_tensor.cpu().numpy(),
                secret_tensor.cpu().numpy(),
                results['stego_image'].cpu().numpy(),
                results['extracted_secret'].cpu().numpy()
            )
            
            # Save results
            demo_results = {
                'cover_image': cover_img,
                'secret_image': secret_img,
                'stego_image': stego_img,
                'extracted_secret': extracted_img,
                'metrics': metrics,
                'attention_maps': results
            }
            
            self.save_demo_results(demo_results, save_dir, strategy)
            
            return demo_results
    
    def extract_secret(self, stego_path, save_dir='./demo_results'):
        """
        Extract secret from stego image
        
        Args:
            stego_path: Path to stego image
            save_dir: Directory to save results
            
        Returns:
            Extracted secret image
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load stego image
        stego_tensor = self.load_image(stego_path)
        
        print("Extracting secret from stego image...")
        
        with torch.no_grad():
            results = self.model(stego_tensor, mode='extract')
            extracted_img = self.tensor_to_image(results['extracted_secret'])
            
            # Save extracted image
            extracted_img.save(os.path.join(save_dir, 'extracted_secret.png'))
            
            return extracted_img
    
    def save_demo_results(self, results, save_dir, strategy):
        """Save demo results with visualizations"""
        
        # Save individual images
        results['cover_image'].save(os.path.join(save_dir, 'cover_image.png'))
        results['secret_image'].save(os.path.join(save_dir, 'secret_image.png'))
        results['stego_image'].save(os.path.join(save_dir, 'stego_image.png'))
        results['extracted_secret'].save(os.path.join(save_dir, 'extracted_secret.png'))
        
        # Create comprehensive visualization
        self.create_demo_visualization(results, save_dir, strategy)
        
        # Print metrics
        self.print_metrics(results['metrics'])
    
    def create_demo_visualization(self, results, save_dir, strategy):
        """Create comprehensive demo visualization"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Convert PIL images to numpy arrays
        cover_np = np.array(results['cover_image']) / 255.0
        secret_np = np.array(results['secret_image']) / 255.0
        stego_np = np.array(results['stego_image']) / 255.0
        extracted_np = np.array(results['extracted_secret']) / 255.0
        
        # Get attention maps
        attention_maps = results['attention_maps']
        cover_attention = attention_maps['cover_attention']['embedding_attention'][0].cpu().numpy().squeeze()
        secret_attention = attention_maps['secret_attention']['embedding_attention'][0].cpu().numpy().squeeze()
        embedding_map = attention_maps['embedding_map'][0].cpu().numpy().squeeze()
        
        # First row: Original images and attention maps
        axes[0, 0].imshow(cover_np)
        axes[0, 0].set_title('Cover Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(secret_np)
        axes[0, 1].set_title('Secret Image', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        im1 = axes[0, 2].imshow(cover_attention, cmap='hot')
        axes[0, 2].set_title('Cover Attention', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        im2 = axes[0, 3].imshow(secret_attention, cmap='hot')
        axes[0, 3].set_title('Secret Attention', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Second row: Results and analysis
        im3 = axes[1, 0].imshow(embedding_map, cmap='viridis')
        axes[1, 0].set_title('Embedding Map', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        axes[1, 1].imshow(stego_np)
        axes[1, 1].set_title('Stego Image', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(extracted_np)
        axes[1, 2].set_title('Extracted Secret', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Difference map
        diff_map = np.abs(cover_np - stego_np)
        diff_map = np.mean(diff_map, axis=2)  # Convert to grayscale
        im4 = axes[1, 3].imshow(diff_map, cmap='plasma')
        axes[1, 3].set_title('Difference Map', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        # Add strategy information
        fig.suptitle(f'Attention-Guided Steganography Demo - {strategy.title()} Strategy', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'demo_visualization_{strategy}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\n" + "="*60)
        print("STEGANOGRAPHY RESULTS")
        print("="*60)
        print(f"Cover Image Preservation:")
        print(f"  PSNR: {metrics['cover_psnr']:.2f} dB")
        print(f"  SSIM: {metrics['cover_ssim']:.4f}")
        print(f"  MSE:  {metrics['cover_mse']:.6f}")
        
        print(f"\nSecret Image Recovery:")
        print(f"  PSNR: {metrics['secret_psnr']:.2f} dB")
        print(f"  SSIM: {metrics['secret_ssim']:.4f}")
        print(f"  MSE:  {metrics['secret_mse']:.6f}")
        
        print(f"\nOverall Quality:")
        print(f"  Average PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Average SSIM: {metrics['ssim']:.4f}")
        print("="*60)
    
    def compare_strategies(self, cover_path, secret_path, save_dir='./demo_results'):
        """
        Compare different embedding strategies
        
        Args:
            cover_path: Path to cover image
            secret_path: Path to secret image
            save_dir: Directory to save results
        """
        strategies = ['adaptive', 'high_low', 'low_high']
        comparison_results = {}
        
        print("Comparing different attention strategies...")
        
        for strategy in strategies:
            print(f"\nTesting {strategy} strategy...")
            results = self.embed_secret(
                cover_path, secret_path, strategy,
                os.path.join(save_dir, f'{strategy}_results')
            )
            comparison_results[strategy] = results
        
        # Create comparison visualization
        self.create_strategy_comparison(comparison_results, save_dir)
        
        # Print comparison summary
        self.print_strategy_comparison(comparison_results)
        
        return comparison_results
    
    def create_strategy_comparison(self, comparison_results, save_dir):
        """Create strategy comparison visualization"""
        
        strategies = list(comparison_results.keys())
        fig, axes = plt.subplots(len(strategies), 5, figsize=(20, 4 * len(strategies)))
        
        if len(strategies) == 1:
            axes = axes.reshape(1, -1)
        
        for i, strategy in enumerate(strategies):
            results = comparison_results[strategy]
            
            # Convert images to numpy
            cover_np = np.array(results['cover_image']) / 255.0
            stego_np = np.array(results['stego_image']) / 255.0
            extracted_np = np.array(results['extracted_secret']) / 255.0
            
            # Get embedding map
            embedding_map = results['attention_maps']['embedding_map'][0].cpu().numpy().squeeze()
            
            # Plot images
            axes[i, 0].imshow(cover_np)
            axes[i, 0].set_title(f'{strategy.title()}\nCover')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(embedding_map, cmap='viridis')
            axes[i, 1].set_title('Embedding Map')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(stego_np)
            axes[i, 2].set_title('Stego Image')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(extracted_np)
            axes[i, 3].set_title('Extracted Secret')
            axes[i, 3].axis('off')
            
            # Metrics text
            metrics = results['metrics']
            metrics_text = f"Cover PSNR: {metrics['cover_psnr']:.1f}\n"
            metrics_text += f"Secret PSNR: {metrics['secret_psnr']:.1f}\n"
            metrics_text += f"Cover SSIM: {metrics['cover_ssim']:.3f}\n"
            metrics_text += f"Secret SSIM: {metrics['secret_ssim']:.3f}"
            
            axes[i, 4].text(0.1, 0.5, metrics_text, fontsize=10, 
                           verticalalignment='center', transform=axes[i, 4].transAxes)
            axes[i, 4].set_title('Metrics')
            axes[i, 4].axis('off')
        
        plt.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'strategy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_strategy_comparison(self, comparison_results):
        """Print strategy comparison summary"""
        print("\n" + "="*80)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison table
        strategies = list(comparison_results.keys())
        
        print(f"{'Strategy':<12} {'Cover PSNR':<12} {'Secret PSNR':<13} {'Cover SSIM':<12} {'Secret SSIM':<12}")
        print("-" * 80)
        
        for strategy in strategies:
            metrics = comparison_results[strategy]['metrics']
            print(f"{strategy:<12} {metrics['cover_psnr']:<12.2f} {metrics['secret_psnr']:<13.2f} "
                  f"{metrics['cover_ssim']:<12.4f} {metrics['secret_ssim']:<12.4f}")
        
        print("="*80)
        
        # Find best strategy for each metric
        best_cover_psnr = max(strategies, key=lambda s: comparison_results[s]['metrics']['cover_psnr'])
        best_secret_psnr = max(strategies, key=lambda s: comparison_results[s]['metrics']['secret_psnr'])
        best_cover_ssim = max(strategies, key=lambda s: comparison_results[s]['metrics']['cover_ssim'])
        best_secret_ssim = max(strategies, key=lambda s: comparison_results[s]['metrics']['secret_ssim'])
        
        print(f"\nBest Strategies:")
        print(f"  Cover PSNR: {best_cover_psnr}")
        print(f"  Secret PSNR: {best_secret_psnr}")
        print(f"  Cover SSIM: {best_cover_ssim}")
        print(f"  Secret SSIM: {best_secret_ssim}")

def main():
    parser = argparse.ArgumentParser(description='Attention-Guided Steganography Demo')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Demo mode
    parser.add_argument('--mode', type=str, default='embed', 
                       choices=['embed', 'extract', 'compare', 'architecture'],
                       help='Demo mode')
    
    # Input images
    parser.add_argument('--cover_image', type=str, help='Path to cover image')
    parser.add_argument('--secret_image', type=str, help='Path to secret image')
    parser.add_argument('--stego_image', type=str, help='Path to stego image (for extraction)')
    
    # Parameters
    parser.add_argument('--strategy', type=str, default='adaptive',
                       choices=['adaptive', 'high_low', 'low_high'],
                       help='Embedding strategy')
    parser.add_argument('--output_dir', type=str, default='./demo_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'architecture':
        # Create architecture diagram
        print("Creating architecture diagram...")
        create_architecture_diagram(os.path.join(args.output_dir, 'architecture.png'))
        print(f"Architecture diagram saved to {args.output_dir}/architecture.png")
        return
    
    # Initialize demo
    demo = SteganographyDemo(args.model_path, args.device)
    
    if args.mode == 'embed':
        if not args.cover_image or not args.secret_image:
            print("Error: Cover and secret images required for embedding mode")
            return
        
        print(f"Embedding secret image into cover image...")
        results = demo.embed_secret(args.cover_image, args.secret_image, 
                                  args.strategy, args.output_dir)
        print(f"Results saved to {args.output_dir}")
    
    elif args.mode == 'extract':
        if not args.stego_image:
            print("Error: Stego image required for extraction mode")
            return
        
        print("Extracting secret from stego image...")
        extracted = demo.extract_secret(args.stego_image, args.output_dir)
        print(f"Extracted secret saved to {args.output_dir}/extracted_secret.png")
    
    elif args.mode == 'compare':
        if not args.cover_image or not args.secret_image:
            print("Error: Cover and secret images required for comparison mode")
            return
        
        print("Comparing different attention strategies...")
        comparison_results = demo.compare_strategies(args.cover_image, args.secret_image, 
                                                   args.output_dir)
        print(f"Comparison results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
