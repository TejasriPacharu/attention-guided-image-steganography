import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from tqdm import tqdm
import json

from models import AttentionGuidedSteganography, SRNetDiscriminator
from utils.dataset import SteganographyDataset
from utils.metrics import compute_metrics, compare_attention_strategies
from utils.visualization import (save_attention_visualizations, create_attention_comparison_grid,
                                save_comparison_results, create_attention_analysis_report)

class AttentionGuidedEvaluator:
    """Comprehensive evaluator for attention-guided steganography"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = AttentionGuidedSteganography(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels
        ).to(self.device)
        
        # Load discriminator for security evaluation
        self.discriminator = SRNetDiscriminator(
            input_channels=config.input_channels
        ).to(self.device)
        
        # Load pretrained weights
        if config.model_path:
            self.load_model(config.model_path)
    
    def load_model(self, model_path):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'generator_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['generator_state_dict'])
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def evaluate_attention_strategies(self, test_loader):
        """
        Evaluate different attention strategies
        
        Args:
            test_loader: DataLoader for test dataset
            
        Returns:
            Dictionary containing results for each strategy
        """
        strategies = ['adaptive', 'high_low', 'low_high']
        results = {}
        
        self.model.eval()
        
        with torch.no_grad():
            for strategy in strategies:
                print(f"Evaluating {strategy} strategy...")
                
                strategy_results = {
                    'cover_images': [],
                    'secret_images': [],
                    'stego_images': [],
                    'extracted_secrets': [],
                    'embedding_maps': [],
                    'attention_maps': []
                }
                
                for batch_idx, (cover_images, secret_images) in enumerate(tqdm(test_loader)):
                    if batch_idx >= self.config.max_batches:
                        break
                    
                    cover_images = cover_images.to(self.device)
                    secret_images = secret_images.to(self.device)
                    
                    # Forward pass with specific strategy
                    outputs = self.model(
                        cover_images, secret_images,
                        mode='train',
                        embedding_strategy=strategy
                    )
                    
                    # Store results
                    strategy_results['cover_images'].append(cover_images.cpu())
                    strategy_results['secret_images'].append(secret_images.cpu())
                    strategy_results['stego_images'].append(outputs['stego_image'].cpu())
                    strategy_results['extracted_secrets'].append(outputs['extracted_secret'].cpu())
                    strategy_results['embedding_maps'].append(outputs['embedding_map'].cpu())
                    
                    # Store attention visualizations for first batch
                    if batch_idx == 0:
                        strategy_results['attention_maps'] = outputs
                
                # Concatenate all results
                for key in ['cover_images', 'secret_images', 'stego_images', 'extracted_secrets', 'embedding_maps']:
                    strategy_results[key] = torch.cat(strategy_results[key], dim=0)
                
                # Compute metrics
                metrics = compute_metrics(
                    strategy_results['cover_images'].numpy(),
                    strategy_results['secret_images'].numpy(),
                    strategy_results['stego_images'].numpy(),
                    strategy_results['extracted_secrets'].numpy()
                )
                
                strategy_results['metrics'] = metrics
                results[strategy] = strategy_results
        
        return results
    
    def evaluate_security(self, test_loader, strategy_results):
        """
        Evaluate security against steganalysis
        
        Args:
            test_loader: DataLoader for test dataset
            strategy_results: Results from different strategies
            
        Returns:
            Security evaluation results
        """
        self.discriminator.eval()
        security_results = {}
        
        with torch.no_grad():
            for strategy, results in strategy_results.items():
                print(f"Evaluating security for {strategy} strategy...")
                
                # Get stego images
                stego_images = results['stego_images'][:self.config.security_samples].to(self.device)
                cover_images = results['cover_images'][:self.config.security_samples].to(self.device)
                
                # Discriminator predictions
                stego_preds = self.discriminator(stego_images)
                cover_preds = self.discriminator(cover_images)
                
                # Calculate detection accuracy
                stego_confidence = stego_preds['confidence'].cpu().numpy()
                cover_confidence = cover_preds['confidence'].cpu().numpy()
                
                # Threshold for detection (0.5)
                stego_detected = (stego_confidence > 0.5).astype(int)
                cover_detected = (cover_confidence > 0.5).astype(int)
                
                # Security metrics
                security_metrics = {
                    'stego_detection_rate': np.mean(stego_detected),
                    'cover_false_positive_rate': np.mean(cover_detected),
                    'average_stego_confidence': np.mean(stego_confidence),
                    'average_cover_confidence': np.mean(cover_confidence),
                    'security_score': 1.0 - (np.mean(stego_detected) + np.mean(cover_detected)) / 2
                }
                
                security_results[strategy] = security_metrics
        
        return security_results
    
    def analyze_attention_patterns(self, strategy_results):
        """
        Analyze attention patterns across different strategies
        
        Args:
            strategy_results: Results from different strategies
            
        Returns:
            Attention analysis results
        """
        analysis_results = {}
        
        for strategy, results in strategy_results.items():
            embedding_maps = results['embedding_maps'].numpy()
            
            # Statistical analysis
            attention_stats = {
                'mean': np.mean(embedding_maps),
                'std': np.std(embedding_maps),
                'min': np.min(embedding_maps),
                'max': np.max(embedding_maps),
                'entropy': self._calculate_entropy(embedding_maps),
                'sparsity': np.mean(embedding_maps < 0.1),  # Percentage of low attention
                'concentration': np.mean(embedding_maps > 0.9)  # Percentage of high attention
            }
            
            analysis_results[strategy] = attention_stats
        
        return analysis_results
    
    def _calculate_entropy(self, attention_maps):
        """Calculate entropy of attention maps"""
        # Flatten and normalize
        flat_maps = attention_maps.flatten()
        
        # Create histogram
        hist, _ = np.histogram(flat_maps, bins=50, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize to probabilities
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return entropy
    
    def evaluate_robustness(self, test_loader, strategy_results):
        """
        Evaluate robustness to image processing operations
        
        Args:
            test_loader: DataLoader for test dataset
            strategy_results: Results from different strategies
            
        Returns:
            Robustness evaluation results
        """
        robustness_results = {}
        
        # Define image processing operations
        operations = {
            'jpeg_compression': lambda x: self._jpeg_compression(x, quality=75),
            'gaussian_noise': lambda x: self._add_gaussian_noise(x, std=0.01),
            'gaussian_blur': lambda x: self._gaussian_blur(x, kernel_size=3),
            'resize': lambda x: self._resize_operation(x, scale=0.8)
        }
        
        for strategy, results in strategy_results.items():
            print(f"Evaluating robustness for {strategy} strategy...")
            
            strategy_robustness = {}
            
            # Get sample stego images
            stego_images = results['stego_images'][:self.config.robustness_samples]
            secret_images = results['secret_images'][:self.config.robustness_samples]
            
            for op_name, operation in operations.items():
                # Apply operation to stego images
                processed_stego = operation(stego_images)
                
                # Extract secrets from processed images
                with torch.no_grad():
                    processed_stego_tensor = processed_stego.to(self.device)
                    extraction_results = self.model(processed_stego_tensor, mode='extract')
                    extracted_secrets = extraction_results['extracted_secret'].cpu()
                
                # Compute extraction quality metrics
                extraction_metrics = compute_metrics(
                    secret_images.numpy(),
                    secret_images.numpy(),  # Dummy cover for this evaluation
                    extracted_secrets.numpy(),
                    extracted_secrets.numpy()
                )
                
                strategy_robustness[op_name] = {
                    'extraction_psnr': extraction_metrics['secret_psnr'],
                    'extraction_ssim': extraction_metrics['secret_ssim']
                }
            
            robustness_results[strategy] = strategy_robustness
        
        return robustness_results
    
    def _jpeg_compression(self, images, quality=75):
        """Simulate JPEG compression"""
        # This is a simplified simulation
        # In practice, you would use actual JPEG compression
        noise = torch.randn_like(images) * (100 - quality) / 1000
        return torch.clamp(images + noise, 0, 1)
    
    def _add_gaussian_noise(self, images, std=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(images) * std
        return torch.clamp(images + noise, 0, 1)
    
    def _gaussian_blur(self, images, kernel_size=3):
        """Apply Gaussian blur (simplified)"""
        # Simplified blur using average pooling
        blur_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        blurred = torch.nn.functional.conv2d(
            images.view(-1, 1, *images.shape[-2:]),
            blur_kernel, padding=kernel_size//2
        )
        return blurred.view_as(images)
    
    def _resize_operation(self, images, scale=0.8):
        """Resize operation"""
        h, w = images.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize down then up
        resized = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        restored = torch.nn.functional.interpolate(
            resized, size=(h, w), mode='bilinear', align_corners=False
        )
        
        return restored
    
    def generate_comprehensive_report(self, strategy_results, security_results, 
                                    attention_analysis, robustness_results, save_dir):
        """
        Generate comprehensive evaluation report
        
        Args:
            strategy_results: Results from strategy evaluation
            security_results: Security evaluation results
            attention_analysis: Attention pattern analysis
            robustness_results: Robustness evaluation results
            save_dir: Directory to save the report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create summary report
        report = {
            'evaluation_summary': {
                'strategies_evaluated': list(strategy_results.keys()),
                'total_samples': len(strategy_results[list(strategy_results.keys())[0]]['cover_images']),
                'device': str(self.device)
            },
            'performance_metrics': {},
            'security_metrics': security_results,
            'attention_analysis': attention_analysis,
            'robustness_metrics': robustness_results
        }
        
        # Extract performance metrics
        for strategy, results in strategy_results.items():
            report['performance_metrics'][strategy] = results['metrics']
        
        # Save JSON report
        with open(os.path.join(save_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_evaluation_visualizations(
            strategy_results, security_results, attention_analysis, save_dir
        )
        
        # Print summary
        self._print_evaluation_summary(report)
    
    def _generate_evaluation_visualizations(self, strategy_results, security_results, 
                                          attention_analysis, save_dir):
        """Generate evaluation visualizations"""
        
        # Strategy comparison grid
        strategies = list(strategy_results.keys())
        sample_covers = strategy_results[strategies[0]]['cover_images'][:4]
        sample_secrets = strategy_results[strategies[0]]['secret_images'][:4]
        
        attention_results = {}
        for strategy in strategies:
            attention_results[strategy] = {
                'embedding_map': strategy_results[strategy]['embedding_maps'][:4]
            }
        
        create_attention_comparison_grid(
            sample_covers, sample_secrets, attention_results, strategies,
            os.path.join(save_dir, 'strategy_comparison.png')
        )
        
        # Attention visualizations for each strategy
        for strategy, results in strategy_results.items():
            if 'attention_maps' in results:
                save_attention_visualizations(
                    results['attention_maps'],
                    sample_covers[:2], sample_secrets[:2],
                    os.path.join(save_dir, f'{strategy}_attention_visualization.png'),
                    num_samples=2
                )
        
        # Performance comparison table
        save_comparison_results(strategy_results, save_dir)
        
        # Attention analysis report
        attention_data = {
            'attention_stats': attention_analysis,
            'performance_metrics': {s: r['metrics'] for s, r in strategy_results.items()},
            'strategy_comparison': {s: r['metrics'] for s, r in strategy_results.items()}
        }
        
        create_attention_analysis_report(
            attention_data,
            os.path.join(save_dir, 'attention_analysis_report.png')
        )
    
    def _print_evaluation_summary(self, report):
        """Print evaluation summary to console"""
        print("\n" + "="*80)
        print("ATTENTION-GUIDED STEGANOGRAPHY EVALUATION SUMMARY")
        print("="*80)
        
        # Performance metrics
        print("\n📊 PERFORMANCE METRICS:")
        print("-" * 50)
        for strategy, metrics in report['performance_metrics'].items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Cover PSNR: {metrics['cover_psnr']:.2f} dB")
            print(f"  Secret PSNR: {metrics['secret_psnr']:.2f} dB")
            print(f"  Cover SSIM: {metrics['cover_ssim']:.4f}")
            print(f"  Secret SSIM: {metrics['secret_ssim']:.4f}")
        
        # Security metrics
        print("\n🔒 SECURITY METRICS:")
        print("-" * 50)
        for strategy, metrics in report['security_metrics'].items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Detection Rate: {metrics['stego_detection_rate']:.3f}")
            print(f"  False Positive Rate: {metrics['cover_false_positive_rate']:.3f}")
            print(f"  Security Score: {metrics['security_score']:.3f}")
        
        # Attention analysis
        print("\n🎯 ATTENTION ANALYSIS:")
        print("-" * 50)
        for strategy, stats in report['attention_analysis'].items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Mean Attention: {stats['mean']:.3f}")
            print(f"  Attention Entropy: {stats['entropy']:.3f}")
            print(f"  Sparsity: {stats['sparsity']:.3f}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Attention-Guided Steganography')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels')
    
    # Data parameters
    parser.add_argument('--test_data_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Evaluation parameters
    parser.add_argument('--max_batches', type=int, default=50, help='Maximum batches to evaluate')
    parser.add_argument('--security_samples', type=int, default=200, help='Samples for security evaluation')
    parser.add_argument('--robustness_samples', type=int, default=100, help='Samples for robustness evaluation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataset
    test_dataset = SteganographyDataset(
        args.test_data_dir,
        image_size=args.image_size,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create evaluator
    evaluator = AttentionGuidedEvaluator(args)
    
    print("Starting comprehensive evaluation...")
    
    # Evaluate attention strategies
    print("1. Evaluating attention strategies...")
    strategy_results = evaluator.evaluate_attention_strategies(test_loader)
    
    # Evaluate security
    print("2. Evaluating security...")
    security_results = evaluator.evaluate_security(test_loader, strategy_results)
    
    # Analyze attention patterns
    print("3. Analyzing attention patterns...")
    attention_analysis = evaluator.analyze_attention_patterns(strategy_results)
    
    # Evaluate robustness
    print("4. Evaluating robustness...")
    robustness_results = evaluator.evaluate_robustness(test_loader, strategy_results)
    
    # Generate comprehensive report
    print("5. Generating comprehensive report...")
    evaluator.generate_comprehensive_report(
        strategy_results, security_results, attention_analysis, 
        robustness_results, args.output_dir
    )
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
