#!/usr/bin/env python3
"""
Novel Attention-Guided Steganography Training Script - FIXED VERSION
Implements multi-strategy embedding with attention guidance
FIXES: Proper visualization saving for both training and validation epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import gc

# Import novel modules
from models.attention_guided_steganography import AttentionGuidedSteganography
from models.discriminator import SRNetDiscriminator
from utils.dataset import SteganographyDataset
from utils.metrics import compute_metrics

# Import comprehensive visualization functions
import matplotlib.pyplot as plt
from demo_visualization import (
    visualize_comprehensive_attention_analysis,
    visualize_embedding_statistics,
    tensor_to_numpy
)

class NovelSteganographyTrainer:
    """Trainer for novel attention-guided steganography"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create models
        self.generator = AttentionGuidedSteganography(
            input_channels=args.input_channels,
            hidden_channels=args.hidden_channels,
            embedding_strategy=args.embedding_strategy
        ).to(self.device)
        
        self.discriminator = SRNetDiscriminator(
            input_channels=args.input_channels
        ).to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=args.gen_lr,
            betas=(0.5, 0.999)
        )
        
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=args.disc_lr,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Tensorboard writer
        self.writer = SummaryWriter(args.log_dir) if args.log_dir else None
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")

    def train_step(self, cover_images, secret_images):
        """Single training step"""
        batch_size = cover_images.shape[0]
        
        # Generator forward pass
        results = self.generator(
            cover_images, secret_images,
            mode='train',
            embedding_strategy=self.args.embedding_strategy
        )
        
        # Generator losses
        gen_losses = self.compute_generator_losses(results, cover_images, secret_images)
        
        # Discriminator forward pass
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train discriminator
        self.disc_optimizer.zero_grad()
        
        real_pred = self.discriminator(cover_images)
        fake_pred = self.discriminator(results['stego_image'].detach())
        
        disc_real_loss = self.mse_loss(real_pred, real_labels)
        disc_fake_loss = self.mse_loss(fake_pred, fake_labels)
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        
        # Adversarial loss
        fake_pred = self.discriminator(results['stego_image'])
        adv_loss = self.mse_loss(fake_pred, real_labels)
        
        # Total generator loss
        total_gen_loss = (
            gen_losses['cover_loss'] * self.args.cover_loss_weight +
            gen_losses['secret_loss'] * self.args.secret_loss_weight +
            gen_losses['attention_loss'] * self.args.attention_loss_weight +
            gen_losses['perceptual_loss'] * self.args.perceptual_loss_weight +
            adv_loss * self.args.adversarial_loss_weight
        )
        
        total_gen_loss.backward()
        self.gen_optimizer.step()
        
        # Prepare loss dict
        step_losses = {
            'total_gen_loss': total_gen_loss.item(),
            'cover_loss': gen_losses['cover_loss'].item(),
            'secret_loss': gen_losses['secret_loss'].item(),
            'attention_loss': gen_losses['attention_loss'].item(),
            'perceptual_loss': gen_losses['perceptual_loss'].item(),
            'adversarial_loss': adv_loss.item(),
            'disc_loss': disc_loss.item()
        }
        
        return step_losses, results

    def compute_generator_losses(self, results, cover_images, secret_images):
        """Compute generator losses"""
        # Cover reconstruction loss
        cover_loss = self.l1_loss(results['stego_image'], cover_images)
        
        # Secret reconstruction loss
        secret_loss = self.l1_loss(results['extracted_secret'], secret_images)
        
        # Attention consistency loss
        attention_loss = self.mse_loss(
            results['cover_attention']['embedding_attention'],
            results['secret_attention']['embedding_attention']
        )
        
        # Perceptual loss (simplified)
        perceptual_loss = self.mse_loss(results['stego_image'], cover_images)
        
        return {
            'cover_loss': cover_loss,
            'secret_loss': secret_loss,
            'attention_loss': attention_loss,
            'perceptual_loss': perceptual_loss
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch - FIXED VERSION"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        num_batches = len(dataloader)
        
        # Store sample data for visualization
        sample_results = None
        sample_cover = None
        sample_secret = None
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover_images, secret_images) in enumerate(pbar):
            cover_images = cover_images.to(self.device)
            secret_images = secret_images.to(self.device)
            
            # Training step
            step_losses, results = self.train_step(cover_images, secret_images)
            
            # Store first batch for visualization
            if batch_idx == 0:
                sample_cover = cover_images[:1].clone()
                sample_secret = secret_images[:1].clone()
                sample_results = {
                    'stego_image': results['stego_image'][:1].clone(),
                    'extracted_secret': results['extracted_secret'][:1].clone(),
                    'cover_attention': {
                        'embedding_attention': results['cover_attention']['embedding_attention'][:1].clone()
                    },
                    'secret_attention': {
                        'embedding_attention': results['secret_attention']['embedding_attention'][:1].clone()
                    },
                    'embedding_map': results['embedding_map'][:1].clone()
                }
                if 'fusion_weights' in results:
                    sample_results['fusion_weights'] = results['fusion_weights'][:1].clone()
            
            # Accumulate losses
            for key, value in step_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value
            
            # Update progress bar
            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    'Gen': f"{step_losses['total_gen_loss']:.4f}",
                    'Disc': f"{step_losses['disc_loss']:.4f}",
                    'Cover': f"{step_losses['cover_loss']:.4f}",
                    'Secret': f"{step_losses['secret_loss']:.4f}"
                })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.args.log_interval == 0:
                global_step = epoch * num_batches + batch_idx
                for key, value in step_losses.items():
                    self.writer.add_scalar(f'Train/{key}', value, global_step)
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Save training visualizations after epoch - NEW FIX
        if sample_results is not None and sample_cover is not None and sample_secret is not None:
            viz_dir = os.path.join(self.args.output_dir, 'visualizations', 'train')
            os.makedirs(viz_dir, exist_ok=True)
            self.save_epoch_visualizations(sample_results, sample_cover, sample_secret, epoch, viz_dir, mode='train')
        
        return epoch_losses

    def validate(self, dataloader, epoch):
        """Validation step - FIXED VERSION"""
        self.generator.eval()
        
        val_losses = {}
        val_metrics = {'psnr': [], 'ssim': []}
        
        # Store sample data for visualization
        sample_results = None
        sample_cover = None
        sample_secret = None
        
        with torch.no_grad():
            for batch_idx, (cover_images, secret_images) in enumerate(tqdm(dataloader, desc='Validation')):
                cover_images = cover_images.to(self.device)
                secret_images = secret_images.to(self.device)
                
                # Forward pass
                results = self.generator(
                    cover_images, secret_images, 
                    mode='train', 
                    embedding_strategy=self.args.embedding_strategy
                )
                
                # Store first batch for visualization - NEW FIX
                if batch_idx == 0:
                    sample_cover = cover_images[:1].clone()
                    sample_secret = secret_images[:1].clone()
                    sample_results = {
                        'stego_image': results['stego_image'][:1].clone(),
                        'extracted_secret': results['extracted_secret'][:1].clone(),
                        'cover_attention': {
                            'embedding_attention': results['cover_attention']['embedding_attention'][:1].clone()
                        },
                        'secret_attention': {
                            'embedding_attention': results['secret_attention']['embedding_attention'][:1].clone()
                        },
                        'embedding_map': results['embedding_map'][:1].clone()
                    }
                    if 'fusion_weights' in results:
                        sample_results['fusion_weights'] = results['fusion_weights'][:1].clone()
                
                # Compute losses
                losses = self.compute_generator_losses(results, cover_images, secret_images)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0
                    val_losses[key] += value.item()
                
                # Compute metrics
                metrics = compute_metrics(
                    results['stego_image'], cover_images,
                    results['extracted_secret'], secret_images
                )
                
                val_metrics['psnr'].append(metrics['psnr'])
                val_metrics['ssim'].append(metrics['ssim'])
        
        # Average losses and metrics
        for key in val_losses:
            val_losses[key] /= len(dataloader)
        
        val_metrics['psnr'] = np.mean(val_metrics['psnr'])
        val_metrics['ssim'] = np.mean(val_metrics['ssim'])
        
        # Log to tensorboard
        if self.writer:
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Save validation visualizations - FIXED VERSION
        if sample_results is not None and sample_cover is not None and sample_secret is not None:
            viz_dir = os.path.join(self.args.output_dir, 'visualizations', 'val')
            os.makedirs(viz_dir, exist_ok=True)
            self.save_epoch_visualizations(sample_results, sample_cover, sample_secret, epoch, viz_dir, mode='val')

        return val_losses, val_metrics

    def save_epoch_visualizations(self, results, cover_images, secret_images, epoch, save_dir, mode='train'):
        """Save comprehensive visualizations for the epoch - FIXED VERSION"""
        try:
            # Create epoch-specific directory
            epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            print(f"  📊 Saving {mode} epoch {epoch} visualizations...")
            
            # Take first sample for visualization (already should be single sample)
            sample_cover = cover_images[:1] if cover_images.shape[0] > 1 else cover_images
            sample_secret = secret_images[:1] if secret_images.shape[0] > 1 else secret_images
            sample_results = {
                'stego_image': results['stego_image'][:1] if results['stego_image'].shape[0] > 1 else results['stego_image'],
                'extracted_secret': results['extracted_secret'][:1] if results['extracted_secret'].shape[0] > 1 else results['extracted_secret'],
                'cover_attention': {
                    'embedding_attention': results['cover_attention']['embedding_attention'][:1] if results['cover_attention']['embedding_attention'].shape[0] > 1 else results['cover_attention']['embedding_attention']
                },
                'secret_attention': {
                    'embedding_attention': results['secret_attention']['embedding_attention'][:1] if results['secret_attention']['embedding_attention'].shape[0] > 1 else results['secret_attention']['embedding_attention']
                },
                'embedding_map': results['embedding_map'][:1] if results['embedding_map'].shape[0] > 1 else results['embedding_map']
            }
            
            # Add fusion weights if available
            if 'fusion_weights' in results:
                sample_results['fusion_weights'] = results['fusion_weights'][:1] if results['fusion_weights'].shape[0] > 1 else results['fusion_weights']
            
            # 1. Comprehensive attention analysis (12-panel visualization)
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
            
            # 3. Training/Validation progress visualization (6-panel layout)
            self.create_training_progress_viz(sample_results, epoch, epoch_dir, mode)
            
            print(f"  ✅ {mode.capitalize()} visualizations saved to: {epoch_dir}")
            
        except Exception as e:
            print(f"  ⚠️  {mode.capitalize()} visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def create_training_progress_viz(self, results, epoch, save_dir, mode='train'):
        """Create training/validation progress specific visualizations - FIXED VERSION"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Extract data
            cover_att = tensor_to_numpy(results['cover_attention']['embedding_attention'], squeeze=True)
            secret_att = tensor_to_numpy(results['secret_attention']['embedding_attention'], squeeze=True)
            embedding_map = tensor_to_numpy(results['embedding_map'], squeeze=True)
            stego_img = tensor_to_numpy(results['stego_image'])
            extracted_img = tensor_to_numpy(results['extracted_secret'])
            
            # Row 1: Attention maps
            im1 = axes[0, 0].imshow(cover_att, cmap='hot', vmin=0, vmax=1)
            axes[0, 0].set_title(f'Cover Attention\n{mode.capitalize()} Epoch {epoch}')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[0, 1].imshow(secret_att, cmap='hot', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Secret Attention\n{mode.capitalize()} Epoch {epoch}')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            im3 = axes[0, 2].imshow(embedding_map, cmap='viridis', vmin=0, vmax=1)
            axes[0, 2].set_title(f'Embedding Map\n{mode.capitalize()} Epoch {epoch}')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Row 2: Results and statistics
            axes[1, 0].imshow(stego_img)
            axes[1, 0].set_title(f'Stego Image\n{mode.capitalize()} Epoch {epoch}')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(extracted_img)
            axes[1, 1].set_title(f'Extracted Secret\n{mode.capitalize()} Epoch {epoch}')
            axes[1, 1].axis('off')
            
            # Embedding statistics
            axes[1, 2].hist(embedding_map.flatten(), bins=30, alpha=0.7, color='skyblue')
            axes[1, 2].set_title(f'Embedding Distribution\n{mode.capitalize()} Epoch {epoch}')
            axes[1, 2].set_xlabel('Embedding Strength')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add fusion weights if available
            if 'fusion_weights' in results:
                fusion_weights = tensor_to_numpy(results['fusion_weights'], squeeze=True)
                # Add text annotation for fusion weights
                weight_text = f"Fusion Weights:\nTexture: {fusion_weights[0]:.3f}\nCodec: {fusion_weights[1]:.3f}\nAdv: {fusion_weights[2]:.3f}"
                axes[1, 2].text(0.02, 0.98, weight_text, transform=axes[1, 2].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  {mode.capitalize()} progress visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def save_checkpoint(self, epoch, train_losses, val_losses=None, val_metrics=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': self.args
        }
        
        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.args.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save best model
        if is_best:
            best_filepath = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            print(f"New best model saved: {best_filepath}")
        
        print(f"Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Novel Attention-Guided Steganography Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--embedding_strategy', type=str, default='adaptive', 
                       choices=['adaptive', 'high_low', 'low_high'], help='Embedding strategy')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--gen_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--disc_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--cpu_only', action='store_true', help='Use CPU only')
    
    # Loss weights
    parser.add_argument('--cover_loss_weight', type=float, default=1.0, help='Cover loss weight')
    parser.add_argument('--secret_loss_weight', type=float, default=1.0, help='Secret loss weight')
    parser.add_argument('--attention_loss_weight', type=float, default=0.1, help='Attention loss weight')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.5, help='Perceptual loss weight')
    parser.add_argument('--adversarial_loss_weight', type=float, default=0.1, help='Adversarial loss weight')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs_cpu', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("=" * 60)
    print("NOVEL ATTENTION-GUIDED STEGANOGRAPHY TRAINING - FIXED")
    print("=" * 60)
    print(f"Embedding Strategy: {args.embedding_strategy}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.image_size}")
    print(f"Hidden Channels: {args.hidden_channels}")
    print(f"Device: {'CPU' if args.cpu_only else 'GPU (if available)'}")
    print("=" * 60)
    
    # Create datasets
    train_dataset = SteganographyDataset(
        os.path.join(args.data_dir, 'train'),
        image_size=args.image_size,
        mode='train'
    )
    
    val_dataset = SteganographyDataset(
        os.path.join(args.data_dir, 'val'),
        image_size=args.image_size,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.cpu_only
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.cpu_only
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    trainer = NovelSteganographyTrainer(args)
    
    # Training loop
    best_psnr = 0
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 40)
        
        # Training - NOW WITH VISUALIZATIONS
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Print training losses
        print(f"Training Losses:")
        for key, value in train_losses.items():
            print(f"  {key}: {value:.6f}")
        
        # Validation - NOW WITH FIXED VISUALIZATIONS
        val_losses, val_metrics = trainer.validate(val_loader, epoch)
        
        print(f"Validation Losses:")
        for key, value in val_losses.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save checkpoint
        is_best = False
        if val_metrics and val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            best_epoch = epoch
            is_best = True
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            trainer.save_checkpoint(epoch, train_losses, val_losses, val_metrics, is_best)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best PSNR: {best_psnr:.4f} (Epoch {best_epoch + 1})")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("=" * 60)
    
    # Close tensorboard writer
    if trainer.writer:
        trainer.writer.close()


if __name__ == '__main__':
    main()
