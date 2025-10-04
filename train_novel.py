#!/usr/bin/env python3
"""
Novel Attention-Guided Steganography Training Script
Implements multi-strategy embedding with attention guidance
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
from utils.visualization import save_attention_visualizations

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
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Perceptual loss (VGG-based)
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16].to(self.device)
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.use_perceptual = True
        except:
            print("Warning: VGG not available, using L1 loss instead of perceptual loss")
            self.use_perceptual = False
        
        # TensorBoard
        self.writer = SummaryWriter(args.log_dir) if args.log_dir else None
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def compute_perceptual_loss(self, x, y):
        """Compute perceptual loss using VGG features"""
        if not self.use_perceptual:
            return self.l1_loss(x, y)
        
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.mse_loss(x_features, y_features)
    
    def compute_generator_losses(self, results, cover_image, secret_image):
        """Compute comprehensive generator losses"""
        losses = {}
        
        # 1. Cover preservation loss
        losses['cover_loss'] = self.mse_loss(results['stego_image'], cover_image)
        
        # 2. Secret reconstruction loss
        losses['secret_loss'] = self.mse_loss(results['extracted_secret'], secret_image)
        
        # 3. Perceptual loss for visual quality
        losses['perceptual_loss'] = self.compute_perceptual_loss(results['stego_image'], cover_image)
        
        # 4. Attention consistency loss
        if 'attention_consistency' in results and results['attention_consistency'] is not None:
            target_consistency = torch.ones_like(results['attention_consistency'])
            losses['attention_consistency_loss'] = self.bce_loss(
                results['attention_consistency'], target_consistency
            )
        else:
            losses['attention_consistency_loss'] = torch.tensor(0.0, device=self.device)
        
        # 5. Strategy diversity loss (encourage different strategies to produce different results)
        if 'texture_stego' in results:
            texture_diff = self.l1_loss(results['texture_stego'], results['codec_stego'])
            codec_adv_diff = self.l1_loss(results['codec_stego'], results['adversarial_stego'])
            losses['diversity_loss'] = -0.1 * (texture_diff + codec_adv_diff)  # Negative to encourage diversity
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=self.device)
        
        # 6. Fusion weight regularization (encourage balanced fusion)
        if 'fusion_weights' in results:
            fusion_weights = results['fusion_weights']
            # Encourage balanced weights (not too concentrated on one strategy)
            weight_entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=1)
            losses['fusion_regularization'] = -0.1 * weight_entropy.mean()  # Encourage high entropy
        else:
            losses['fusion_regularization'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def train_step(self, cover_images, secret_images):
        """Single training step"""
        batch_size = cover_images.size(0)
        
        # Labels for discriminator
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # =================== Train Generator ===================
        self.gen_optimizer.zero_grad()
        
        # Forward pass through generator
        gen_results = self.generator(
            cover_images, secret_images, 
            mode='train', 
            embedding_strategy=self.args.embedding_strategy
        )
        
        # Compute generator losses
        gen_losses = self.compute_generator_losses(gen_results, cover_images, secret_images)
        
        # Adversarial loss for generator
        disc_fake = self.discriminator(gen_results['stego_image'])
        disc_fake_logits = disc_fake['logits'] if isinstance(disc_fake, dict) else disc_fake
        gen_losses['adversarial_loss'] = self.bce_loss(disc_fake_logits, real_labels)
        
        # Total generator loss
        total_gen_loss = (
            self.args.cover_loss_weight * gen_losses['cover_loss'] +
            self.args.secret_loss_weight * gen_losses['secret_loss'] +
            self.args.perceptual_loss_weight * gen_losses['perceptual_loss'] +
            self.args.attention_loss_weight * gen_losses['attention_consistency_loss'] +
            self.args.adversarial_loss_weight * gen_losses['adversarial_loss'] +
            0.1 * gen_losses['diversity_loss'] +
            0.1 * gen_losses['fusion_regularization']
        )
        
        total_gen_loss.backward()
        self.gen_optimizer.step()
        
        # =================== Train Discriminator ===================
        self.disc_optimizer.zero_grad()
        
        # Real images
        disc_real = self.discriminator(cover_images)
        real_loss = self.bce_loss(disc_real, real_labels)
        
        # Fake images (detached to avoid generator gradients)
        disc_fake = self.discriminator(gen_results['stego_image'].detach())
        fake_loss = self.bce_loss(disc_fake, fake_labels)
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Prepare return values
        step_losses = {
            'total_gen_loss': total_gen_loss.item(),
            'disc_loss': disc_loss.item(),
            **{k: v.item() if torch.is_tensor(v) else v for k, v in gen_losses.items()}
        }
        
        return step_losses, gen_results
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover_images, secret_images) in enumerate(pbar):
            cover_images = cover_images.to(self.device)
            secret_images = secret_images.to(self.device)
            
            # Training step
            step_losses, results = self.train_step(cover_images, secret_images)
            
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
        
        return epoch_losses
    
    def validate(self, dataloader, epoch):
        """Validation step"""
        self.generator.eval()
        
        val_losses = {}
        val_metrics = {'psnr': [], 'ssim': []}
        
        with torch.no_grad():
            for cover_images, secret_images in tqdm(dataloader, desc='Validation'):
                cover_images = cover_images.to(self.device)
                secret_images = secret_images.to(self.device)
                
                # Forward pass
                results = self.generator(
                    cover_images, secret_images, 
                    mode='train', 
                    embedding_strategy=self.args.embedding_strategy
                )
                
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

        return val_losses, val_metrics
    

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
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("=" * 60)
    print("NOVEL ATTENTION-GUIDED STEGANOGRAPHY TRAINING")
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
        
        # Training
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Print training losses
        print(f"Training Losses:")
        for key, value in train_losses.items():
            print(f"  {key}: {value:.6f}")
        
        # Validation
        val_losses = None
        val_metrics = None
        
        if (epoch + 1) % args.val_interval == 0:
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
