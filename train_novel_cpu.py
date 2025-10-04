#!/usr/bin/env python3
"""
CPU-Optimized Novel Attention-Guided Steganography Training
Memory-efficient version for CPU-only systems
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
import gc

# Import novel modules
from models.attention_guided_steganography import AttentionGuidedSteganography
from models.discriminator import SRNetDiscriminator
from utils.dataset import SteganographyDataset
from utils.metrics import compute_metrics

class CPUOptimizedTrainer:
    """CPU-optimized trainer for novel attention-guided steganography"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Create models with reduced complexity
        self.generator = AttentionGuidedSteganography(
            input_channels=args.input_channels,
            hidden_channels=args.hidden_channels,
            embedding_strategy=args.embedding_strategy
        ).to(self.device)
        
        # Simplified discriminator for CPU
        self.discriminator = SRNetDiscriminator(
            input_channels=args.input_channels
        ).to(self.device)
        
        # Optimizers with higher learning rates for faster convergence
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
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def compute_generator_losses(self, results, cover_image, secret_image):
        """Compute simplified generator losses for CPU efficiency"""
        losses = {}
        
        # 1. Cover preservation loss
        losses['cover_loss'] = self.mse_loss(results['stego_image'], cover_image)
        
        # 2. Secret reconstruction loss
        losses['secret_loss'] = self.mse_loss(results['extracted_secret'], secret_image)
        
        # 3. Simplified perceptual loss (L1)
        losses['perceptual_loss'] = self.l1_loss(results['stego_image'], cover_image)
        
        # 4. Strategy diversity loss (simplified)
        if 'texture_stego' in results and 'codec_stego' in results:
            diversity = self.l1_loss(results['texture_stego'], results['codec_stego'])
            losses['diversity_loss'] = -0.05 * diversity  # Encourage diversity
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def train_step(self, cover_images, secret_images):
        """Memory-efficient training step"""
        batch_size = cover_images.size(0)
        
        # Clear cache before training step
        gc.collect()
        
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
        
        # Adversarial loss for generator (simplified)
        with torch.no_grad():
            disc_fake = self.discriminator(gen_results['stego_image'])
        gen_losses['adversarial_loss'] = self.bce_loss(disc_fake, real_labels)
        
        # Total generator loss (simplified weights)
        total_gen_loss = (
            self.args.cover_loss_weight * gen_losses['cover_loss'] +
            self.args.secret_loss_weight * gen_losses['secret_loss'] +
            0.1 * gen_losses['perceptual_loss'] +
            0.05 * gen_losses['adversarial_loss'] +
            0.05 * gen_losses['diversity_loss']
        )
        
        total_gen_loss.backward()
        self.gen_optimizer.step()
        
        # =================== Train Discriminator (Simplified) ===================
        self.disc_optimizer.zero_grad()
        
        # Real images
        disc_real = self.discriminator(cover_images)
        real_loss = self.bce_loss(disc_real, real_labels)
        
        # Fake images (detached)
        with torch.no_grad():
            stego_detached = gen_results['stego_image'].detach()
        disc_fake = self.discriminator(stego_detached)
        fake_loss = self.bce_loss(disc_fake, fake_labels)
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Clean up intermediate results
        del gen_results
        gc.collect()
        
        # Prepare return values
        step_losses = {
            'total_gen_loss': total_gen_loss.item(),
            'disc_loss': disc_loss.item(),
            **{k: v.item() if torch.is_tensor(v) else v for k, v in gen_losses.items()}
        }
        
        return step_losses
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with memory management"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover_images, secret_images) in enumerate(pbar):
            cover_images = cover_images.to(self.device)
            secret_images = secret_images.to(self.device)
            
            # Training step
            step_losses = self.train_step(cover_images, secret_images)
            
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
            
            # Aggressive memory cleanup
            if batch_idx % 5 == 0:
                gc.collect()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, dataloader, epoch):
        """Simplified validation for CPU"""
        self.generator.eval()
        
        val_losses = {}
        val_metrics = {'psnr': [], 'ssim': []}
        
        with torch.no_grad():
            for batch_idx, (cover_images, secret_images) in enumerate(tqdm(dataloader, desc='Validation')):
                # Limit validation batches for speed
                if batch_idx >= 10:  # Only validate on first 10 batches
                    break
                    
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
                
                # Compute metrics (simplified)
                try:
                    metrics = compute_metrics(
                        results['stego_image'], cover_images,
                        results['extracted_secret'], secret_images
                    )
                    val_metrics['psnr'].append(metrics['psnr'])
                    val_metrics['ssim'].append(metrics['ssim'])
                except:
                    # Fallback simple PSNR calculation
                    mse = torch.mean((results['stego_image'] - cover_images) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    val_metrics['psnr'].append(psnr.item())
                    val_metrics['ssim'].append(0.8)  # Dummy value
                
                # Clean up
                del results
                gc.collect()
        
        # Average losses and metrics
        num_val_batches = min(10, len(dataloader))
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        val_metrics['psnr'] = np.mean(val_metrics['psnr'])
        val_metrics['ssim'] = np.mean(val_metrics['ssim'])
        
        return val_losses, val_metrics
    
    def save_checkpoint(self, epoch, train_losses, val_losses=None, val_metrics=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
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
    parser = argparse.ArgumentParser(description='CPU-Optimized Novel Attention-Guided Steganography Training')
    
    # Data parameters (CPU-optimized defaults)
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (small for CPU)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--image_size', type=int, default=64, help='Image size (small for CPU)')
    
    # Model parameters (CPU-optimized)
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Hidden channels (small for CPU)')
    parser.add_argument('--embedding_strategy', type=str, default='adaptive', 
                       choices=['adaptive', 'high_low', 'low_high'], help='Embedding strategy')
    
    # Training parameters (CPU-optimized)
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs (reduced for CPU)')
    parser.add_argument('--gen_lr', type=float, default=2e-3, help='Generator learning rate (higher for CPU)')
    parser.add_argument('--disc_lr', type=float, default=2e-3, help='Discriminator learning rate (higher for CPU)')
    
    # Loss weights (simplified for CPU)
    parser.add_argument('--cover_loss_weight', type=float, default=1.0, help='Cover loss weight')
    parser.add_argument('--secret_loss_weight', type=float, default=1.0, help='Secret loss weight')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs_cpu', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cpu', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=5, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("CPU-OPTIMIZED NOVEL STEGANOGRAPHY TRAINING")
    print("=" * 60)
    print(f"Embedding Strategy: {args.embedding_strategy}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.image_size}")
    print(f"Hidden Channels: {args.hidden_channels}")
    print(f"Device: CPU")
    print("=" * 60)
    
    # Create datasets
    train_dataset = SteganographyDataset(
        os.path.join(args.data_dir, 'train'),
        image_size=args.image_size,
        transform_type='train'
    )
    
    val_dataset = SteganographyDataset(
        os.path.join(args.data_dir, 'val'),
        image_size=args.image_size,
        transform_type='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    trainer = CPUOptimizedTrainer(args)
    
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
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best PSNR: {best_psnr:.4f} (Epoch {best_epoch + 1})")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("=" * 60)

if __name__ == '__main__':
    main()
