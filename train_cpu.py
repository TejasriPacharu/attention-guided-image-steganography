#!/usr/bin/env python3
"""
CPU-Optimized Training Script for Attention-Guided Steganography
Uses memory-efficient transformer to avoid OOM errors
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import gc

# Import memory-efficient modules
from models.attention_heatmap import AttentionHeatmapGenerator
from models.transformer_lite import CAISFormerBlock, ConvLNReLU
from models.discriminator import SRNetDiscriminator
from utils.dataset import SteganographyDataset
from utils.metrics import compute_metrics

class MemoryEfficientSteganography(nn.Module):
    """Memory-efficient version of attention-guided steganography"""
    
    def __init__(self, input_channels=3, hidden_channels=32):
        super(MemoryEfficientSteganography, self).__init__()
        
        # Simplified attention generators
        self.cover_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels//2)
        self.secret_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels//2)
        
        # Lightweight encoders
        self.cover_encoder = nn.Sequential(
            ConvLNReLU(input_channels, hidden_channels//2),
            ConvLNReLU(hidden_channels//2, hidden_channels)
        )
        
        self.secret_encoder = nn.Sequential(
            ConvLNReLU(input_channels, hidden_channels//2),
            ConvLNReLU(hidden_channels//2, hidden_channels)
        )
        
        # Single CAISFormer block (reduced from 3)
        self.caisformer_block = CAISFormerBlock(hidden_channels)
        
        # Simplified attention modulator
        self.attention_modulator = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            ConvLNReLU(hidden_channels * 2, hidden_channels),
            nn.Conv2d(hidden_channels, input_channels, 3, padding=1)
        )
        
        # Extraction network (simplified)
        self.extraction_network = nn.Sequential(
            ConvLNReLU(input_channels, hidden_channels//2),
            ConvLNReLU(hidden_channels//2, hidden_channels),
            CAISFormerBlock(hidden_channels),
            ConvLNReLU(hidden_channels, hidden_channels//2),
            nn.Conv2d(hidden_channels//2, input_channels, 3, padding=1)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, cover_image, secret_image, mode='train', embedding_strategy='adaptive'):
        batch_size, channels, height, width = cover_image.shape
        
        # Generate attention maps (simplified)
        with torch.no_grad():  # Reduce memory usage
            cover_attention = self.cover_attention_gen(cover_image)
            secret_attention = self.secret_attention_gen(secret_image)
        
        # Compute embedding strategy
        if embedding_strategy == 'adaptive':
            embedding_map = (cover_attention['embedding_attention'] + 
                           secret_attention['embedding_attention']) / 2
        else:
            embedding_map = cover_attention['embedding_attention']
        
        # Feature extraction
        cover_features = self.cover_encoder(cover_image)
        secret_features = self.secret_encoder(secret_image)
        
        # Single transformer block
        cover_features = self.caisformer_block(cover_features, secret_features)
        
        # Attention-guided modulation
        modulated_features = torch.cat([cover_features, embedding_map], dim=1)
        modulated_features = self.attention_modulator(modulated_features)
        
        # Feature fusion
        fused_features = torch.cat([modulated_features, secret_features], dim=1)
        stego_residual = self.feature_fusion(fused_features)
        
        # Generate stego image
        stego_image = cover_image + self.residual_weight * stego_residual
        stego_image = torch.clamp(stego_image, 0, 1)
        
        # Extract secret
        extracted_secret = self.extraction_network(stego_image)
        extracted_secret = torch.clamp(extracted_secret, 0, 1)
        
        results = {
            'stego_image': stego_image,
            'extracted_secret': extracted_secret,
            'cover_attention': cover_attention,
            'secret_attention': secret_attention,
            'embedding_map': embedding_map
        }
        
        return results

class CPUTrainer:
    """CPU-optimized trainer with memory management"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        
        # Create models
        self.generator = MemoryEfficientSteganography(
            input_channels=args.input_channels,
            hidden_channels=args.hidden_channels
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
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
    
    def train_step(self, cover_images, secret_images):
        """Single training step with memory management"""
        
        # Clear cache
        gc.collect()
        
        batch_size = cover_images.size(0)
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        gen_results = self.generator(
            cover_images, secret_images, 
            mode='train', 
            embedding_strategy=self.args.embedding_strategy
        )
        
        # Generator losses
        cover_loss = self.mse_loss(gen_results['stego_image'], cover_images)
        secret_loss = self.mse_loss(gen_results['extracted_secret'], secret_images)
        
        # Discriminator loss (simplified)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        disc_fake = self.discriminator(gen_results['stego_image'].detach())
        disc_real = self.discriminator(cover_images)
        
        disc_loss = (self.bce_loss(disc_real, real_labels) + 
                    self.bce_loss(disc_fake, fake_labels)) / 2
        
        # Adversarial loss for generator
        disc_fake_gen = self.discriminator(gen_results['stego_image'])
        adv_loss = self.bce_loss(disc_fake_gen, real_labels)
        
        # Total generator loss
        gen_loss = (self.args.cover_loss_weight * cover_loss + 
                   self.args.secret_loss_weight * secret_loss + 
                   self.args.adversarial_loss_weight * adv_loss)
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Clear intermediate results
        del gen_results
        gc.collect()
        
        return {
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'cover_loss': cover_loss.item(),
            'secret_loss': secret_loss.item(),
            'adv_loss': adv_loss.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_losses = {'gen_loss': 0, 'disc_loss': 0, 'cover_loss': 0, 'secret_loss': 0, 'adv_loss': 0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover_images, secret_images) in enumerate(pbar):
            cover_images = cover_images.to(self.device)
            secret_images = secret_images.to(self.device)
            
            # Training step
            losses = self.train_step(cover_images, secret_images)
            
            # Update totals
            for key in total_losses:
                total_losses[key] += losses[key]
            
            # Update progress bar
            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    'Gen': f"{losses['gen_loss']:.4f}",
                    'Disc': f"{losses['disc_loss']:.4f}",
                    'Cover': f"{losses['cover_loss']:.4f}",
                    'Secret': f"{losses['secret_loss']:.4f}"
                })
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def save_checkpoint(self, epoch, losses, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'losses': losses
        }
        
        filename = f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.args.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
        
        print(f"Checkpoint saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='CPU-Optimized Attention-Guided Steganography Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (small for CPU)')
    parser.add_argument('--image_size', type=int, default=64, help='Image size (small for CPU)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Hidden channels (very small)')
    parser.add_argument('--embedding_strategy', type=str, default='adaptive', help='Embedding strategy')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs (reduced)')
    parser.add_argument('--gen_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--disc_lr', type=float, default=1e-3, help='Discriminator learning rate')
    
    # Loss weights
    parser.add_argument('--cover_loss_weight', type=float, default=1.0, help='Cover loss weight')
    parser.add_argument('--secret_loss_weight', type=float, default=1.0, help='Secret loss weight')
    parser.add_argument('--adversarial_loss_weight', type=float, default=0.01, help='Adversarial loss weight')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cpu', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=5, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 50)
    print("CPU-Optimized Training Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image Size: {args.image_size}x{args.image_size}")
    print(f"  Hidden Channels: {args.hidden_channels}")
    print(f"  Epochs: {args.num_epochs}")
    print("=" * 50)
    
    # Create datasets
    train_dataset = SteganographyDataset(
        os.path.join(args.data_dir, 'train'),
        image_size=args.image_size,
        transform_type='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # Disable for CPU
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create trainer
    trainer = CPUTrainer(args)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Print losses
        print(f"Train Losses - Gen: {train_losses['gen_loss']:.4f}, "
              f"Disc: {train_losses['disc_loss']:.4f}, "
              f"Cover: {train_losses['cover_loss']:.4f}, "
              f"Secret: {train_losses['secret_loss']:.4f}")
        
        # Save checkpoint
        is_best = train_losses['gen_loss'] < best_loss
        if is_best:
            best_loss = train_losses['gen_loss']
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            trainer.save_checkpoint(epoch, train_losses, is_best)
        
        # Memory cleanup
        gc.collect()
    
    print("\nTraining completed!")
    print(f"Best model saved in: {args.checkpoint_dir}/best_model.pth")

if __name__ == '__main__':
    main()
