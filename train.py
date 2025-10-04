import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models import AttentionGuidedSteganography, SRNetDiscriminator
from utils.dataset import SteganographyDataset
from utils.metrics import compute_metrics
from utils.visualization import save_attention_visualizations

class AttentionGuidedTrainer:
    """Trainer for attention-guided steganography system"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = AttentionGuidedSteganography(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels
        ).to(self.device)
        
        self.discriminator = SRNetDiscriminator(
            input_channels=config.input_channels
        ).to(self.device)
        
        # Initialize optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.gen_lr,
            betas=(0.5, 0.999)
        )
        
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.disc_lr,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Learning rate schedulers
        self.gen_scheduler = optim.lr_scheduler.StepLR(
            self.gen_optimizer, step_size=config.lr_step, gamma=config.lr_gamma
        )
        self.disc_scheduler = optim.lr_scheduler.StepLR(
            self.disc_optimizer, step_size=config.lr_step, gamma=config.lr_gamma
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        
        # Loss weights
        self.loss_weights = {
            'cover_loss': config.cover_loss_weight,
            'secret_loss': config.secret_loss_weight,
            'attention_loss': config.attention_loss_weight,
            'perceptual_loss': config.perceptual_loss_weight,
            'adversarial_loss': config.adversarial_loss_weight
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_cover_loss = 0.0
        total_secret_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover_images, secret_images) in enumerate(pbar):
            cover_images = cover_images.to(self.device)
            secret_images = secret_images.to(self.device)
            batch_size = cover_images.size(0)
            
            # =====================================
            # Train Generator
            # =====================================
            self.gen_optimizer.zero_grad()
            
            # Forward pass through generator
            gen_results = self.generator(
                cover_images, secret_images, 
                mode='train', 
                embedding_strategy=self.config.embedding_strategy
            )
            
            # Compute generator losses
            gen_losses = self.generator.compute_losses(
                gen_results, cover_images, secret_images, self.loss_weights
            )
            
            # Adversarial loss (fool discriminator)
            stego_pred = self.discriminator(gen_results['stego_image'])
            real_labels = torch.ones(batch_size, device=self.device)
            adversarial_loss = self.adversarial_loss(
                stego_pred['logits'][:, 1], real_labels
            )
            
            # Total generator loss
            total_gen_loss_batch = (
                gen_losses['total_loss'] + 
                self.loss_weights['adversarial_loss'] * adversarial_loss
            )
            
            total_gen_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.gen_optimizer.step()
            
            # =====================================
            # Train Discriminator
            # =====================================
            self.disc_optimizer.zero_grad()
            
            # Real images (cover images)
            real_pred = self.discriminator(cover_images)
            real_labels = torch.zeros(batch_size, device=self.device)
            real_loss = self.adversarial_loss(real_pred['logits'][:, 1], real_labels)
            
            # Fake images (stego images)
            fake_pred = self.discriminator(gen_results['stego_image'].detach())
            fake_labels = torch.ones(batch_size, device=self.device)
            fake_loss = self.adversarial_loss(fake_pred['logits'][:, 1], fake_labels)
            
            # Total discriminator loss
            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.disc_optimizer.step()
            
            # Update running losses
            total_gen_loss += total_gen_loss_batch.item()
            total_disc_loss += disc_loss.item()
            total_cover_loss += gen_losses['cover_loss'].item()
            total_secret_loss += gen_losses['secret_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Gen Loss': f'{total_gen_loss_batch.item():.4f}',
                'Disc Loss': f'{disc_loss.item():.4f}',
                'Cover Loss': f'{gen_losses["cover_loss"].item():.4f}',
                'Secret Loss': f'{gen_losses["secret_loss"].item():.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar('Train/Generator_Loss', total_gen_loss_batch.item(), global_step)
                self.writer.add_scalar('Train/Discriminator_Loss', disc_loss.item(), global_step)
                self.writer.add_scalar('Train/Cover_Loss', gen_losses['cover_loss'].item(), global_step)
                self.writer.add_scalar('Train/Secret_Loss', gen_losses['secret_loss'].item(), global_step)
                self.writer.add_scalar('Train/Attention_Loss', gen_losses['attention_loss'].item(), global_step)
        
        # Average losses for the epoch
        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_disc_loss = total_disc_loss / len(train_loader)
        avg_cover_loss = total_cover_loss / len(train_loader)
        avg_secret_loss = total_secret_loss / len(train_loader)
        
        return {
            'gen_loss': avg_gen_loss,
            'disc_loss': avg_disc_loss,
            'cover_loss': avg_cover_loss,
            'secret_loss': avg_secret_loss
        }
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_cover_psnr = 0.0
        total_secret_psnr = 0.0
        
        with torch.no_grad():
            for batch_idx, (cover_images, secret_images) in enumerate(val_loader):
                cover_images = cover_images.to(self.device)
                secret_images = secret_images.to(self.device)
                
                # Forward pass
                results = self.generator(
                    cover_images, secret_images, 
                    mode='train',
                    embedding_strategy=self.config.embedding_strategy
                )
                
                # Compute metrics
                metrics = compute_metrics(
                    cover_images.cpu().numpy(),
                    secret_images.cpu().numpy(),
                    results['stego_image'].cpu().numpy(),
                    results['extracted_secret'].cpu().numpy()
                )
                
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                total_cover_psnr += metrics['cover_psnr']
                total_secret_psnr += metrics['secret_psnr']
                
                # Save attention visualizations for first batch
                if batch_idx == 0:
                    save_attention_visualizations(
                        results, cover_images, secret_images,
                        os.path.join(self.config.output_dir, f'attention_epoch_{epoch}.png')
                    )
        
        # Average metrics
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_cover_psnr = total_cover_psnr / len(val_loader)
        avg_secret_psnr = total_secret_psnr / len(val_loader)
        
        # Log validation metrics
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, epoch)
        self.writer.add_scalar('Val/Cover_PSNR', avg_cover_psnr, epoch)
        self.writer.add_scalar('Val/Secret_PSNR', avg_secret_psnr, epoch)
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'cover_psnr': avg_cover_psnr,
            'secret_psnr': avg_secret_psnr
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'gen_scheduler_state_dict': self.gen_scheduler.state_dict(),
            'disc_scheduler_state_dict': self.disc_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'New best model saved with PSNR: {metrics["psnr"]:.4f}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
        self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['metrics']['psnr']
        
        print(f'Checkpoint loaded from epoch {self.current_epoch}')
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f'Starting training on {self.device}')
        print(f'Generator parameters: {sum(p.numel() for p in self.generator.parameters())}')
        print(f'Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}')
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update learning rates
            self.gen_scheduler.step()
            self.disc_scheduler.step()
            
            # Check if this is the best model
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print epoch summary
            print(f'Epoch {epoch}:')
            print(f'  Train - Gen Loss: {train_metrics["gen_loss"]:.4f}, '
                  f'Disc Loss: {train_metrics["disc_loss"]:.4f}')
            print(f'  Val - PSNR: {val_metrics["psnr"]:.4f}, '
                  f'SSIM: {val_metrics["ssim"]:.4f}')
            print(f'  Cover PSNR: {val_metrics["cover_psnr"]:.4f}, '
                  f'Secret PSNR: {val_metrics["secret_psnr"]:.4f}')
            
            self.current_epoch = epoch + 1

def main():
    parser = argparse.ArgumentParser(description='Train Attention-Guided Steganography')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--embedding_strategy', type=str, default='adaptive', 
                       choices=['high_low', 'low_high', 'adaptive'], help='Embedding strategy')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--gen_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--disc_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--lr_step', type=int, default=30, help='LR scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='LR scheduler gamma')
    
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
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create trainer
    trainer = AttentionGuidedTrainer(args)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
