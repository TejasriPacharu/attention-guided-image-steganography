import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

class SteganographyDataset(Dataset):
    """
    Dataset for steganography training with cover and secret image pairs
    """
    
    def __init__(self, data_dir, image_size=256, mode='train', pair_mode='random'):
        """
        Args:
            data_dir: Directory containing images
            image_size: Size to resize images to
            mode: 'train' or 'val' or 'test'
            pair_mode: 'random' (random pairing) or 'fixed' (fixed pairing)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        self.pair_mode = pair_mode
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend([
                f for f in os.listdir(data_dir) 
                if f.lower().endswith(ext)
            ])
        
        self.image_files.sort()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # Define transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load cover image
        cover_path = os.path.join(self.data_dir, self.image_files[idx])
        cover_image = Image.open(cover_path).convert('RGB')
        
        # Load secret image
        if self.pair_mode == 'random':
            # Randomly select a different image as secret
            secret_idx = random.choice([i for i in range(len(self.image_files)) if i != idx])
        else:
            # Use next image as secret (with wraparound)
            secret_idx = (idx + 1) % len(self.image_files)
        
        secret_path = os.path.join(self.data_dir, self.image_files[secret_idx])
        secret_image = Image.open(secret_path).convert('RGB')
        
        # Apply transforms
        cover_tensor = self.transform(cover_image)
        secret_tensor = self.transform(secret_image)
        
        # Normalize to [0, 1] range for the model
        cover_tensor = (cover_tensor + 1.0) / 2.0
        secret_tensor = (secret_tensor + 1.0) / 2.0
        
        return cover_tensor, secret_tensor

class PairedSteganographyDataset(Dataset):
    """
    Dataset for steganography with predefined cover-secret pairs
    """
    
    def __init__(self, cover_dir, secret_dir, image_size=256, mode='train'):
        """
        Args:
            cover_dir: Directory containing cover images
            secret_dir: Directory containing secret images
            image_size: Size to resize images to
            mode: 'train' or 'val' or 'test'
        """
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.image_size = image_size
        self.mode = mode
        
        # Get image files from both directories
        cover_files = set([
            f for f in os.listdir(cover_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        secret_files = set([
            f for f in os.listdir(secret_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        # Find common files
        self.common_files = sorted(list(cover_files.intersection(secret_files)))
        
        if len(self.common_files) == 0:
            raise ValueError("No common files found between cover and secret directories")
        
        print(f"Found {len(self.common_files)} paired images")
        
        # Define transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.common_files)
    
    def __getitem__(self, idx):
        filename = self.common_files[idx]
        
        # Load cover and secret images
        cover_path = os.path.join(self.cover_dir, filename)
        secret_path = os.path.join(self.secret_dir, filename)
        
        cover_image = Image.open(cover_path).convert('RGB')
        secret_image = Image.open(secret_path).convert('RGB')
        
        # Apply transforms
        cover_tensor = self.transform(cover_image)
        secret_tensor = self.transform(secret_image)
        
        return cover_tensor, secret_tensor

class CelebASteganographyDataset(Dataset):
    """
    Specialized dataset for CelebA images with face-aware pairing
    """
    
    def __init__(self, data_dir, image_size=256, mode='train', same_identity=False):
        """
        Args:
            data_dir: Directory containing CelebA images
            image_size: Size to resize images to
            mode: 'train' or 'val' or 'test'
            same_identity: If True, pair images of same identity when possible
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        self.same_identity = same_identity
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_files.sort()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_files)} CelebA images")
        
        # Group by identity if same_identity is True
        if same_identity:
            self.identity_groups = self._group_by_identity()
        
        # Define transforms with face-specific augmentations
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
    
    def _group_by_identity(self):
        """Group images by identity (assuming filename format contains identity info)"""
        groups = {}
        for filename in self.image_files:
            # Extract identity from filename (customize based on your naming convention)
            identity = filename.split('_')[0] if '_' in filename else filename[:6]
            if identity not in groups:
                groups[identity] = []
            groups[identity].append(filename)
        
        # Filter groups with multiple images
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        cover_filename = self.image_files[idx]
        cover_path = os.path.join(self.data_dir, cover_filename)
        cover_image = Image.open(cover_path).convert('RGB')
        
        # Select secret image
        if self.same_identity and hasattr(self, 'identity_groups'):
            # Try to find same identity
            identity = cover_filename.split('_')[0] if '_' in cover_filename else cover_filename[:6]
            if identity in self.identity_groups and len(self.identity_groups[identity]) > 1:
                secret_candidates = [f for f in self.identity_groups[identity] if f != cover_filename]
                secret_filename = random.choice(secret_candidates)
            else:
                secret_filename = random.choice([f for f in self.image_files if f != cover_filename])
        else:
            secret_filename = random.choice([f for f in self.image_files if f != cover_filename])
        
        secret_path = os.path.join(self.data_dir, secret_filename)
        secret_image = Image.open(secret_path).convert('RGB')
        
        # Apply transforms
        cover_tensor = self.transform(cover_image)
        secret_tensor = self.transform(secret_image)
        
        return cover_tensor, secret_tensor

def create_dataset(dataset_type, **kwargs):
    """Factory function to create appropriate dataset"""
    if dataset_type == 'standard':
        return SteganographyDataset(**kwargs)
    elif dataset_type == 'paired':
        return PairedSteganographyDataset(**kwargs)
    elif dataset_type == 'celeba':
        return CelebASteganographyDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Data preparation utilities
def prepare_data_splits(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test sets
    
    Args:
        data_dir: Directory containing all images
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    import shutil
    
    # Get all image files
    image_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Create split directories
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    for split_name, files in splits.items():
        split_dir = os.path.join(os.path.dirname(data_dir), split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for filename in files:
            src = os.path.join(data_dir, filename)
            dst = os.path.join(split_dir, filename)
            shutil.copy2(src, dst)
        
        print(f"Created {split_name} set with {len(files)} images")

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare steganography dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--split', action='store_true', help='Split data into train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    
    args = parser.parse_args()
    
    if args.split:
        prepare_data_splits(args.data_dir, args.train_ratio, args.val_ratio, args.test_ratio)
