import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
import lpips

def psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image (numpy array or torch tensor)
        img2: Second image (numpy array or torch tensor)
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in [0, max_val] range
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr_val = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr_val

def ssim(img1, img2, max_val=1.0, multichannel=True):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image (numpy array or torch tensor)
        img2: Second image (numpy array or torch tensor)
        max_val: Maximum possible pixel value
        multichannel: Whether to treat as multichannel image
        
    Returns:
        SSIM value between -1 and 1
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in [0, max_val] range
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    # Handle batch dimension
    if len(img1.shape) == 4:  # Batch of images
        ssim_vals = []
        for i in range(img1.shape[0]):
            if multichannel and img1.shape[1] == 3:
                # Convert from CHW to HWC
                im1 = np.transpose(img1[i], (1, 2, 0))
                im2 = np.transpose(img2[i], (1, 2, 0))
                ssim_val = ssim_skimage(im1, im2, data_range=max_val, multichannel=True, channel_axis=-1)
            else:
                ssim_val = ssim_skimage(img1[i], img2[i], data_range=max_val)
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    else:
        if multichannel and len(img1.shape) == 3 and img1.shape[0] == 3:
            # Convert from CHW to HWC
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            return ssim_skimage(img1, img2, data_range=max_val, multichannel=True, channel_axis=-1)
        else:
            return ssim_skimage(img1, img2, data_range=max_val)

def lpips_distance(img1, img2, net='alex', device='cuda'):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) distance
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W]
        img2: Second image tensor [B, C, H, W] or [C, H, W]
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to run computation on
        
    Returns:
        LPIPS distance (lower is better)
    """
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net).to(device)
    
    if not torch.is_tensor(img1):
        img1 = torch.tensor(img1).to(device)
    if not torch.is_tensor(img2):
        img2 = torch.tensor(img2).to(device)
    
    # Ensure tensors are on the right device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Normalize to [-1, 1] range for LPIPS
    img1 = img1 * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0
    
    with torch.no_grad():
        distance = lpips_model(img1, img2)
    
    return distance.mean().item()

def mse(img1, img2):
    """Calculate Mean Squared Error"""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    return np.mean((img1 - img2) ** 2)

def mae(img1, img2):
    """Calculate Mean Absolute Error"""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    return np.mean(np.abs(img1 - img2))

def ncc(img1, img2):
    """Calculate Normalized Cross Correlation"""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Flatten images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0

def histogram_similarity(img1, img2, bins=256):
    """
    Calculate histogram similarity using correlation
    
    Args:
        img1: First image
        img2: Second image
        bins: Number of histogram bins
        
    Returns:
        Histogram correlation coefficient
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Convert to uint8 for histogram calculation
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)
    
    # Calculate histograms
    if len(img1.shape) == 3 and img1.shape[0] == 3:  # RGB image
        hist1 = cv2.calcHist([img1_uint8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_uint8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    else:  # Grayscale
        hist1 = cv2.calcHist([img1_uint8], [0], None, [bins], [0, 256])
        hist2 = cv2.calcHist([img2_uint8], [0], None, [bins], [0, 256])
    
    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation

def compute_metrics(cover_images, secret_images, stego_images, extracted_secrets, device='cuda'):
    """
    Compute comprehensive metrics for steganography evaluation
    
    Args:
        cover_images: Original cover images
        secret_images: Original secret images  
        stego_images: Generated stego images
        extracted_secrets: Extracted secret images
        device: Device for LPIPS computation
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Cover preservation metrics (cover vs stego)
    metrics['cover_psnr'] = psnr(cover_images, stego_images)
    metrics['cover_ssim'] = ssim(cover_images, stego_images)
    metrics['cover_mse'] = mse(cover_images, stego_images)
    metrics['cover_mae'] = mae(cover_images, stego_images)
    
    # Secret reconstruction metrics (secret vs extracted)
    metrics['secret_psnr'] = psnr(secret_images, extracted_secrets)
    metrics['secret_ssim'] = ssim(secret_images, extracted_secrets)
    metrics['secret_mse'] = mse(secret_images, extracted_secrets)
    metrics['secret_mae'] = mae(secret_images, extracted_secrets)
    
    # Overall quality metrics
    metrics['psnr'] = (metrics['cover_psnr'] + metrics['secret_psnr']) / 2
    metrics['ssim'] = (metrics['cover_ssim'] + metrics['secret_ssim']) / 2
    
    # Perceptual metrics (if device available)
    try:
        if torch.cuda.is_available() or device == 'cpu':
            metrics['cover_lpips'] = lpips_distance(
                torch.tensor(cover_images), 
                torch.tensor(stego_images), 
                device=device
            )
            metrics['secret_lpips'] = lpips_distance(
                torch.tensor(secret_images), 
                torch.tensor(extracted_secrets), 
                device=device
            )
    except Exception as e:
        print(f"LPIPS computation failed: {e}")
        metrics['cover_lpips'] = 0.0
        metrics['secret_lpips'] = 0.0
    
    # Histogram similarity
    metrics['cover_hist_sim'] = histogram_similarity(cover_images, stego_images)
    metrics['secret_hist_sim'] = histogram_similarity(secret_images, extracted_secrets)
    
    # Normalized cross correlation
    metrics['cover_ncc'] = ncc(cover_images, stego_images)
    metrics['secret_ncc'] = ncc(secret_images, extracted_secrets)
    
    return metrics

def compute_batch_metrics(cover_batch, secret_batch, stego_batch, extracted_batch):
    """
    Compute metrics for a batch of images
    
    Args:
        cover_batch: Batch of cover images [B, C, H, W]
        secret_batch: Batch of secret images [B, C, H, W]
        stego_batch: Batch of stego images [B, C, H, W]
        extracted_batch: Batch of extracted images [B, C, H, W]
        
    Returns:
        Dictionary of averaged metrics
    """
    batch_size = cover_batch.shape[0]
    
    # Initialize metric accumulators
    total_metrics = {
        'cover_psnr': 0.0, 'cover_ssim': 0.0,
        'secret_psnr': 0.0, 'secret_ssim': 0.0,
        'cover_mse': 0.0, 'secret_mse': 0.0
    }
    
    # Compute metrics for each image in batch
    for i in range(batch_size):
        cover_img = cover_batch[i].cpu().numpy()
        secret_img = secret_batch[i].cpu().numpy()
        stego_img = stego_batch[i].cpu().numpy()
        extracted_img = extracted_batch[i].cpu().numpy()
        
        # Individual metrics
        total_metrics['cover_psnr'] += psnr(cover_img, stego_img)
        total_metrics['cover_ssim'] += ssim(cover_img, stego_img)
        total_metrics['secret_psnr'] += psnr(secret_img, extracted_img)
        total_metrics['secret_ssim'] += ssim(secret_img, extracted_img)
        total_metrics['cover_mse'] += mse(cover_img, stego_img)
        total_metrics['secret_mse'] += mse(secret_img, extracted_img)
    
    # Average metrics
    avg_metrics = {k: v / batch_size for k, v in total_metrics.items()}
    
    # Overall metrics
    avg_metrics['psnr'] = (avg_metrics['cover_psnr'] + avg_metrics['secret_psnr']) / 2
    avg_metrics['ssim'] = (avg_metrics['cover_ssim'] + avg_metrics['secret_ssim']) / 2
    
    return avg_metrics

class MetricsTracker:
    """Class to track metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics = {
            'cover_psnr': [], 'cover_ssim': [], 'cover_mse': [],
            'secret_psnr': [], 'secret_ssim': [], 'secret_mse': [],
            'psnr': [], 'ssim': []
        }
    
    def update(self, batch_metrics):
        """Update metrics with batch results"""
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_averages(self):
        """Get average metrics"""
        return {key: np.mean(values) for key, values in self.metrics.items() if values}
    
    def get_latest(self):
        """Get latest metric values"""
        return {key: values[-1] if values else 0.0 for key, values in self.metrics.items()}

# Utility functions for specific evaluations
def evaluate_imperceptibility(cover_images, stego_images):
    """Evaluate imperceptibility of stego images"""
    return {
        'psnr': psnr(cover_images, stego_images),
        'ssim': ssim(cover_images, stego_images),
        'mse': mse(cover_images, stego_images),
        'mae': mae(cover_images, stego_images)
    }

def evaluate_extraction_quality(secret_images, extracted_images):
    """Evaluate quality of extracted secret images"""
    return {
        'psnr': psnr(secret_images, extracted_images),
        'ssim': ssim(secret_images, extracted_images),
        'mse': mse(secret_images, extracted_images),
        'mae': mae(secret_images, extracted_images)
    }

def compare_attention_strategies(results_dict):
    """
    Compare different attention strategies
    
    Args:
        results_dict: Dictionary with strategy names as keys and results as values
        
    Returns:
        Comparison metrics
    """
    comparison = {}
    
    for strategy, results in results_dict.items():
        metrics = compute_metrics(
            results['cover_images'],
            results['secret_images'], 
            results['stego_images'],
            results['extracted_secrets']
        )
        comparison[strategy] = metrics
    
    return comparison
