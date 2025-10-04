import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict

class SpatialAttention(nn.Module):
    """Spatial attention mechanism to highlight important spatial regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_input)
        
        return self.sigmoid(attention_map)

class ChannelAttention(nn.Module):
    """Channel attention mechanism to highlight important feature channels"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return self.sigmoid(attention)

class GradCAM:
    """Gradient-based Class Activation Mapping for attention visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam

class AttentionHeatmapGenerator(nn.Module):
    """
    Comprehensive attention heatmap generator for image steganography.
    Generates multiple types of attention maps to guide embedding process.
    """
    
    def __init__(self, input_channels=3, feature_channels=64):
        super(AttentionHeatmapGenerator, self).__init__()
        
        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(feature_channels)
        
        # Multi-scale attention
        self.multiscale_conv1 = nn.Conv2d(feature_channels, feature_channels//2, 1)
        self.multiscale_conv3 = nn.Conv2d(feature_channels, feature_channels//2, 3, padding=1)
        self.multiscale_conv5 = nn.Conv2d(feature_channels, feature_channels//2, 5, padding=2)
        
        # Final attention map generation
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Texture analysis for embedding suitability
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def compute_edge_attention(self, x):
        """Compute edge-based attention using Sobel operators"""
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        edge_attention = torch.sigmoid(edge_magnitude)
        
        return edge_attention
    
    def compute_texture_complexity(self, x):
        """Compute texture complexity for embedding suitability"""
        # Local binary pattern approximation
        kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], 
                             dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        texture_response = F.conv2d(gray, kernel, padding=1)
        texture_complexity = torch.sigmoid(torch.abs(texture_response))
        
        return texture_complexity
    
    def forward(self, image, return_intermediate=False):
        """
        Generate comprehensive attention heatmaps for an input image
        
        Args:
            image: Input image tensor [B, C, H, W]
            return_intermediate: Whether to return intermediate attention maps
            
        Returns:
            Dict containing various attention maps
        """
        batch_size, channels, height, width = image.shape
        
        # Extract features
        features = self.feature_extractor(image)
        
        # Spatial attention
        spatial_att = self.spatial_attention(features)
        
        # Channel attention
        channel_att = self.channel_attention(features)
        channel_weighted_features = features * channel_att
        
        # Multi-scale attention
        ms_feat1 = self.multiscale_conv1(features)
        ms_feat3 = self.multiscale_conv3(features)
        ms_feat5 = self.multiscale_conv5(features)
        multiscale_features = torch.cat([ms_feat1, ms_feat3, ms_feat5], dim=1)
        
        # Edge-based attention
        edge_attention = self.compute_edge_attention(image)
        
        # Texture complexity
        texture_complexity = self.compute_texture_complexity(image)
        
        # Combine all attention mechanisms
        combined_features = torch.cat([
            channel_weighted_features,
            multiscale_features
        ], dim=1)
        
        # Generate final attention map
        final_attention = self.attention_fusion(combined_features)
        
        # Texture suitability map
        texture_suitability = self.texture_analyzer(image)
        
        # Combine attention maps for embedding guidance
        embedding_attention = (final_attention * 0.4 + 
                             spatial_att * 0.3 + 
                             edge_attention * 0.2 + 
                             texture_complexity * 0.1)
        
        # Normalize to [0, 1]
        embedding_attention = (embedding_attention - embedding_attention.min()) / (
            embedding_attention.max() - embedding_attention.min() + 1e-8)
        
        attention_maps = {
            'final_attention': final_attention,
            'embedding_attention': embedding_attention,
            'spatial_attention': spatial_att,
            'edge_attention': edge_attention,
            'texture_complexity': texture_complexity,
            'texture_suitability': texture_suitability
        }
        
        if return_intermediate:
            attention_maps.update({
                'channel_attention': channel_att,
                'features': features,
                'channel_weighted_features': channel_weighted_features
            })
        
        return attention_maps
    
    def visualize_attention_maps(self, image, attention_maps, save_path=None):
        """Visualize attention maps for analysis"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original image
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            if img_np.shape[2] == 3:
                img_np = np.clip(img_np, 0, 1)
        else:
            img_np = image
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Attention maps
        attention_names = ['final_attention', 'embedding_attention', 
                          'spatial_attention', 'edge_attention',
                          'texture_complexity', 'texture_suitability']
        
        for i, name in enumerate(attention_names):
            if name in attention_maps:
                row = i // 4
                col = (i + 1) % 4
                
                att_map = attention_maps[name].squeeze().cpu().numpy()
                im = axes[row, col].imshow(att_map, cmap='hot', interpolation='nearest')
                axes[row, col].set_title(name.replace('_', ' ').title())
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Remove empty subplots
        for i in range(len(attention_names) + 1, 8):
            row = i // 4
            col = i % 4
            axes[row, col].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_embedding_regions(self, attention_map, threshold=0.5, region_type='high'):
        """
        Get regions suitable for embedding based on attention maps
        
        Args:
            attention_map: Attention map tensor
            threshold: Threshold for region selection
            region_type: 'high' for high-attention regions, 'low' for low-attention regions
            
        Returns:
            Binary mask indicating embedding regions
        """
        if region_type == 'high':
            mask = (attention_map > threshold).float()
        else:
            mask = (attention_map < threshold).float()
        
        return mask
