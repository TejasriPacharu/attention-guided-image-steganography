import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLNReLU(nn.Module):
    """Basic building block: Conv + LayerNorm + LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln = nn.LayerNorm([out_channels])
        self.relu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        # LayerNorm expects (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # Back to (N, C, H, W)
        x = self.relu(x)
        return x

class NonLinearEnhancement(nn.Module):
    """Non-Linear Enhancement (NLE) layer with dual-branch structure"""
    
    def __init__(self, channels):
        super(NonLinearEnhancement, self).__init__()
        self.ln = nn.LayerNorm([channels])
        
        # Dual branch structure
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU()
        )
        
        self.output_conv = nn.Conv2d(channels * 2, channels, 1)
    
    def forward(self, x):
        # Apply layer normalization
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        # Dual branch processing
        branch1_out = self.branch1(x_norm)
        branch2_out = self.branch2(x_norm)
        
        # Element-wise multiplication and output
        combined = branch1_out * branch2_out
        output = self.output_conv(combined)
        
        return output

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention using depthwise separable convolutions"""
    
    def __init__(self, channels, reduction_ratio=4):
        super(MemoryEfficientAttention, self).__init__()
        self.channels = channels
        self.ln = nn.LayerNorm([channels])
        
        # Use depthwise separable convolutions instead of full attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels // reduction_ratio, 1),           # Pointwise
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        # Layer normalization
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        # Spatial attention
        spatial_att = self.spatial_attention(x_norm)
        
        # Channel attention
        channel_att = self.channel_attention(x_norm)
        
        # Apply attention
        attended = x_norm * spatial_att * channel_att
        
        # Output projection and residual connection
        out = self.output_conv(attended)
        return out + x

class ChannelSelfAttentionModule(nn.Module):
    """Memory-efficient Channel Self-Attention Module (CSAM)"""
    
    def __init__(self, channels):
        super(ChannelSelfAttentionModule, self).__init__()
        self.attention = MemoryEfficientAttention(channels)
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, x):
        # Memory-efficient attention
        x_att = self.attention(x)
        
        # Non-linear enhancement with residual connection
        x_enhanced = self.nle(x_att) + x_att
        
        return x_enhanced

class MemoryEfficientCrossAttention(nn.Module):
    """Memory-efficient cross attention using convolutions"""
    
    def __init__(self, channels):
        super(MemoryEfficientCrossAttention, self).__init__()
        self.channels = channels
        self.ln_cover = nn.LayerNorm([channels])
        self.ln_secret = nn.LayerNorm([channels])
        
        # Use convolutions instead of matrix multiplication
        self.cover_proj = nn.Conv2d(channels, channels, 3, padding=1)
        self.secret_proj = nn.Conv2d(channels, channels, 3, padding=1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, cover_features, secret_features):
        # Layer normalization
        cover_norm = cover_features.permute(0, 2, 3, 1)
        cover_norm = self.ln_cover(cover_norm).permute(0, 3, 1, 2)
        
        secret_norm = secret_features.permute(0, 2, 3, 1)
        secret_norm = self.ln_secret(secret_norm).permute(0, 3, 1, 2)
        
        # Project features
        cover_proj = self.cover_proj(cover_norm)
        secret_proj = self.secret_proj(secret_norm)
        
        # Concatenate and fuse
        combined = torch.cat([cover_proj, secret_proj], dim=1)
        fused = self.fusion(combined)
        
        # Residual connection
        return fused + cover_features

class ChannelCrossAttentionModule(nn.Module):
    """Memory-efficient Channel-wise Cross Attention Module (CCAM)"""
    
    def __init__(self, channels):
        super(ChannelCrossAttentionModule, self).__init__()
        self.cross_attention = MemoryEfficientCrossAttention(channels)
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, cover_features, secret_features):
        # Cross attention
        cover_adjusted = self.cross_attention(cover_features, secret_features)
        
        # Non-linear enhancement
        cover_enhanced = self.nle(cover_adjusted) + cover_adjusted
        
        return cover_enhanced

class GlobalLocalAggregationModule(nn.Module):
    """Memory-efficient Global-Local Aggregation Module (GLAM)"""
    
    def __init__(self, channels):
        super(GlobalLocalAggregationModule, self).__init__()
        
        # Global context using adaptive pooling
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Local features using depthwise convolution
        self.local_features = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, x):
        # Global context
        global_att = self.global_context(x)
        
        # Local features
        local_feat = self.local_features(x)
        
        # Combine global and local
        combined = local_feat * global_att + x
        
        # Non-linear enhancement
        enhanced = self.nle(combined) + combined
        
        return enhanced

class CAISFormerBlock(nn.Module):
    """Memory-efficient CAISFormer block"""
    
    def __init__(self, channels, num_heads=4):
        super(CAISFormerBlock, self).__init__()
        self.conv_ln_relu = ConvLNReLU(channels, channels)
        self.csam = ChannelSelfAttentionModule(channels)
        self.ccam = ChannelCrossAttentionModule(channels)
        self.glam = GlobalLocalAggregationModule(channels)
    
    def forward(self, cover_features, secret_features=None):
        # Initial feature processing
        cover_features = self.conv_ln_relu(cover_features)
        
        # Self-attention on cover features
        cover_features = self.csam(cover_features)
        
        # Cross-attention if secret features provided
        if secret_features is not None:
            secret_features = self.conv_ln_relu(secret_features)
            secret_features = self.csam(secret_features)
            cover_features = self.ccam(cover_features, secret_features)
        
        # Global-local aggregation
        output = self.glam(cover_features)
        
        return output