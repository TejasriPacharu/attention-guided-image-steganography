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

class ChannelTransposeAttention(nn.Module):
    """Channel Transpose Attention (CTA) layer"""
    
    def __init__(self, channels, reduction_ratio=1):
        super(ChannelTransposeAttention, self).__init__()
        self.channels = channels
        self.ln = nn.LayerNorm([channels])
        
        # Query, Key, Value projections
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        self.output_conv = nn.Conv2d(channels, channels, 1)
        self.scale = math.sqrt(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Layer normalization
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        # Generate Q, K, V
        q = self.q_conv(x_norm)  # [B, C, H, W]
        k = self.k_conv(x_norm)  # [B, C, H, W]
        v = self.v_conv(x_norm)  # [B, C, H, W]
        
        # Reshape for channel-wise attention
        q = q.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.view(B, C, H * W)                    # [B, C, HW]
        v = v.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # Channel-wise attention computation
        attention = torch.matmul(q, k) / self.scale  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        
        # Output projection and residual connection
        out = self.output_conv(out)
        return out + x

class ChannelSelfAttentionModule(nn.Module):
    """Channel Self-Attention Module (CSAM)"""
    
    def __init__(self, channels):
        super(ChannelSelfAttentionModule, self).__init__()
        self.cta = ChannelTransposeAttention(channels)
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, x):
        # Channel transpose attention
        x_att = self.cta(x)
        
        # Non-linear enhancement with residual connection
        x_enhanced = self.nle(x_att) + x_att
        
        return x_enhanced

class ChannelCrossAttention(nn.Module):
    """Channel-wise Cross Attention (CCA) layer"""
    
    def __init__(self, channels):
        super(ChannelCrossAttention, self).__init__()
        self.channels = channels
        self.ln_cover = nn.LayerNorm([channels])
        self.ln_secret = nn.LayerNorm([channels])
        
        # Projections for cover image (K, V)
        self.k_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Projection for secret image (Q)
        self.q_conv = nn.Conv2d(channels, channels, 1)
        
        self.output_conv = nn.Conv2d(channels, channels, 1)
        self.scale = math.sqrt(channels)
    
    def forward(self, cover_features, secret_features):
        B, C, H, W = cover_features.shape
        
        # Layer normalization
        cover_norm = cover_features.permute(0, 2, 3, 1)
        cover_norm = self.ln_cover(cover_norm).permute(0, 3, 1, 2)
        
        secret_norm = secret_features.permute(0, 2, 3, 1)
        secret_norm = self.ln_secret(secret_norm).permute(0, 3, 1, 2)
        
        # Generate K, V from cover and Q from secret
        k = self.k_conv(cover_norm)  # [B, C, H, W]
        v = self.v_conv(cover_norm)  # [B, C, H, W]
        q = self.q_conv(secret_norm) # [B, C, H, W]
        
        # Reshape for cross-attention
        q = q.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.view(B, C, H * W)                    # [B, C, HW]
        v = v.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # Cross-attention computation
        attention = torch.matmul(q, k) / self.scale  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        
        # Output projection and residual connection
        out = self.output_conv(out)
        return out + cover_features

class ChannelCrossAttentionModule(nn.Module):
    """Channel-wise Cross Attention Module (CCAM)"""
    
    def __init__(self, channels):
        super(ChannelCrossAttentionModule, self).__init__()
        self.cca = ChannelCrossAttention(channels)
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, cover_features, secret_features):
        # Cross attention
        cover_adjusted = self.cca(cover_features, secret_features)
        
        # Non-linear enhancement
        cover_enhanced = self.nle(cover_adjusted) + cover_adjusted
        
        return cover_enhanced

class GlobalLocalAttention(nn.Module):
    """Global-Local Attention (GLA) layer"""
    
    def __init__(self, channels, reduction_ratio1=4, reduction_ratio2=8):
        super(GlobalLocalAttention, self).__init__()
        self.channels = channels
        
        # Projections for global attention
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Local attention (depth-wise convolution)
        self.local_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Spatial and channel attention for fusion
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio1, channels, 1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio2, channels, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv2d(channels, channels, 1)
        self.scale = math.sqrt(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global attention computation
        q = self.q_conv(x).view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = self.k_conv(x).view(B, C, H * W)                   # [B, C, HW]
        v = self.v_conv(x).view(B, C, H * W).permute(0, 2, 1) # [B, HW, C]
        
        global_attention = torch.matmul(q, k) / self.scale
        global_attention = F.softmax(global_attention, dim=-1)
        global_out = torch.matmul(global_attention, v)
        global_out = global_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Local attention computation
        local_out = self.local_conv(v.permute(0, 2, 1).view(B, C, H, W))
        
        # Fusion with spatial and channel attention
        spatial_weight = self.spatial_att(local_out)
        channel_weight = self.channel_att(global_out)
        
        fused = global_out * channel_weight + local_out * spatial_weight
        
        # Output projection and residual connection
        output = self.output_conv(fused) + x
        
        return output

class GlobalLocalAggregationModule(nn.Module):
    """Global-Local Aggregation Module (GLAM)"""
    
    def __init__(self, channels):
        super(GlobalLocalAggregationModule, self).__init__()
        self.gla = GlobalLocalAttention(channels)
        self.nle = NonLinearEnhancement(channels)
    
    def forward(self, x):
        # Global-local attention
        x_att = self.gla(x)
        
        # Non-linear enhancement with residual connection
        x_enhanced = self.nle(x_att) + x_att
        
        return x_enhanced

class CAISFormerBlock(nn.Module):
    """Complete CAISFormer block with all attention modules"""
    
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
