import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attention_heatmap import AttentionHeatmapGenerator

class AttentionGuidedConvBlock(nn.Module):
    """Attention-guided convolution block for neural codec"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Attention gate
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        
        # Apply attention gating
        attention_weights = self.attention_gate(out)
        out = out * attention_weights
        
        return out

class AttentionTextureSynthesis(nn.Module):
    """Novel: Synthesize textures that hide secret information"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        # Texture synthesis network
        self.texture_synthesizer = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, 3, padding=1),  # cover + secret
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Attention-guided blending network
        self.blending_network = nn.Sequential(
            nn.Conv2d(input_channels + 1, 32, 3, padding=1),  # cover + attention
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover_image, secret_image, embedding_map):
        # Synthesize texture that encodes secret information
        texture_input = torch.cat([cover_image, secret_image], dim=1)
        synthesized_texture = self.texture_synthesizer(texture_input)
        
        # Compute attention-guided blending weights
        blend_input = torch.cat([cover_image, embedding_map], dim=1)
        blending_weights = self.blending_network(blend_input)
        
        # Adaptive blending based on attention
        # High attention = more texture synthesis
        # Low attention = preserve original cover
        stego_image = (
            blending_weights * synthesized_texture +
            (1 - blending_weights) * cover_image
        )
        
        return torch.clamp(stego_image, 0, 1)

class AttentionNeuralCodec(nn.Module):
    """Novel: End-to-end learnable codec with attention guidance"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Encoder: Cover + Secret + Attention → Stego
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2 + 1, 128, 3, padding=1),  # cover(3) + secret(3) + attention(1)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Attention-guided feature processing
            AttentionGuidedConvBlock(128, 128),
            AttentionGuidedConvBlock(128, 64),
            AttentionGuidedConvBlock(64, 32),
            
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder: Stego → Secret
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Attention consistency network
        self.attention_consistency = nn.Sequential(
            nn.Conv2d(input_channels + 1, 32, 3, padding=1),  # stego + original_attention
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover_image, secret_image, embedding_map):
        # Encode: Generate stego image
        encoder_input = torch.cat([cover_image, secret_image, embedding_map], dim=1)
        stego_image = self.encoder(encoder_input)
        
        # Decode: Extract secret
        extracted_secret = self.decoder(stego_image)
        
        # Attention consistency check
        consistency_input = torch.cat([stego_image, embedding_map], dim=1)
        attention_consistency = self.attention_consistency(consistency_input)
        
        return {
            'stego_image': stego_image,
            'extracted_secret': extracted_secret,
            'attention_consistency': attention_consistency
        }

class AttentionAdversarialEmbedding(nn.Module):
    """Novel: Generate adversarial perturbations guided by attention"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        # Perturbation generator
        self.perturbation_net = nn.Sequential(
            nn.Conv2d(input_channels * 2 + 1, 64, 3, padding=1),  # cover + secret + attention
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Attention-based perturbation scaling
        self.scaling_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover_image, secret_image, embedding_map):
        # Generate base perturbation
        perturbation_input = torch.cat([cover_image, secret_image, embedding_map], dim=1)
        base_perturbation = self.perturbation_net(perturbation_input)
        
        # Scale perturbation based on attention
        scaling_factor = self.scaling_net(embedding_map)
        
        # Apply attention-scaled perturbation
        scaled_perturbation = base_perturbation * scaling_factor * 0.1  # Max 10% perturbation
        
        # Generate stego image
        stego_image = cover_image + scaled_perturbation
        
        return torch.clamp(stego_image, 0, 1)

class AttentionGuidedSteganography(nn.Module):
    """
    Novel Attention-Guided Image Steganography System
    Combines multiple embedding strategies with attention guidance
    """
    
    def __init__(self, input_channels=3, hidden_channels=64, embedding_strategy='adaptive'):
        super(AttentionGuidedSteganography, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embedding_strategy = embedding_strategy
        
        # Attention heatmap generators
        self.cover_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels//2)
        self.secret_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels//2)
        
        # Novel embedding strategies
        self.texture_synthesis = AttentionTextureSynthesis(input_channels)
        self.neural_codec = AttentionNeuralCodec(input_channels)
        self.adversarial_embedding = AttentionAdversarialEmbedding(input_channels)
        
        # Fusion weight computation
        self.fusion_weight_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  # 3 weights for 3 strategies
            nn.Softmax(dim=1)
        )
        
        # Extraction network (unified for all strategies)
        self.extraction_network = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            AttentionGuidedConvBlock(64, 128),
            AttentionGuidedConvBlock(128, 64),
            AttentionGuidedConvBlock(64, 32),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Conv2d(input_channels * 2 + 2, 64, 3, padding=1),  # cover + secret + both attentions
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 strategies
            nn.Softmax(dim=1)
        )
    
    def compute_embedding_strategy(self, cover_attention, secret_attention, strategy='adaptive'):
        """
        Compute embedding strategy based on attention maps of both images
        """
        cover_embedding_att = cover_attention['embedding_attention']
        secret_embedding_att = secret_attention['embedding_attention']
        
        if strategy == 'high_low':
            # Embed in high-attention areas of cover, low-attention areas of secret
            embedding_map = cover_embedding_att * (1 - secret_embedding_att)
        elif strategy == 'low_high':
            # Embed in low-attention areas of cover, high-attention areas of secret
            embedding_map = (1 - cover_embedding_att) * secret_embedding_att
        elif strategy == 'adaptive':
            # Adaptive combination based on content analysis
            embedding_map = (cover_embedding_att + secret_embedding_att) / 2
        else:
            # Default: use cover attention directly
            embedding_map = cover_embedding_att
        
        return embedding_map
    
    def select_optimal_strategy(self, cover_image, secret_image, cover_attention, secret_attention):
        """
        Automatically select optimal embedding strategy based on image content
        """
        strategy_input = torch.cat([
            cover_image, 
            secret_image, 
            cover_attention['embedding_attention'],
            secret_attention['embedding_attention']
        ], dim=1)
        
        strategy_weights = self.strategy_selector(strategy_input)
        return strategy_weights
    
    def forward(self, cover_image, secret_image, mode='train', embedding_strategy='adaptive'):
        """
        Forward pass for novel attention-guided steganography
        """
        batch_size, channels, height, width = cover_image.shape
        
        # Generate attention heatmaps
        cover_attention = self.cover_attention_gen(cover_image)
        secret_attention = self.secret_attention_gen(secret_image)
        
        # Compute embedding map
        embedding_map = self.compute_embedding_strategy(
            cover_attention, secret_attention, embedding_strategy
        )
        
        if mode == 'train' or mode == 'embed':
            # Novel multi-strategy embedding
            
            # Strategy 1: Texture Synthesis
            texture_stego = self.texture_synthesis(cover_image, secret_image, embedding_map)
            
            # Strategy 2: Neural Codec
            codec_results = self.neural_codec(cover_image, secret_image, embedding_map)
            codec_stego = codec_results['stego_image']
            
            # Strategy 3: Adversarial Embedding
            adversarial_stego = self.adversarial_embedding(cover_image, secret_image, embedding_map)
            
            # Compute fusion weights based on attention characteristics
            fusion_weights = self.fusion_weight_net(embedding_map)  # [B, 3, H, W]
            
            # Adaptive fusion of strategies
            final_stego = (
                fusion_weights[:, 0:1] * texture_stego +
                fusion_weights[:, 1:2] * codec_stego +
                fusion_weights[:, 2:3] * adversarial_stego
            )
            
            # Extract secret using unified extraction network
            extracted_secret = self.extraction_network(final_stego)
            
            results = {
                'stego_image': final_stego,
                'extracted_secret': extracted_secret,
                'cover_attention': cover_attention,
                'secret_attention': secret_attention,
                'embedding_map': embedding_map,
                'fusion_weights': fusion_weights,
                'texture_stego': texture_stego,
                'codec_stego': codec_stego,
                'adversarial_stego': adversarial_stego,
                'attention_consistency': codec_results.get('attention_consistency', None)
            }
            
        elif mode == 'extract':
            # Extract secret from stego image
            extracted_secret = self.extraction_network(cover_image)  # cover_image is actually stego_image in extract mode
            
            results = {
                'extracted_secret': extracted_secret
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return results

class AttentionGuidedEmbeddingNetwork(AttentionGuidedSteganography):
    """
    Backward compatibility wrapper
    """
    def __init__(self, input_channels=3, hidden_channels=64):
        super().__init__(input_channels, hidden_channels)

class AttentionGuidedExtractionNetwork(nn.Module):
    """
    Standalone extraction network for inference
    """
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Unified extraction network
        self.extraction_network = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            AttentionGuidedConvBlock(64, 128),
            AttentionGuidedConvBlock(128, 64),
            AttentionGuidedConvBlock(64, 32),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, stego_image):
        """
        Extract secret from stego image
        """
        extracted_secret = self.extraction_network(stego_image)
        return extracted_secret
