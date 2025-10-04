import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_heatmap import AttentionHeatmapGenerator
from .transformer import CAISFormerBlock, ConvLNReLU

class AttentionGuidedEmbeddingNetwork(nn.Module):
    """
    Attention-guided embedding network that uses attention heatmaps 
    to determine optimal embedding regions
    """
    
    def __init__(self, input_channels=3, hidden_channels=64):
        super(AttentionGuidedEmbeddingNetwork, self).__init__()
        
        # Attention heatmap generators
        self.cover_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels)
        self.secret_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels)
        
        # Initial feature extraction
        self.cover_encoder = nn.Sequential(
            ConvLNReLU(input_channels, 32),
            ConvLNReLU(32, hidden_channels)
        )
        
        self.secret_encoder = nn.Sequential(
            ConvLNReLU(input_channels, 32),
            ConvLNReLU(32, hidden_channels)
        )
        
        # Attention fusion module
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # CAISFormer blocks for feature processing
        self.caisformer_blocks = nn.ModuleList([
            CAISFormerBlock(hidden_channels) for _ in range(3)
        ])
        
        # Attention-guided feature modulation
        self.attention_modulator = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion and output generation
        self.feature_fusion = nn.Sequential(
            ConvLNReLU(hidden_channels * 2, hidden_channels),
            ConvLNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, input_channels, 3, padding=1)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def compute_embedding_strategy(self, cover_attention, secret_attention, strategy='adaptive'):
        """
        Compute embedding strategy based on attention maps of both images
        
        Args:
            cover_attention: Attention map of cover image
            secret_attention: Attention map of secret image  
            strategy: Embedding strategy ('high_low', 'low_high', 'adaptive')
        """
        if strategy == 'high_low':
            # Embed in high-attention areas of cover, low-attention areas of secret
            embedding_map = cover_attention['embedding_attention'] * (1 - secret_attention['embedding_attention'])
        elif strategy == 'low_high':
            # Embed in low-attention areas of cover, high-attention areas of secret
            embedding_map = (1 - cover_attention['embedding_attention']) * secret_attention['embedding_attention']
        elif strategy == 'adaptive':
            # Adaptive strategy based on texture complexity and edge information
            cover_texture = cover_attention['texture_complexity']
            secret_edges = secret_attention['edge_attention']
            
            # Prefer textured areas in cover with important edges in secret
            embedding_map = cover_texture * secret_edges * 0.6 + \
                           cover_attention['embedding_attention'] * 0.4
        else:
            # Default: use cover attention directly
            embedding_map = cover_attention['embedding_attention']
        
        return embedding_map
    
    def forward(self, cover_image, secret_image, embedding_strategy='adaptive'):
        """
        Forward pass for attention-guided embedding
        
        Args:
            cover_image: Cover image tensor [B, C, H, W]
            secret_image: Secret image tensor [B, C, H, W]
            embedding_strategy: Strategy for combining attention maps
            
        Returns:
            Stego image and attention visualizations
        """
        batch_size, channels, height, width = cover_image.shape
        
        # Generate attention heatmaps
        cover_attention = self.cover_attention_gen(cover_image)
        secret_attention = self.secret_attention_gen(secret_image)
        
        # Compute embedding strategy
        embedding_map = self.compute_embedding_strategy(
            cover_attention, secret_attention, embedding_strategy
        )
        
        # Feature extraction
        cover_features = self.cover_encoder(cover_image)
        secret_features = self.secret_encoder(secret_image)
        
        # Apply CAISFormer blocks with cross-attention
        for block in self.caisformer_blocks:
            cover_features = block(cover_features, secret_features)
        
        # Attention-guided feature modulation
        # Concatenate features with embedding map
        modulated_features = torch.cat([cover_features, embedding_map], dim=1)
        modulated_features = self.attention_modulator(modulated_features)
        
        # Apply embedding map as a gating mechanism
        gated_secret_features = secret_features * embedding_map
        
        # Feature fusion
        fused_features = torch.cat([modulated_features, gated_secret_features], dim=1)
        embedding_residual = self.feature_fusion(fused_features)
        
        # Generate stego image with residual connection
        stego_image = cover_image + self.residual_weight * embedding_residual
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return {
            'stego_image': stego_image,
            'embedding_map': embedding_map,
            'cover_attention': cover_attention,
            'secret_attention': secret_attention,
            'embedding_residual': embedding_residual
        }

class AttentionGuidedExtractionNetwork(nn.Module):
    """
    Extraction network that recovers the secret image from stego image
    using attention guidance
    """
    
    def __init__(self, input_channels=3, hidden_channels=64):
        super(AttentionGuidedExtractionNetwork, self).__init__()
        
        # Attention generator for stego image
        self.stego_attention_gen = AttentionHeatmapGenerator(input_channels, hidden_channels)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            ConvLNReLU(input_channels, 32),
            ConvLNReLU(32, hidden_channels),
            ConvLNReLU(hidden_channels, hidden_channels)
        )
        
        # CAISFormer blocks for extraction
        self.extraction_blocks = nn.ModuleList([
            CAISFormerBlock(hidden_channels) for _ in range(4)
        ])
        
        # Attention-guided extraction
        self.attention_guided_extractor = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Secret image reconstruction
        self.secret_decoder = nn.Sequential(
            ConvLNReLU(hidden_channels, hidden_channels // 2),
            ConvLNReLU(hidden_channels // 2, 32),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, stego_image):
        """
        Extract secret image from stego image
        
        Args:
            stego_image: Stego image tensor [B, C, H, W]
            
        Returns:
            Extracted secret image and attention maps
        """
        # Generate attention map for stego image
        stego_attention = self.stego_attention_gen(stego_image)
        extraction_map = stego_attention['embedding_attention']
        
        # Feature extraction
        stego_features = self.feature_extractor(stego_image)
        
        # Apply extraction blocks
        for block in self.extraction_blocks:
            stego_features = block(stego_features)
        
        # Attention-guided extraction
        guided_features = torch.cat([stego_features, extraction_map], dim=1)
        extracted_features = self.attention_guided_extractor(guided_features)
        
        # Apply attention as gating
        gated_features = extracted_features * extraction_map
        
        # Decode secret image
        extracted_secret = self.secret_decoder(gated_features)
        
        return {
            'extracted_secret': extracted_secret,
            'extraction_map': extraction_map,
            'stego_attention': stego_attention
        }

class AttentionGuidedSteganography(nn.Module):
    """
    Complete attention-guided steganography system with embedding and extraction
    """
    
    def __init__(self, input_channels=3, hidden_channels=64):
        super(AttentionGuidedSteganography, self).__init__()
        
        self.embedding_network = AttentionGuidedEmbeddingNetwork(input_channels, hidden_channels)
        self.extraction_network = AttentionGuidedExtractionNetwork(input_channels, hidden_channels)
        
        # Loss computation modules
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, cover_image, secret_image=None, mode='train', embedding_strategy='adaptive'):
        """
        Forward pass for training or inference
        
        Args:
            cover_image: Cover image tensor
            secret_image: Secret image tensor (required for embedding)
            mode: 'train', 'embed', or 'extract'
            embedding_strategy: Strategy for attention-guided embedding
            
        Returns:
            Dictionary with results based on mode
        """
        if mode == 'train' or mode == 'embed':
            if secret_image is None:
                raise ValueError("Secret image required for embedding mode")
            
            # Embedding phase
            embed_results = self.embedding_network(cover_image, secret_image, embedding_strategy)
            
            if mode == 'embed':
                return embed_results
            
            # Extraction phase (for training)
            extract_results = self.extraction_network(embed_results['stego_image'])
            
            return {
                **embed_results,
                **extract_results,
                'mode': 'train'
            }
        
        elif mode == 'extract':
            # Extraction only
            return self.extraction_network(cover_image)  # cover_image is actually stego_image here
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute_losses(self, results, cover_image, secret_image, loss_weights=None):
        """
        Compute comprehensive loss for training
        
        Args:
            results: Forward pass results
            cover_image: Original cover image
            secret_image: Original secret image
            loss_weights: Dictionary of loss weights
            
        Returns:
            Dictionary of losses
        """
        if loss_weights is None:
            loss_weights = {
                'cover_loss': 1.0,
                'secret_loss': 1.0,
                'attention_loss': 0.1,
                'perceptual_loss': 0.5
            }
        
        losses = {}
        
        # Cover image preservation loss
        losses['cover_loss'] = self.mse_loss(results['stego_image'], cover_image)
        
        # Secret image reconstruction loss
        losses['secret_loss'] = self.mse_loss(results['extracted_secret'], secret_image)
        
        # Attention consistency loss (embedding and extraction maps should be similar)
        losses['attention_loss'] = self.mse_loss(
            results['embedding_map'], 
            results['extraction_map']
        )
        
        # Perceptual loss using L1 in feature space
        losses['perceptual_loss'] = self.l1_loss(results['stego_image'], cover_image)
        
        # Total weighted loss
        total_loss = sum(loss_weights[k] * losses[k] for k in losses.keys() if k in loss_weights)
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_attention_visualizations(self, results):
        """
        Get attention visualizations for analysis
        
        Returns:
            Dictionary of attention maps for visualization
        """
        visualizations = {
            'cover_attention': results['cover_attention']['embedding_attention'],
            'secret_attention': results['secret_attention']['embedding_attention'],
            'embedding_map': results['embedding_map'],
            'extraction_map': results.get('extraction_map', None)
        }
        
        return visualizations
