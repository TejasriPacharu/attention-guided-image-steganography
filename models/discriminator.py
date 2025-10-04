import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNetBlock(nn.Module):
    """Basic SRNet block with separable convolutions"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(SRNetBlock, self).__init__()
        
        # Separable convolution: depthwise + pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                                  padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SRNetDiscriminator(nn.Module):
    """
    SRNet-based discriminator for steganography detection
    Based on the SRNet architecture for steganalysis
    """
    
    def __init__(self, input_channels=3, num_classes=2):
        super(SRNetDiscriminator, self).__init__()
        
        # Initial preprocessing layer
        self.preprocessing = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # SRNet blocks with increasing channels
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple SRNet blocks"""
        layers = []
        layers.append(SRNetBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(SRNetBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the discriminator
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Classification logits and confidence score
        """
        # Preprocessing
        x = self.preprocessing(x)
        
        # Feature extraction through SRNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        
        # Convert to probability (confidence score)
        confidence = torch.softmax(logits, dim=1)[:, 1]  # Probability of being stego
        
        return {
            'logits': logits,
            'confidence': confidence
        }

class AdvancedDiscriminator(nn.Module):
    """
    Advanced discriminator with attention mechanisms for better detection
    """
    
    def __init__(self, input_channels=3, num_classes=2):
        super(AdvancedDiscriminator, self).__init__()
        
        # Multi-scale feature extraction
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.scale2_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.scale3_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # SRNet backbone
        self.srnet_backbone = SRNetDiscriminator(64, num_classes)
        
        # Additional classification layers
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass with multi-scale attention
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Enhanced classification results
        """
        # Multi-scale feature extraction
        scale1_feat = self.scale1_conv(x)
        scale2_feat = self.scale2_conv(x)
        scale3_feat = self.scale3_conv(x)
        
        # Feature fusion
        fused_features = torch.cat([scale1_feat, scale2_feat, scale3_feat], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Apply attention
        attention_map = self.attention(fused_features)
        attended_features = fused_features * attention_map
        
        # Main classification through SRNet
        main_results = self.srnet_backbone(attended_features)
        
        # Auxiliary classification
        aux_logits = self.aux_classifier(attended_features)
        aux_confidence = torch.softmax(aux_logits, dim=1)[:, 1]
        
        # Combine results
        combined_confidence = (main_results['confidence'] + aux_confidence) / 2
        
        return {
            'logits': main_results['logits'],
            'confidence': combined_confidence,
            'aux_logits': aux_logits,
            'attention_map': attention_map,
            'main_confidence': main_results['confidence'],
            'aux_confidence': aux_confidence
        }

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for comprehensive steganalysis
    """
    
    def __init__(self, input_channels=3, scales=[1, 0.5, 0.25]):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.scales = scales
        self.discriminators = nn.ModuleList([
            SRNetDiscriminator(input_channels) for _ in scales
        ])
        
        # Scale weight learning
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, x):
        """
        Multi-scale discrimination
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Aggregated discrimination results
        """
        results = []
        confidences = []
        
        for i, (scale, discriminator) in enumerate(zip(self.scales, self.discriminators)):
            if scale != 1.0:
                # Resize input for different scales
                h, w = x.shape[2], x.shape[3]
                scaled_x = F.interpolate(x, size=(int(h * scale), int(w * scale)), 
                                       mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Get discrimination results
            result = discriminator(scaled_x)
            results.append(result)
            confidences.append(result['confidence'])
        
        # Weighted aggregation
        weights = F.softmax(self.scale_weights, dim=0)
        aggregated_confidence = sum(w * conf for w, conf in zip(weights, confidences))
        
        # Aggregate logits (simple average)
        aggregated_logits = sum(result['logits'] for result in results) / len(results)
        
        return {
            'logits': aggregated_logits,
            'confidence': aggregated_confidence,
            'scale_results': results,
            'scale_weights': weights
        }
