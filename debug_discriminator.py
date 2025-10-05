#!/usr/bin/env python3
"""
Debug script to check discriminator output shapes
"""

import torch
from models.discriminator import SRNetDiscriminator

def debug_discriminator():
    """Debug discriminator shapes"""
    print("🔍 Debugging Discriminator Shapes...")
    
    # Create discriminator
    discriminator = SRNetDiscriminator(input_channels=3, num_classes=2)
    
    # Create sample input
    batch_size = 8
    sample_input = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = discriminator(sample_input)
    
    print(f"Output type: {type(output)}")
    
    if isinstance(output, dict):
        print("Output is dictionary:")
        for key, value in output.items():
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
    else:
        print(f"Output shape: {output.shape} (dtype: {output.dtype})")
    
    # Test labels
    real_labels = torch.zeros(batch_size, dtype=torch.long)
    fake_labels = torch.ones(batch_size, dtype=torch.long)
    
    print(f"Real labels shape: {real_labels.shape} (dtype: {real_labels.dtype})")
    print(f"Fake labels shape: {fake_labels.shape} (dtype: {fake_labels.dtype})")
    
    # Test loss function
    import torch.nn as nn
    ce_loss = nn.CrossEntropyLoss()
    
    try:
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
            
        print(f"Logits shape: {logits.shape}")
        
        loss_real = ce_loss(logits, real_labels)
        loss_fake = ce_loss(logits, fake_labels)
        
        print(f"✅ Loss computation successful!")
        print(f"Real loss: {loss_real.item():.4f}")
        print(f"Fake loss: {loss_fake.item():.4f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        print(f"Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
        print(f"Labels shape: {real_labels.shape}")

if __name__ == "__main__":
    debug_discriminator()