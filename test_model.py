#!/usr/bin/env python3
"""
Quick test script to verify model architecture works
"""

import torch
from models import AttentionGuidedSteganography

def test_model():
    """Test the model with dummy data"""
    print("Testing model architecture...")
    
    # Create model
    model = AttentionGuidedSteganography(input_channels=3, hidden_channels=64)
    
    # Create dummy input
    batch_size = 1
    cover_images = torch.randn(batch_size, 3, 256, 256)
    secret_images = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input shapes:")
    print(f"  Cover: {cover_images.shape}")
    print(f"  Secret: {secret_images.shape}")
    
    # Test forward pass
    with torch.no_grad():
        try:
            results = model(cover_images, secret_images, mode='train', embedding_strategy='adaptive')
            print(f"✅ Forward pass successful!")
            print(f"Output shapes:")
            print(f"  Stego: {results['stego_image'].shape}")
            print(f"  Extracted: {results['extracted_secret'].shape}")
            return True
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("🎉 Model test passed!")
    else:
        print("❌ Model test failed!")
        exit(1)
