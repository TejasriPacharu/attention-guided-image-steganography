#!/usr/bin/env python3
"""
Lightweight test for memory-constrained environments
"""

import torch
import gc

def test_model_lightweight():
    """Test model with smaller inputs to avoid memory issues"""
    print("🧠 Testing model (lightweight)...")
    
    try:
        from models import AttentionGuidedSteganography
        
        # Create model with smaller hidden channels
        model = AttentionGuidedSteganography(input_channels=3, hidden_channels=32)
        print("  ✅ Model created (32 hidden channels)")
        
        # Test with smaller input size
        batch_size = 1
        height, width = 128, 128  # Smaller than 256x256
        
        cover = torch.randn(batch_size, 3, height, width)
        secret = torch.randn(batch_size, 3, height, width)
        
        print(f"  📏 Input size: {height}x{width}")
        
        # Clear any existing tensors
        gc.collect()
        
        # Test forward pass
        with torch.no_grad():
            results = model(cover, secret, mode='train', embedding_strategy='adaptive')
            
            print("  ✅ Forward pass successful!")
            print(f"  📏 Stego shape: {results['stego_image'].shape}")
            print(f"  📏 Extracted shape: {results['extracted_secret'].shape}")
        
        # Clean up
        del model, results, cover, secret
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def test_attention_generator():
    """Test just the attention generator"""
    print("\n🎯 Testing attention generator...")
    
    try:
        from models.attention_heatmap import AttentionHeatmapGenerator
        
        # Create smaller attention generator
        attention_gen = AttentionHeatmapGenerator(input_channels=3, feature_channels=32)
        print("  ✅ Attention generator created")
        
        # Test with small input
        test_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            attention_maps = attention_gen(test_input)
            print("  ✅ Attention generation successful!")
            print(f"  📏 Embedding attention shape: {attention_maps['embedding_attention'].shape}")
        
        del attention_gen, attention_maps, test_input
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Attention test failed: {e}")
        return False

def main():
    print("=" * 50)
    print("🪶 LIGHTWEIGHT MODEL TEST")
    print("=" * 50)
    
    tests = [
        test_attention_generator,
        test_model_lightweight
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        else:
            break  # Stop on first failure to avoid memory issues
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 LIGHTWEIGHT TESTS PASSED!")
        print("\n💡 Model architecture is working!")
        print("   For training, consider:")
        print("   - Smaller batch size (4-6)")
        print("   - Smaller image size (128x128)")
        print("   - Reduced hidden channels (32-48)")
    else:
        print("❌ Tests failed - check model architecture")

if __name__ == "__main__":
    main()
