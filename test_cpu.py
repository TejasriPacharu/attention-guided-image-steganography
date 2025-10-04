#!/usr/bin/env python3
"""
Test CPU-optimized model
"""

import torch
import gc
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cpu_model():
    """Test the CPU-optimized model"""
    print("🧠 Testing CPU-optimized model...")
    
    try:
        # Import the CPU-optimized model
        from train_cpu import MemoryEfficientSteganography
        
        # Create model with very small parameters
        model = MemoryEfficientSteganography(input_channels=3, hidden_channels=8)
        print("  ✅ Model created successfully")
        
        # Test with very small input
        batch_size = 1
        height, width = 32, 32  # Very small
        
        cover = torch.randn(batch_size, 3, height, width)
        secret = torch.randn(batch_size, 3, height, width)
        
        print(f"  📏 Input size: {height}x{width}")
        print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Clear memory
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
        import traceback
        traceback.print_exc()
        return False

def test_attention_only():
    """Test just the attention generator"""
    print("\n🎯 Testing attention generator...")
    
    try:
        from models.attention_heatmap import AttentionHeatmapGenerator
        
        # Create very small attention generator
        attention_gen = AttentionHeatmapGenerator(input_channels=3, feature_channels=8)
        print("  ✅ Attention generator created")
        
        # Test with tiny input
        test_input = torch.randn(1, 3, 32, 32)
        
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
    print("🪶 CPU MODEL TEST (Ultra Lightweight)")
    print("=" * 50)
    
    # Check memory before starting
    import psutil
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    tests = [
        test_attention_only,
        test_cpu_model
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        else:
            print("❌ Stopping tests due to failure")
            break
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 CPU MODEL TESTS PASSED!")
        print("\n💡 Ready for CPU training!")
        print("   python train_cpu.py")
        print("   # Uses 64x64 images, batch size 2, 16 hidden channels")
    else:
        print("❌ Tests failed - model still too large for system")
        print("💡 Try reducing parameters further or use cloud GPU")

if __name__ == "__main__":
    main()
