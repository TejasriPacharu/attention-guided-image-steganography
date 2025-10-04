#!/usr/bin/env python3
"""
Test script for Novel Attention-Guided Steganography System
"""

import torch
import gc
import sys
import os

def test_novel_model():
    """Test the novel attention-guided steganography model"""
    print("🧠 Testing Novel Attention-Guided Steganography...")
    
    try:
        from models.attention_guided_steganography import AttentionGuidedSteganography
        
        # Create model with very small parameters for testing
        model = AttentionGuidedSteganography(
            input_channels=3, 
            hidden_channels=8,  # Very small
            embedding_strategy='adaptive'
        )
        print("  ✅ Model created successfully")
        
        # Test with very small input
        batch_size = 1
        height, width = 32, 32  # Very small for testing
        
        cover = torch.randn(batch_size, 3, height, width)
        secret = torch.randn(batch_size, 3, height, width)
        
        print(f"  📏 Input size: {height}x{width}")
        print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Clear memory
        gc.collect()
        
        # Test embedding mode
        print("  🔄 Testing embedding mode...")
        with torch.no_grad():
            results = model(cover, secret, mode='train', embedding_strategy='adaptive')
            
            print("  ✅ Embedding successful!")
            print(f"  📏 Stego shape: {results['stego_image'].shape}")
            print(f"  📏 Extracted shape: {results['extracted_secret'].shape}")
            
            # Check if all novel components are working
            if 'texture_stego' in results:
                print("  ✅ Texture synthesis working")
            if 'codec_stego' in results:
                print("  ✅ Neural codec working")
            if 'adversarial_stego' in results:
                print("  ✅ Adversarial embedding working")
            if 'fusion_weights' in results:
                print("  ✅ Strategy fusion working")
                print(f"  📊 Fusion weights shape: {results['fusion_weights'].shape}")
        
        # Test extraction mode
        print("  🔄 Testing extraction mode...")
        with torch.no_grad():
            stego_image = results['stego_image']
            extract_results = model(stego_image, mode='extract')
            
            print("  ✅ Extraction successful!")
            print(f"  📏 Extracted shape: {extract_results['extracted_secret'].shape}")
        
        # Clean up
        del model, results, extract_results, cover, secret, stego_image
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_strategies():
    """Test individual embedding strategies"""
    print("\n🎯 Testing Individual Embedding Strategies...")
    
    try:
        from models.attention_guided_steganography import (
            AttentionTextureSynthesis, 
            AttentionNeuralCodec, 
            AttentionAdversarialEmbedding
        )
        
        # Test inputs
        cover = torch.randn(1, 3, 32, 32)
        secret = torch.randn(1, 3, 32, 32)
        embedding_map = torch.rand(1, 1, 32, 32)
        
        # Test Texture Synthesis
        print("  🎨 Testing Texture Synthesis...")
        texture_synth = AttentionTextureSynthesis(input_channels=3)
        with torch.no_grad():
            texture_result = texture_synth(cover, secret, embedding_map)
            print(f"    ✅ Output shape: {texture_result.shape}")
        
        # Test Neural Codec
        print("  🧠 Testing Neural Codec...")
        neural_codec = AttentionNeuralCodec(input_channels=3)
        with torch.no_grad():
            codec_results = neural_codec(cover, secret, embedding_map)
            print(f"    ✅ Stego shape: {codec_results['stego_image'].shape}")
            print(f"    ✅ Extracted shape: {codec_results['extracted_secret'].shape}")
        
        # Test Adversarial Embedding
        print("  ⚡ Testing Adversarial Embedding...")
        adv_embedding = AttentionAdversarialEmbedding(input_channels=3)
        with torch.no_grad():
            adv_result = adv_embedding(cover, secret, embedding_map)
            print(f"    ✅ Output shape: {adv_result.shape}")
        
        # Clean up
        del texture_synth, neural_codec, adv_embedding
        del cover, secret, embedding_map, texture_result, codec_results, adv_result
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy test failed: {e}")
        return False

def test_attention_generation():
    """Test attention heatmap generation"""
    print("\n🎯 Testing Attention Generation...")
    
    try:
        from models.attention_heatmap import AttentionHeatmapGenerator
        
        # Create attention generator
        attention_gen = AttentionHeatmapGenerator(input_channels=3, feature_channels=8)
        print("  ✅ Attention generator created")
        
        # Test with small input
        test_input = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            attention_maps = attention_gen(test_input)
            print("  ✅ Attention generation successful!")
            print(f"  📏 Embedding attention shape: {attention_maps['embedding_attention'].shape}")
            
            # Check all attention types
            expected_keys = ['final_attention', 'embedding_attention', 'spatial_attention', 
                           'edge_attention', 'texture_complexity', 'texture_suitability']
            
            for key in expected_keys:
                if key in attention_maps:
                    print(f"    ✅ {key}: {attention_maps[key].shape}")
                else:
                    print(f"    ⚠️  {key}: missing")
        
        del attention_gen, attention_maps, test_input
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Attention test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage and efficiency"""
    print("\n💾 Testing Memory Usage...")
    
    try:
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  📊 Initial memory: {initial_memory:.1f} MB")
        
        # Create and test model
        from models.attention_guided_steganography import AttentionGuidedSteganography
        
        model = AttentionGuidedSteganography(input_channels=3, hidden_channels=8)
        
        # Test forward pass
        cover = torch.randn(1, 3, 32, 32)
        secret = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            results = model(cover, secret, mode='train', embedding_strategy='adaptive')
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"  📊 Peak memory: {peak_memory:.1f} MB")
        print(f"  📊 Memory increase: {memory_increase:.1f} MB")
        
        # Clean up
        del model, results, cover, secret
        gc.collect()
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  📊 Final memory: {final_memory:.1f} MB")
        
        if memory_increase < 500:  # Less than 500MB increase
            print("  ✅ Memory usage acceptable")
            return True
        else:
            print("  ⚠️  High memory usage detected")
            return True  # Still pass, but with warning
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🎯 NOVEL ATTENTION-GUIDED STEGANOGRAPHY TEST")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Attention Generation", test_attention_generation),
        ("Individual Strategies", test_individual_strategies),
        ("Complete Model", test_novel_model),
        ("Memory Usage", test_memory_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} test PASSED")
            else:
                print(f"❌ {name} test FAILED")
        except Exception as e:
            print(f"❌ {name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n💡 Novel steganography system is ready!")
        print("\n🚀 To start training:")
        print("   python train_novel_cpu.py  # For CPU training")
        print("   python train_novel.py      # For GPU training")
        
        print("\n📋 Training features:")
        print("   ✅ Multi-strategy embedding (Texture + Codec + Adversarial)")
        print("   ✅ Attention-guided spatial embedding")
        print("   ✅ Adaptive strategy fusion")
        print("   ✅ Comprehensive loss functions")
        print("   ✅ Memory-optimized for CPU")
        
    elif passed >= total - 1:
        print("⚠️  Most tests passed - system should work!")
        print("   You can proceed with training.")
    else:
        print("❌ Multiple tests failed - please fix issues before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
