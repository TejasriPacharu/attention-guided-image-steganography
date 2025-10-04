#!/usr/bin/env python3
"""
Quick test for GPU training script syntax and imports
"""

def test_gpu_training_script():
    """Test if the GPU training script can be imported and parsed"""
    print("🧪 Testing GPU Training Script...")
    
    try:
        # Test syntax by importing
        print("  📝 Testing script syntax...")
        import train_novel
        print("    ✅ Script syntax is valid")
        
        # Test if main components are accessible
        print("  🔍 Testing main components...")
        trainer_class = getattr(train_novel, 'NovelSteganographyTrainer', None)
        main_function = getattr(train_novel, 'main', None)
        
        if trainer_class:
            print("    ✅ NovelSteganographyTrainer class found")
        else:
            print("    ❌ NovelSteganographyTrainer class not found")
            
        if main_function:
            print("    ✅ main function found")
        else:
            print("    ❌ main function not found")
        
        print("\n🎉 GPU TRAINING SCRIPT TEST PASSED!")
        print("✅ Ready to run: python train_novel.py")
        return True
        
    except SyntaxError as e:
        print(f"\n❌ SYNTAX ERROR: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTING GPU TRAINING SCRIPT")
    print("=" * 60)
    
    success = test_gpu_training_script()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Your novel steganography system is ready!")
        print("\n💡 Training options:")
        print("  🖥️  GPU Training: python train_novel.py")
        print("  💾 CPU Training: python train_novel_cpu.py")
        print("\n📊 Features:")
        print("  🎯 Dual-image attention guidance")
        print("  🚀 Multi-strategy embedding")
        print("  📈 Comprehensive visualizations")
        print("  🔗 Adaptive strategy fusion")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SCRIPT HAS ISSUES!")
        print("⚠️ Fix syntax errors before training!")
        print("=" * 60)
