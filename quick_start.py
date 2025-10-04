#!/usr/bin/env python3
"""
Quick Start Script for Attention-Guided Image Steganography
This script helps you get started with the project quickly.
"""

import os
import sys
import subprocess
import torch

def print_header():
    print("="*60)
    print("🎯 ATTENTION-GUIDED IMAGE STEGANOGRAPHY")
    print("   Dynamic Embedding using Attention Heatmaps")
    print("="*60)

def check_requirements():
    """Check if all requirements are installed"""
    print("📋 Checking requirements...")
    
    # Map package names to their import names
    package_imports = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pillow': 'PIL',
        'scikit-image': 'skimage',
        'tensorboard': 'tensorboard',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("  pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def check_dataset():
    """Check dataset structure"""
    print("\n📁 Checking dataset structure...")
    
    dataset_dir = "./dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return False
    
    required_dirs = ['train', 'val', 'test']
    total_images = 0
    
    for subdir in required_dirs:
        path = os.path.join(dataset_dir, subdir)
        if not os.path.exists(path):
            print(f"❌ Missing subdirectory: {subdir}")
            return False
        
        # Count images
        image_count = 0
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            import glob
            image_count += len(glob.glob(os.path.join(path, f"*{ext}")))
            image_count += len(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        
        total_images += image_count
        print(f"  ✓ {subdir}: {image_count:,} images")
    
    print(f"✅ Dataset ready with {total_images:,} total images!")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"  ✅ CUDA available")
        print(f"  🔥 GPU: {gpu_name}")
        print(f"  💾 Memory: {memory_gb:.1f} GB")
        print(f"  📊 Device count: {gpu_count}")
        return True
    else:
        print("  ⚠️  CUDA not available - will use CPU")
        print("     Training will be significantly slower")
        return False

def test_model():
    """Test model creation and forward pass"""
    print("\n🧠 Testing model...")
    
    try:
        from models import AttentionGuidedSteganography
        
        # Create model
        model = AttentionGuidedSteganography(input_channels=3, hidden_channels=64)
        
        # Test forward pass
        with torch.no_grad():
            cover = torch.randn(1, 3, 256, 256)
            secret = torch.randn(1, 3, 256, 256)
            results = model(cover, secret, mode='train', embedding_strategy='adaptive')
        
        print("  ✅ Model creation successful")
        print("  ✅ Forward pass successful")
        print(f"  📏 Output shapes: {results['stego_image'].shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def show_usage():
    """Show usage instructions"""
    print("\n🚀 QUICK START GUIDE")
    print("-" * 40)
    
    print("\n1️⃣  Training:")
    print("   chmod +x train.sh")
    print("   ./train.sh")
    print("   # Or with custom parameters:")
    print("   ./train.sh --batch_size 16 --num_epochs 50")
    
    print("\n2️⃣  Demo (after training):")
    print("   python demo.py --mode embed \\")
    print("     --cover_image path/to/cover.jpg \\")
    print("     --secret_image path/to/secret.jpg")
    
    print("\n3️⃣  Evaluation:")
    print("   python evaluate.py \\")
    print("     --model_path ./checkpoints/best_model.pth \\")
    print("     --test_data_dir ./dataset/test")
    
    print("\n4️⃣  Monitor training:")
    print("   tensorboard --logdir=./logs")
    
    print("\n📚 For more details, see README.md")

def main():
    """Main function"""
    print_header()
    
    # Run all checks
    checks = [
        ("Requirements", check_requirements),
        ("Dataset", check_dataset),
        ("GPU", check_gpu),
        ("Model", test_model)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
    
    print("\n" + "="*60)
    print(f"📊 SETUP STATUS: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 SETUP COMPLETE! Ready to start training.")
        show_usage()
        
        # Ask if user wants to start training
        print("\n" + "="*60)
        response = input("🤔 Would you like to start training now? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("\n🚀 Starting training...")
            try:
                # Make train.sh executable
                os.chmod("train.sh", 0o755)
                # Run training script
                subprocess.run(["./train.sh"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Training failed: {e}")
            except FileNotFoundError:
                print("❌ train.sh not found. Please run manually:")
                print("   chmod +x train.sh && ./train.sh")
        else:
            print("\n👍 Setup complete! Run './train.sh' when ready to train.")
    else:
        print("❌ Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
