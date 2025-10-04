#!/usr/bin/env python3
"""
Memory Check and Optimization Recommendations
"""

import torch
import psutil
import os

def check_system_memory():
    """Check system RAM"""
    memory = psutil.virtual_memory()
    print(f"🖥️  System Memory:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent:.1f}%")
    return memory.available / (1024**3)

def check_gpu_memory():
    """Check GPU memory if available"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"\n🔥 GPU Memory:")
        print(f"   Total: {gpu_memory / (1024**3):.1f} GB")
        
        # Check current usage
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        
        print(f"   Allocated: {allocated / (1024**3):.1f} GB")
        print(f"   Cached: {cached / (1024**3):.1f} GB")
        print(f"   Free: {(gpu_memory - cached) / (1024**3):.1f} GB")
        
        return gpu_memory / (1024**3)
    else:
        print("\n⚠️  No GPU available")
        return 0

def recommend_parameters(available_ram, gpu_memory):
    """Recommend optimal parameters based on available memory"""
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 40)
    
    if available_ram < 4:
        print("❌ Low RAM detected (<4GB available)")
        print("   Recommended: Use Google Colab or cloud GPU")
        return
    
    # RAM-based recommendations
    if available_ram >= 8:
        ram_config = {
            'batch_size': 8,
            'image_size': 256,
            'hidden_channels': 64,
            'num_workers': 4
        }
        print("✅ Good RAM (8GB+)")
    elif available_ram >= 4:
        ram_config = {
            'batch_size': 4,
            'image_size': 128,
            'hidden_channels': 32,
            'num_workers': 2
        }
        print("⚠️  Limited RAM (4-8GB)")
    else:
        ram_config = {
            'batch_size': 2,
            'image_size': 64,
            'hidden_channels': 16,
            'num_workers': 1
        }
        print("❌ Very Limited RAM (<4GB)")
    
    # GPU-based adjustments
    if gpu_memory >= 12:
        print("✅ High-end GPU (12GB+)")
        gpu_multiplier = 1.5
    elif gpu_memory >= 8:
        print("✅ Good GPU (8-12GB)")
        gpu_multiplier = 1.2
    elif gpu_memory >= 4:
        print("⚠️  Mid-range GPU (4-8GB)")
        gpu_multiplier = 1.0
    elif gpu_memory > 0:
        print("❌ Low-end GPU (<4GB)")
        gpu_multiplier = 0.7
    else:
        print("❌ CPU only")
        gpu_multiplier = 0.5
    
    # Final recommendations
    final_config = {
        'batch_size': max(1, int(ram_config['batch_size'] * gpu_multiplier)),
        'image_size': ram_config['image_size'],
        'hidden_channels': ram_config['hidden_channels'],
        'num_workers': ram_config['num_workers']
    }
    
    print(f"\n📋 OPTIMAL CONFIGURATION:")
    print(f"   Batch Size: {final_config['batch_size']}")
    print(f"   Image Size: {final_config['image_size']}x{final_config['image_size']}")
    print(f"   Hidden Channels: {final_config['hidden_channels']}")
    print(f"   Num Workers: {final_config['num_workers']}")
    
    # Generate command
    if gpu_memory > 0:
        script = "./train_lightweight.sh"
    else:
        script = "./train_lightweight.sh"
    
    print(f"\n🚀 RECOMMENDED COMMAND:")
    print(f"   chmod +x {script}")
    print(f"   {script} \\")
    print(f"     --batch_size {final_config['batch_size']} \\")
    print(f"     --image_size {final_config['image_size']} \\")
    print(f"     --hidden_channels {final_config['hidden_channels']}")
    
    return final_config

def estimate_memory_usage(config):
    """Estimate memory usage for given configuration"""
    batch_size = config['batch_size']
    image_size = config['image_size']
    hidden_channels = config['hidden_channels']
    
    # Rough estimation (in GB)
    # Input images: batch_size * 3 * image_size^2 * 4 bytes * 2 (cover + secret)
    input_memory = batch_size * 3 * image_size * image_size * 4 * 2 / (1024**3)
    
    # Model parameters and activations (rough estimate)
    model_memory = hidden_channels * image_size * image_size * 4 / (1024**2) * 10  # MB
    
    total_estimate = input_memory + model_memory / 1024
    
    print(f"\n📊 ESTIMATED MEMORY USAGE:")
    print(f"   Input tensors: {input_memory:.2f} GB")
    print(f"   Model + activations: {model_memory:.0f} MB")
    print(f"   Total estimate: {total_estimate:.2f} GB")
    
    return total_estimate

def main():
    print("=" * 50)
    print("🔍 MEMORY CHECK & OPTIMIZATION")
    print("=" * 50)
    
    # Check available resources
    available_ram = check_system_memory()
    gpu_memory = check_gpu_memory()
    
    # Get recommendations
    config = recommend_parameters(available_ram, gpu_memory)
    
    if config:
        # Estimate memory usage
        estimated_usage = estimate_memory_usage(config)
        
        # Safety check
        if estimated_usage > available_ram * 0.8:
            print(f"\n⚠️  WARNING: Estimated usage ({estimated_usage:.1f}GB) may exceed available RAM")
            print("   Consider reducing batch size or image size further")

if __name__ == "__main__":
    main()
