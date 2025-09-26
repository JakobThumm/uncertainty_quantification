#!/usr/bin/env python3
"""
Test script to verify YOLO GPU usage

This script tests if YOLO is actually using the GPU for inference
and provides methods to check and control device usage.
"""

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

def check_gpu_memory_usage():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f} GB")
    else:
        print("CUDA not available")

def test_yolo_device_usage():
    """Test YOLO with different device configurations"""
    print("=" * 60)
    print("Testing YOLO Device Usage")
    print("=" * 60)

    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)

    try:
        # Load YOLO model
        print("Loading YOLO model...")
        model = YOLO("yolo11n.pt")  # Download pretrained model
        print("✓ YOLO model loaded")

        # Check where model parameters are
        device_info = next(model.model.parameters()).device if hasattr(model, 'model') else "Unknown"
        print(f"Model parameters device: {device_info}")

        print("\n" + "-" * 40)
        print("Testing CPU Inference")
        print("-" * 40)

        # Test CPU inference
        print("Initial GPU memory:")
        check_gpu_memory_usage()

        start_time = time.time()
        results_cpu = model.predict(pil_image, device='cpu', verbose=False)
        cpu_time = time.time() - start_time

        print(f"CPU inference time: {cpu_time:.3f}s")
        print("GPU memory after CPU inference:")
        check_gpu_memory_usage()

        if torch.cuda.is_available():
            print("\n" + "-" * 40)
            print("Testing GPU Inference")
            print("-" * 40)

            # Clear GPU cache
            torch.cuda.empty_cache()
            print("GPU memory after cache clear:")
            check_gpu_memory_usage()

            # Test GPU inference
            start_time = time.time()
            results_gpu = model.predict(pil_image, device=0, verbose=False)  # device=0 means GPU 0
            gpu_time = time.time() - start_time

            print(f"GPU inference time: {gpu_time:.3f}s")
            print("GPU memory after GPU inference:")
            check_gpu_memory_usage()

            # Compare results
            print(f"\nPerformance comparison:")
            print(f"CPU time: {cpu_time:.3f}s")
            print(f"GPU time: {gpu_time:.3f}s")
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"Speedup (CPU/GPU): {speedup:.2f}x")

            if speedup > 1.2:
                print("✓ GPU is providing speedup - GPU inference is working!")
            elif speedup < 0.8:
                print("⚠ GPU is slower than CPU - might be overhead or compatibility issues")
            else:
                print("≈ GPU and CPU performance similar - check if GPU is actually being used")

            # Test with larger batch to see more GPU benefit
            print("\n" + "-" * 40)
            print("Testing Batch Processing")
            print("-" * 40)

            # Create batch of images
            batch_images = [pil_image] * 4

            start_time = time.time()
            results_cpu_batch = model.predict(batch_images, device='cpu', verbose=False)
            cpu_batch_time = time.time() - start_time

            torch.cuda.empty_cache()
            start_time = time.time()
            results_gpu_batch = model.predict(batch_images, device=0, verbose=False)
            gpu_batch_time = time.time() - start_time

            print(f"Batch CPU time: {cpu_batch_time:.3f}s")
            print(f"Batch GPU time: {gpu_batch_time:.3f}s")
            batch_speedup = cpu_batch_time / gpu_batch_time if gpu_batch_time > 0 else 0
            print(f"Batch speedup: {batch_speedup:.2f}x")

        else:
            print("CUDA not available - skipping GPU tests")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_model_device_movement():
    """Test moving model to different devices"""
    print("\n" + "=" * 60)
    print("Testing Model Device Movement")
    print("=" * 60)

    try:
        model = YOLO("yolo11n.pt")

        # Check initial device
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            initial_device = next(model.model.parameters()).device
            print(f"Initial model device: {initial_device}")

        if torch.cuda.is_available():
            # Move to GPU
            print("Moving model to GPU...")
            model.to('cuda')

            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                gpu_device = next(model.model.parameters()).device
                print(f"Model device after .to('cuda'): {gpu_device}")

            # Test inference
            test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            pil_image = Image.fromarray(test_image)

            start_time = time.time()
            results = model.predict(pil_image, verbose=False)
            inference_time = time.time() - start_time

            print(f"Inference time after model.to('cuda'): {inference_time:.3f}s")
            print("GPU memory after inference:")
            check_gpu_memory_usage()

        else:
            print("CUDA not available for device movement test")

    except Exception as e:
        print(f"Error during device movement test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("YOLO GPU Usage Verification")
    print("=" * 60)

    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    # Test device usage
    test_yolo_device_usage()

    # Test model device movement
    test_model_device_movement()

    print("\n" + "=" * 60)
    print("Summary and Recommendations")
    print("=" * 60)

    if torch.cuda.is_available():
        print("For optimal GPU usage with ultralytics YOLO:")
        print("1. Use device=0 (or device='cuda') in predict() calls")
        print("2. Use model.to('cuda') to move model to GPU")
        print("3. Check GPU memory usage to verify GPU is being used")
        print("4. Compare inference times between CPU and GPU")
        print("\nExample usage:")
        print("  model = YOLO('yolo11n.pt')")
        print("  model.to('cuda')  # Move model to GPU")
        print("  results = model.predict(image, device=0)  # Use GPU for inference")
    else:
        print("CUDA not available - will use CPU for YOLO inference")

if __name__ == "__main__":
    main()