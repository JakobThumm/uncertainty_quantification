#!/usr/bin/env python3
"""
Test script for YOLOv5 loading and compatibility

This script tests YOLOv5 loading with different device configurations
to check GPU/CPU compatibility, particularly for RTX 5090 sm_120 issues.
"""

import sys
import traceback
from PIL import Image
import numpy as np
from torch import device
from ultralytics import YOLO


def test_torch_basic():
    """Test basic PyTorch functionality"""
    print("=" * 50)
    print("Testing PyTorch Basic Functionality")
    print("=" * 50)

    try:
        import torch
        print(f"✓ PyTorch imported successfully")
        print(f"  - Version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device count: {torch.cuda.device_count()}")
            print(f"  - Current device: {torch.cuda.current_device()}")
            print(f"  - Device name: {torch.cuda.get_device_name()}")
        return True
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def test_yolo_cpu():
    """Test YOLOv5 loading on CPU"""
    print("\n" + "=" * 50)
    print("Testing YOLOv5 on CPU")
    print("=" * 50)

    try:
        import torch
        print("Attempting to load YOLOv5 on CPU...")

        # Force CPU device
        model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True,
            trust_repo=True,
            device='cpu'
        )

        print("✓ YOLOv5 loaded successfully on CPU")
        print(f"  - Model type: {type(model)}")
        print(f"  - Model device: {next(model.parameters()).device}")

        # Test inference on dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        results = model(dummy_image)
        print(f"✓ Inference test passed - {len(results.xyxy[0])} detections")

        return model, 'cpu'

    except Exception as e:
        print(f"✗ YOLOv5 CPU loading failed: {e}")
        traceback.print_exc()
        return None, None

def test_yolo_gpu():
    """Test YOLOv5 loading on GPU"""
    print("\n" + "=" * 50)
    print("Testing YOLOv5 on GPU")
    print("=" * 50)

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping GPU test")
            return None, None

        print("Attempting to load YOLOv5 on GPU...")

        # Try GPU device
        # Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
        model = YOLO("yolo11n.yaml")  # build a new model from YAML
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

        # model = torch.hub.load(
        #     'ultralytics/yolov5',
        #     'yolov5s',
        #     pretrained=True,
        #     trust_repo=True,
        #     device='cuda',
        #     force_reload=True
        # )

        print("✓ YOLOv11 loaded successfully")
        print(f"  - Model type: {type(model)}")
        print(f"  - Model device: {next(model.parameters()).device}")

        # Test inference on dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        results = model(dummy_image, device=0)

        return model, 0

    except Exception as e:
        print(f"✗ YOLOv5 GPU loading failed: {e}")
        print("This is likely due to RTX 5090 sm_120 compatibility issues")
        traceback.print_exc()
        return None, None

def test_human_detection(model, device_str):
    """Test human detection on a sample image"""
    print("\n" + "=" * 50)
    print(f"Testing Human Detection on {device_str}")
    print("=" * 50)

    try:
        # Create a more realistic test image (people-like shapes)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some simple shapes that might be detected as people
        test_image[100:400, 200:250] = [128, 64, 32]  # Simple rectangle
        test_image[150:350, 350:400] = [64, 128, 96]  # Another rectangle

        pil_image = Image.fromarray(test_image)

        # Run detection
        results = model.predict(pil_image, device=device_str, conf=0.25, iou=0.45, max_det=10, save=True)
        detections = results.xyxy[0].cpu().numpy()

        # Filter for person class (class 0 in COCO)
        person_detections = []
        for *xyxy, conf, cls in detections:
            if int(cls) == 0:  # person class
                person_detections.append((*xyxy, conf))

        print(f"✓ Human detection test completed")
        print(f"  - Total detections: {len(detections)}")
        print(f"  - Person detections: {len(person_detections)}")

        if person_detections:
            print("  - Person detection details:")
            for i, (*bbox, conf) in enumerate(person_detections):
                print(f"    Person {i+1}: bbox={bbox}, confidence={conf:.3f}")

        return True

    except Exception as e:
        print(f"✗ Human detection test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("YOLOv5 Loading and Compatibility Test")
    print("=" * 60)

    # Test basic PyTorch
    if not test_torch_basic():
        print("\n✗ PyTorch basic test failed - cannot proceed")
        return

    # Test different device configurations
    working_model = None
    working_device = None

    # Try CPU first (most likely to work)
    # model_cpu, device_cpu = test_yolo_cpu()
    # if model_cpu is not None:
    #     working_model = model_cpu
    #     working_device = device_cpu
    #     test_human_detection(model_cpu, device_cpu)

    # Try GPU
    model_gpu, device_gpu = test_yolo_gpu()
    if model_gpu is not None:
        working_model = model_gpu
        working_device = device_gpu
        test_human_detection(model_gpu, device_gpu)

    # Try auto device selection
    # model_auto, device_auto = test_yolo_auto()
    # if model_auto is not None and working_model is None:
    #     working_model = model_auto
    #     working_device = device_auto
    #     test_human_detection(model_auto, device_auto)

    # Summary
    # print("\n" + "=" * 60)
    # print("Test Summary")
    # print("=" * 60)
# 
    # if working_model is not None:
    #     print(f"✓ YOLOv5 is working on device: {working_device}")
    #     print("✓ Ready for human pose estimation pipeline")
# 
    #     if working_device == 'cpu':
    #         print("\nNote: Using CPU for YOLOv5 due to GPU compatibility issues")
    #         print("This is expected with RTX 5090 and current PyTorch versions")
    #         print("JAX pose estimation can still use GPU")
# 
    # else:
    #     print("✗ YOLOv5 failed to load on any device")
    #     print("Check PyTorch installation and dependencies")
# 
    # print(f"\nRecommendation:")
    # if working_device == 'cpu':
    #     print("- Use CPU for YOLOv5 human detection")
    #     print("- Use GPU for JAX pose estimation")
    #     print("- This hybrid approach should work well")
    # elif working_device in ['cuda', 'cuda:0']:
    #     print("- GPU is working fine for YOLOv5")
    #     print("- Use GPU for both human detection and pose estimation")
    # else:
    #     print("- Consider updating PyTorch or using CPU fallback")

if __name__ == "__main__":
    main()