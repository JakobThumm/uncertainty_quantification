#!/usr/bin/env python3
"""
Test script for loading H36M dataset with the extracted data
This verifies the dataset is correctly loaded before implementing Experiment 2
"""

import os
import sys
import numpy as np

# Add root directory to path to access src
sys.path.append('../..')

from src.datasets.wrapper import dataloader_from_string

def test_h36m_loading():
    """Test H36M dataset loading with the extracted data"""
    print("=" * 60)
    print("Testing H36M Dataset Loading")
    print("=" * 60)

    # Path to the extracted H36M data
    h36m_path = "../../datasets/H36M/extracted"

    print(f"Dataset path: {h36m_path}")
    print(f"Path exists: {os.path.exists(h36m_path)}")

    # Check subjects
    subjects = ['S1', 'S5', 'S11']
    for subject in subjects:
        subject_path = os.path.join(h36m_path, subject)
        poses_path = os.path.join(subject_path, 'Poses_D2_Positions')
        videos_path = os.path.join(subject_path, 'Videos')

        print(f"\n{subject}:")
        print(f"  Subject path exists: {os.path.exists(subject_path)}")
        print(f"  Poses path exists: {os.path.exists(poses_path)}")
        print(f"  Videos path exists: {os.path.exists(videos_path)}")

        if os.path.exists(poses_path):
            cdf_files = [f for f in os.listdir(poses_path) if f.endswith('.cdf')]
            print(f"  CDF files: {len(cdf_files)}")
            if cdf_files:
                print(f"    First: {cdf_files[0]}")

        if os.path.exists(videos_path):
            video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
            print(f"  Video files: {len(video_files)}")
            if video_files:
                print(f"    First: {video_files[0]}")

def test_dataloader_creation():
    """Test creating dataloader with correct path"""
    print("\n" + "=" * 60)
    print("Testing H36M Dataloader Creation")
    print("=" * 60)

    # Path to the extracted H36M data
    h36m_path = "../../datasets/H36M/extracted"

    try:
        # Try to create dataloader with custom path
        # Note: We need to modify the dataset loader to accept custom paths
        print("Attempting to create H36M dataloader...")

        # For now, just test if we can import and the path structure is correct
        from src.datasets.h36m import Human36mDataset

        print(f"✓ H36M dataset class imported successfully")

        # Try to instantiate with our extracted data path
        dataset = Human36mDataset(
            base_directory=h36m_path,
            split='train',
            num_frames_per_video=5
        )

        print(f"✓ Dataset created successfully")
        print(f"  Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            print("✓ Testing first sample...")
            sample = dataset[0]
            print(f"  Sample type: {type(sample)}")
            if isinstance(sample, tuple):
                frame, pose = sample
                print(f"  Frame shape: {frame.shape}")
                print(f"  Pose shape: {pose.shape}")
            print("✓ First sample loaded successfully")

        return True

    except Exception as e:
        print(f"✗ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing H36M Dataset Access")
    print("This verifies the dataset is ready for Experiment 2")
    print()

    # Test basic file structure
    test_h36m_loading()

    # Test dataloader creation
    success = test_dataloader_creation()

    print("\n" + "=" * 60)
    print("H36M Dataset Test Summary")
    print("=" * 60)

    if success:
        print("✓ H36M dataset is ready for Experiment 2!")
        print("Next steps:")
        print("  - Implement JAX evaluation metrics (MPJPE, PCK)")
        print("  - Create Experiment 2 evaluation pipeline")
        print("  - Test pose estimation with real H36M examples")
    else:
        print("✗ H36M dataset loading failed")
        print("Please check the dataset path and file structure")

if __name__ == "__main__":
    main()