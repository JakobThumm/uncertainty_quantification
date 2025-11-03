# Human Pose Pipeline Examples

This directory contains example scripts and tests for the human pose pipeline implementation.

## Files

### `test_pose_model.py`
- **Purpose**: Test script for loading and validating the pre-trained RegressFlow pose estimation model
- **Status**: ✅ Working
- **What it tests**:
  - Model loading from `human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/`
  - JAX model initialization and parameter loading
  - Forward pass with dummy input (1, 3, 256, 192) → (1, 34)
  - Dataset loader instantiation (H36M)

### Usage

Run from this directory:
```bash
# Activate environment
source ../../unc/bin/activate

# Run pose model test
python test_pose_model.py
```

### Expected Output
- ✅ Model loads successfully
- ✅ Forward pass produces (1, 34) output for 17 joints × 2 coordinates
- ⚠️  Dataset loader shows warning (expected until H36M download completes)

## Future Examples

Planned examples for the complete pipeline:

- `example_pose_estimation.py` - End-to-end 2D pose estimation with real images
- `example_3d_triangulation.py` - 3D pose estimation from stereo cameras
- `example_ood_detection.py` - OOD detection on pose estimates using SLU
- `example_motion_prediction.py` - Motion prediction from pose sequences
- `example_full_pipeline.py` - Complete pipeline demonstration

## Implementation Status

- ✅ JAX RegressFlow model loading and inference
- ✅ Model wrapper compatibility
- ⏳ Waiting for H36M dataset download
- 📋 Next: Implement evaluation pipeline (Experiment 2)