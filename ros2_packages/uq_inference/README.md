# UQ Inference - Human Pose Pipeline ROS2 Node

This package provides a ROS2 node for real-time human pose estimation and motion prediction with uncertainty quantification.

## Features

- **Real-time 2D pose estimation** with uncertainty using JAX models
- **3D pose reconstruction** via stereo triangulation or RGB-D depth lifting
- **Motion prediction** with uncertainty quantification
- **Out-of-distribution (OOD) detection** for both pose and motion
- **Uncertainty visualization** through covariance matrices and set radii

## System Requirements

- ROS2 (Humble or newer)
- CUDA-capable GPU (recommended)
- Python 3.8+
- JAX with CUDA support
- PyTorch with CUDA support

## Installation

### 1. Build Custom Messages

First, build the custom message package:

```bash
cd /home/thumm/code/sketching_lanczos/uncertainty_quantification/ros2_ws
source /opt/ros/humble/setup.bash  # Or your ROS2 distro

# Build uq_msgs package
colcon build --packages-select uq_msgs
source install/setup.bash
```

### 2. Build UQ Inference Package

```bash
# Build uq_inference package
colcon build --symlink-install
sed -i 's|#!/usr/bin/python3|#!/workspace/unc/bin/python3|g' install/uq_inference/lib/uq_inference/pose_pipeline
source install/setup.bash
```

## Configuration

### Operating Modes

The node supports two modes:

1. **Stereo Mode** (default): Uses two RGB cameras for 3D triangulation
   - Requires two synchronized RGB camera streams
   - Topics: `/realsense/camera_1/color/image_raw`, `/realsense/camera_2/color/image_raw`

2. **RGB-D Mode**: Uses single RGB-D camera with depth information
   - Requires synchronized color and depth streams
   - Topics: `/realsense/camera_1/color/image_raw`, `/realsense/camera_1/aligned_depth_to_color/image_raw`
   - Uses depth-based 3D lifting with uncertainty propagation via Jacobian
   - Fully implemented and ready to use

### Camera Setup

#### For Stereo Mode (Two RealSense Cameras)

You need two RealSense cameras streaming to separate topics. Make sure both cameras are:
- Time-synchronized (hardware sync or NTP)
- Publishing with SensorDataQoS profile
- Have valid camera calibration parameters

#### For RGB-D Mode (Single RealSense Camera)

One RealSense camera with aligned depth to color:
- Enable aligned depth stream in your camera driver
- Both color and depth should use SensorDataQoS profile
- Camera intrinsics are automatically obtained from camera_info topic

**How RGB-D Mode Works:**
1. Runs 2D pose estimation on the color image
2. For each detected 2D keypoint (u, v), reads the depth value Z at that pixel
3. Back-projects to 3D using: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
4. Propagates 2D uncertainty to 3D using the Jacobian of the back-projection
5. Handles invalid depth readings (0 values) gracefully

### Model Paths

The node requires several pre-trained models:

1. **Pose Estimation Model**: JAX RegressFlow model for 2D pose estimation
   - Default: `human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/jax_resnet50_regressflow`

2. **Motion Prediction Model**: DCT Pose Transformer
   - Default: `human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle`

3. **Camera Parameters**: Calibration file
   - Default: `human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/camera-parameters.json`

4. **OOD Score Functions** (optional):
   - Motion: `human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle`

Ensure all models are available in the workspace before starting the node.

## Usage

### Launch with Default Parameters (Stereo Mode)

```bash
ros2 launch uq_inference pose_pipeline.launch.py
```

### Launch with Custom Parameters

```bash
# Use RGB-D mode
ros2 launch uq_inference pose_pipeline.launch.py mode:=rgbd

# Disable OOD detection
ros2 launch uq_inference pose_pipeline.launch.py enable_ood:=false

# Use CPU instead of CUDA
ros2 launch uq_inference pose_pipeline.launch.py device:=cpu
```

### Run Node Directly

```bash
ros2 run uq_inference pose_pipeline
```

### With Custom Configuration File

```bash
ros2 run uq_inference pose_pipeline --ros-args --params-file src/uq_inference/config/pose_pipeline.yaml
```

## Published Topics

### `/uq/pose_3d` (uq_msgs/Pose3D)

3D pose estimation with uncertainty:
- `points_3d`: Flattened array of 3D joint positions [N_JOINTS * 3]
- `covariance_3d`: Flattened covariance matrices [N_JOINTS * 3 * 3]
- `n_joints`: Number of joints (13 for H36M)
- `is_ood`: Whether pose is out-of-distribution
- `ood_score`: OOD confidence score
- `human_detected`: Whether human was detected in frame

### `/uq/motion_prediction` (uq_msgs/MotionPrediction)

Motion prediction with uncertainty:
- `motion_predicted`: Predicted motion [PREDICTION_HORIZON * N_JOINTS * 3]
- `motion_covariance`: Covariance matrices [PREDICTION_HORIZON * N_JOINTS * 3 * 3]
- `set_radius`: Uncertainty set radius [PREDICTION_HORIZON * N_JOINTS]
- `prediction_horizon_length`: Number of future timesteps (10)
- `n_joints`: Number of joints (13)
- `is_ood`: Whether prediction is out-of-distribution
- `ood_score`: OOD confidence score
- `is_valid`: Whether prediction is valid (enough good poses in buffer)

## Subscribed Topics

### Stereo Mode
- `/realsense/camera_1/color/image_raw` (sensor_msgs/Image)
- `/realsense/camera_2/color/image_raw` (sensor_msgs/Image)

### RGB-D Mode
- `/realsense/camera_1/color/image_raw` (sensor_msgs/Image)
- `/realsense/camera_1/aligned_depth_to_color/image_raw` (sensor_msgs/Image)

## Parameters

See `config/pose_pipeline.yaml` for all available parameters.

Key parameters:
- `mode`: Operating mode ('stereo' or 'rgbd')
- `enable_ood`: Enable out-of-distribution detection
- `device`: Computation device ('cuda' or 'cpu')
- `pose_model_path`: Path to pose estimation model
- `motion_model_path`: Path to motion prediction model
- `camera_1_id`, `camera_2_id`: Camera IDs for calibration lookup

## Message Formats

### Pose3D Message Format

```
std_msgs/Header header
float64[] points_3d          # Shape: [N_JOINTS, 3] flattened
float64[] covariance_3d      # Shape: [N_JOINTS, 3, 3] flattened
int32 n_joints               # Number of joints
bool is_ood                  # Out-of-distribution flag
float64 ood_score            # OOD confidence
bool human_detected          # Human detection flag
```

### MotionPrediction Message Format

```
std_msgs/Header header
float64[] motion_predicted   # Shape: [HORIZON, N_JOINTS, 3] flattened
float64[] motion_covariance  # Shape: [HORIZON, N_JOINTS, 3, 3] flattened
float64[] set_radius         # Shape: [HORIZON, N_JOINTS] flattened
int32 prediction_horizon_length
int32 n_joints
bool is_ood
float64 ood_score
bool is_valid
```

## Architecture

The pipeline follows these steps:

1. **Image Acquisition**: Synchronized capture of stereo RGB or RGB-D images
2. **Human Detection**: YOLO-based human bounding box detection
3. **2D Pose Estimation**: RegressFlow model for 2D keypoint detection with uncertainty
4. **3D Reconstruction**:
   - Stereo: Triangulation from two camera views
   - RGB-D: Depth-based lifting (TODO)
5. **Pose Buffer Management**: Maintains history of recent poses
6. **Motion Prediction**: DCT Pose Transformer predicts future motion
7. **Uncertainty Quantification**: Sketched Lanczos for OOD detection
8. **Output Publishing**: Real-time streaming of results

## Performance Considerations

- **Frame Rate**: Processes every 2nd frame to match motion prediction frequency
- **GPU Memory**: CUDA operations for pose estimation and human detection
- **Buffer Requirements**: Maintains 50-frame history for motion prediction
- **QoS Settings**: Uses SensorDataQoS (best-effort) for input, reliable for output

## Troubleshooting

### "uq_msgs not found" Error
Build the uq_msgs package first and source the install directory.

### Model Not Found Errors
Verify all model paths are correct and files exist in the workspace.

### CUDA Out of Memory
Try reducing batch size or using CPU mode with `device:=cpu`.

### No Human Detected
Check YOLO confidence threshold and lighting conditions.

### Camera Synchronization Issues
Increase `slop` parameter in ApproximateTimeSynchronizer (currently 0.05s).

## Development

### Running Tests

```bash
colcon test --packages-select uq_inference
```

### Building in Debug Mode

```bash
colcon build --packages-select uq_inference --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

## Citation

If you use this work, please cite:
```
@inproceedings{your_paper,
  title={Sketched Lanczos Uncertainty Score},
  author={Your Name},
  year={2024}
}
```

## License

MIT

## Contact

For issues and questions, please open an issue on the repository.
