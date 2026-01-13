# Quick Start Guide - ROS2 Human Pose Pipeline

This guide will help you quickly set up and run the human pose estimation and motion prediction pipeline.

## Prerequisites

- ROS2 (Humble or newer) installed
- CUDA-capable GPU
- Two RealSense cameras (for stereo mode) or one RGB-D RealSense camera
- Pre-trained models in the workspace

## Step 1: Build the Packages

```bash
# From anywhere in the workspace
cd /home/thumm/code/sketching_lanczos/uncertainty_quantification/ros2_packages
./build_packages.sh
```

This will:
1. Build the `uq_msgs` custom message package
2. Build the `uq_inference` main package
3. Set up all necessary symlinks

## Step 2: Source the Workspace

```bash
source /home/thumm/code/sketching_lanczos/uncertainty_quantification/ros2_ws/install/setup.bash
```

Add this to your `~/.bashrc` for convenience:
```bash
echo "source /home/thumm/code/sketching_lanczos/uncertainty_quantification/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Step 3: Configure Your Setup

### For Stereo Mode (Two Cameras)

Edit `uq_inference/config/pose_pipeline.yaml`:
```yaml
mode: 'stereo'
camera_1_color_topic: '/realsense/camera_1/color/image_raw'
camera_2_color_topic: '/realsense/camera_2/color/image_raw'
```

Make sure your camera topics match your RealSense driver configuration.

### For RGB-D Mode (Single Camera)

Edit `uq_inference/config/pose_pipeline.yaml`:
```yaml
mode: 'rgbd'
rgbd_color_topic: '/realsense/camera_1/color/image_raw'
rgbd_depth_topic: '/realsense/camera_1/aligned_depth_to_color/image_raw'
rgbd_info_topic: '/realsense/camera_1/aligned_depth_to_color/camera_info'
```

**RGB-D Mode Features:**
- ✅ Fully implemented depth-based 3D lifting
- ✅ Automatic camera intrinsics from camera_info
- ✅ Uncertainty propagation via Jacobian
- ✅ Handles invalid depth readings gracefully

**When to use RGB-D vs Stereo:**
- Use **RGB-D** if: Single camera, indoor environment, subjects within 0.3-10m
- Use **Stereo** if: Longer range needed, outdoor environment, or already have stereo setup

See `RGBD_DETAILS.md` for implementation details.

## Step 4: Start Your Cameras

Make sure your RealSense camera(s) are publishing on the correct topics with SensorDataQoS.

Check available topics:
```bash
ros2 topic list
```

You should see:
- `/realsense/camera_1/color/image_raw`
- `/realsense/camera_2/color/image_raw` (for stereo mode)

## Step 5: Launch the Pipeline

```bash
ros2 launch uq_inference pose_pipeline.launch.py
```

Or with custom parameters:
```bash
ros2 launch uq_inference pose_pipeline.launch.py mode:=stereo enable_ood:=true device:=cuda
```

## Step 6: Monitor the Output

In separate terminals:

### View Pose Estimates
```bash
ros2 topic echo /uq/pose_3d
```

### View Motion Predictions
```bash
ros2 topic echo /uq/motion_prediction
```

### Check Node Status
```bash
ros2 node info /pose_pipeline_node
```

### Monitor Frame Rate
```bash
ros2 topic hz /uq/pose_3d
```

## Output Data Structure

### Pose3D Message
- **points_3d**: 3D joint positions, shape [13, 3] flattened to [39]
  - Joints: Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle
- **covariance_3d**: Covariance matrices [13, 3, 3] flattened to [117]
- **is_ood**: Boolean flag for out-of-distribution detection
- **human_detected**: Whether a human was detected in the frame

### MotionPrediction Message
- **motion_predicted**: Predicted future poses [10, 13, 3] flattened to [390]
  - 10 future timesteps, 13 joints, 3D coordinates
- **motion_covariance**: Uncertainty matrices [10, 13, 3, 3] flattened
- **set_radius**: Uncertainty radius for each joint [10, 13] flattened to [130]
- **is_valid**: Whether prediction is based on sufficient good poses

## Accessing Data in Python

```python
import rclpy
from rclpy.node import Node
from uq_msgs.msg import Pose3D, MotionPrediction
import numpy as np

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        self.pose_sub = self.create_subscription(
            Pose3D, '/uq/pose_3d', self.pose_callback, 10)
        self.motion_sub = self.create_subscription(
            MotionPrediction, '/uq/motion_prediction', self.motion_callback, 10)

    def pose_callback(self, msg):
        # Reshape to original dimensions
        points = np.array(msg.points_3d).reshape(msg.n_joints, 3)
        covariance = np.array(msg.covariance_3d).reshape(msg.n_joints, 3, 3)

        print(f"Pose detected: OOD={msg.is_ood}, Human={msg.human_detected}")
        print(f"Nose position: {points[0]}")

    def motion_callback(self, msg):
        # Reshape to original dimensions
        motion = np.array(msg.motion_predicted).reshape(
            msg.prediction_horizon_length, msg.n_joints, 3)
        radius = np.array(msg.set_radius).reshape(
            msg.prediction_horizon_length, msg.n_joints)

        print(f"Motion prediction: valid={msg.is_valid}, OOD={msg.is_ood}")
        print(f"Future position (t=1): {motion[0]}")

def main():
    rclpy.init()
    node = DataSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting

### No messages received
- Check camera topics are publishing: `ros2 topic list`
- Verify QoS compatibility: cameras should use SensorDataQoS
- Check node is running: `ros2 node list`

### "uq_msgs not found" error
- Rebuild packages: `./build_packages.sh`
- Source workspace: `source ros2_ws/install/setup.bash`

### CUDA out of memory
- Use CPU mode: `ros2 launch uq_inference pose_pipeline.launch.py device:=cpu`
- Close other GPU applications

### Low frame rate
- Check GPU utilization: `nvidia-smi`
- Verify camera streams are 30fps
- Node processes every 2nd frame by design

### No human detected
- Check lighting conditions
- Adjust YOLO threshold in h36m_settings.py
- Verify camera is pointing at person

## Docker Usage

If running in Docker, make sure:
1. X11 forwarding is configured for visualization
2. GPU passthrough is enabled (`--gpus all`)
3. Network mode allows topic discovery (`--network host`)

Example:
```bash
docker run --gpus all --network host \
  -v /path/to/models:/workspace/models \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  your_image_name
```

## Performance Tips

1. **Frame Rate**: Pipeline processes every 2nd frame (by design for motion prediction)
2. **Buffering**: Requires 50 frames of history before motion predictions are valid
3. **GPU**: Use CUDA for 10-20x speedup over CPU
4. **Sync**: Keep camera time sync tight (<50ms) for best triangulation

## Next Steps

- Configure camera calibration parameters for your specific cameras
- Adjust OOD thresholds based on your use case
- Implement visualization nodes to display results
- Add recording functionality for data collection

## Support

For issues, check:
1. ROS2 logs: `ros2 run uq_inference pose_pipeline 2>&1 | tee pipeline.log`
2. Camera diagnostics: `ros2 topic hz /realsense/camera_1/color/image_raw`
3. Node parameters: `ros2 param list /pose_pipeline_node`

Refer to the full README.md for detailed documentation.
