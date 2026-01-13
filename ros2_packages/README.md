# ROS2 Packages for Uncertainty Quantification

This directory contains ROS2 packages for deploying uncertainty quantification models in real-time applications.

## Directory Structure

```
ros2_packages/
├── uq_inference/              # Main UQ inference package
│   ├── package.xml            # ROS2 package metadata
│   ├── setup.py               # Python package setup
│   ├── resource/              # ROS2 resource markers
│   ├── config/                # Configuration files
│   │   └── default_params.yaml
│   ├── launch/                # Launch files (to be added)
│   └── uq_inference/          # Python package
│       ├── __init__.py
│       ├── uq_wrapper.py      # Wrapper around main UQ codebase
│       └── image_processor_node.py  # ROS2 node for image processing
└── README.md                  # This file
```

## Building the Workspace

### First Time Setup

1. **Start the Docker container:**
   ```bash
   cd docker
   ./run.sh
   ./shell.sh
   ```

2. **Create the ROS2 workspace directory structure:**
   ```bash
   cd /workspace
   mkdir -p ros2_ws/src
   ```

3. **Link the ROS2 packages into the workspace:**
   ```bash
   cd /workspace/ros2_ws/src
   ln -s ../../ros2_packages/uq_inference .
   ```

4. **Build the workspace:**
   ```bash
   cd /workspace/ros2_ws
   colcon build
   ```

5. **Source the workspace:**
   ```bash
   source install/setup.bash
   ```

   Note: This is done automatically when you open a new shell (configured in the Docker entrypoint).

### Rebuilding After Changes

```bash
cd /workspace/ros2_ws
colcon build --packages-select uq_inference
source install/setup.bash
```

## Using the UQ Inference Node

### Running the Node

```bash
# With default parameters
ros2 run uq_inference image_processor

# With custom parameters
ros2 run uq_inference image_processor --ros-args \
  -p model_path:=/models/my_model.pkl \
  -p model_type:=ResNet \
  -p dataset:=CIFAR-10 \
  -p device:=cuda

# With a parameter file
ros2 run uq_inference image_processor --ros-args \
  --params-file /workspace/ros2_packages/uq_inference/config/default_params.yaml
```

### Topics

**Subscribed Topics:**
- `/camera/image_raw` (sensor_msgs/Image): Input images from remote PC

**Published Topics:**
- `/uq/uncertainty_score` (std_msgs/Float32): Uncertainty score for each image
- `/uq/prediction` (std_msgs/String): Model prediction

### Testing the Node

You can test the node by publishing images from another terminal or PC:

```bash
# Install image publisher tools if needed
sudo apt-get install ros-jazzy-image-publisher

# Publish a test image
ros2 run image_publisher image_publisher_node /path/to/image.jpg
```

Or use a camera:
```bash
# Install camera drivers if needed
sudo apt-get install ros-jazzy-usb-cam

# Start camera node
ros2 run usb_cam usb_cam_node_exe
```

## Implementing the UQ Wrapper

The `uq_wrapper.py` file currently contains placeholder implementations. To integrate with your trained models:

1. **Update the `initialize()` method** to load your trained model:
   ```python
   def initialize(self):
       from src.models.wrapper import load_model
       self.model = load_model(self.model_path, model_type=self.model_type)
       self.is_initialized = True
   ```

2. **Update the `preprocess_image()` method** with your preprocessing pipeline:
   ```python
   def preprocess_image(self, image):
       # Add your preprocessing steps
       # - Resize, normalize, convert to JAX/PyTorch format
       pass
   ```

3. **Update the `compute_uncertainty()` method** to use your UQ method:
   ```python
   def compute_uncertainty(self, image):
       # Run your UQ scoring method
       # e.g., SLU, SCOD, ensemble, etc.
       pass
   ```

## Network Configuration for Multi-PC Setup

### On This Workstation (Processing PC)

The Docker container uses `network_mode: host`, so it shares the host network. Make sure:

1. **Check your IP address:**
   ```bash
   ip addr show
   ```

2. **Verify ROS2 can discover nodes:**
   ```bash
   ros2 node list
   ```

### On Remote PC (Camera PC)

1. **Install ROS2 Jazzy** (same version as workstation)

2. **Set ROS_DOMAIN_ID** (optional, for network isolation):
   ```bash
   export ROS_DOMAIN_ID=42
   ```
   Make sure both PCs use the same domain ID.

3. **Verify network connectivity:**
   ```bash
   # Should see nodes from workstation
   ros2 node list
   ```

4. **Publish images:**
   ```bash
   ros2 run your_camera_package camera_node
   ```

### Troubleshooting Network Issues

If nodes on different PCs can't see each other:

1. **Check firewall settings:**
   ```bash
   # On Ubuntu, allow ROS2 ports
   sudo ufw allow from <remote_pc_ip>
   ```

2. **Check multicast connectivity:**
   ```bash
   # Install tools
   sudo apt-get install iputils-ping avahi-utils

   # Test multicast
   ping -c 3 224.0.0.1
   ```

3. **Use ROS_LOCALHOST_ONLY for debugging:**
   ```bash
   # On workstation, disable localhost-only mode
   export ROS_LOCALHOST_ONLY=0
   ```

## Development Workflow

1. **Make changes** to the Python code in `ros2_packages/uq_inference/uq_inference/`
2. **Rebuild** the package: `colcon build --packages-select uq_inference`
3. **Source** the workspace: `source install/setup.bash`
4. **Test** your changes: `ros2 run uq_inference image_processor`

For development, you can also run the node without installing:
```bash
cd /workspace/ros2_packages/uq_inference
python3 uq_inference/image_processor_node.py
```

## Next Steps

1. **Implement the UQ wrapper** (`uq_wrapper.py`) with your actual model loading and inference code
2. **Create launch files** for easier deployment
3. **Add custom message types** if you need more complex result structures
4. **Add visualization** using RViz if needed
5. **Implement model caching** for faster initialization
6. **Add performance monitoring** and profiling

## Resources

- [ROS2 Documentation](https://docs.ros.org/en/jazzy/)
- [colcon Documentation](https://colcon.readthedocs.io/)
- [cv_bridge Tutorial](https://github.com/ros-perception/vision_opencv/tree/jazzy/cv_bridge)
