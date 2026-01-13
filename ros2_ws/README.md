# ROS2 Workspace

This is the ROS2 workspace directory for the uncertainty quantification project.

## Quick Setup

Run the setup script to initialize the workspace:

```bash
cd /workspace/ros2_ws
bash setup_workspace.sh
```

This will:
1. Create the `src/` directory
2. Link packages from `../ros2_packages/`
3. Build the workspace with `colcon build`

## Structure

```
ros2_ws/
├── src/                  # Symlinks to ROS2 packages (auto-generated)
├── build/                # Build artifacts (git-ignored)
├── install/              # Install space (git-ignored)
├── log/                  # Build logs (git-ignored)
├── setup_workspace.sh    # Setup script
└── README.md             # This file
```

## Manual Setup

If you prefer to set up manually:

```bash
# Create source directory
mkdir -p src

# Link packages
cd src
ln -s ../../ros2_packages/uq_inference .

# Build
cd ..
colcon build

# Source the workspace
source install/setup.bash
```

## Rebuilding

After making changes to packages:

```bash
# Rebuild specific package
colcon build --packages-select uq_inference

# Rebuild all packages
colcon build

# Clean build (start fresh)
rm -rf build install log
colcon build
```

## Using the Workspace

The workspace is automatically sourced in your `.bashrc` (when using the Docker container).

To manually source:
```bash
source /workspace/ros2_ws/install/setup.bash
```

For more information, see `../ros2_packages/README.md`
