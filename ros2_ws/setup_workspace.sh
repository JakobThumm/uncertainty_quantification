#!/bin/bash
# Script to set up the ROS2 workspace

set -e

echo "Setting up ROS2 workspace..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROS2_WS="${SCRIPT_DIR}"

# Install ultralytics
cd "${ROS2_WS}/../ultralytics"
python -m pip install -e .
cd "${ROS2_WS}/.."
python -m pip install -e .

echo "ROS2 workspace: ${ROS2_WS}"

# Create src directory if it doesn't exist
if [ ! -d "${ROS2_WS}/src" ]; then
    echo "Creating src directory..."
    mkdir -p "${ROS2_WS}/src"
fi

# Link packages from ros2_packages directory
echo "Linking ROS2 packages..."
cd "${ROS2_WS}/src"

# Link uq_inference package
if [ ! -L "uq_inference" ]; then
    ln -s ../../ros2_packages/uq_inference .
    ln -s ../../ros2_packages/simple_image_publisher .
    echo "  Linked workspace packages"
else
    echo "  workspace packages already linked"
fi

# Build the workspace
echo "Building workspace..."
cd "${ROS2_WS}"
colcon build --symlink-install

echo ""
echo "Workspace setup complete!"
echo ""
echo "To use the workspace, run:"
echo "  source ${ROS2_WS}/install/setup.bash"
echo ""
echo "Or add it to your ~/.bashrc (already configured in Docker container)"
