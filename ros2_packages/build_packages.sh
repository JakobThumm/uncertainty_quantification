#!/bin/bash
# Build script for UQ Inference ROS2 packages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Building UQ Inference ROS2 Packages${NC}"
echo -e "${GREEN}======================================${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS2 environment not sourced!${NC}"
    echo -e "${YELLOW}Please run: source /opt/ros/<distro>/setup.bash${NC}"
    exit 1
fi

echo -e "${GREEN}ROS2 Distribution: $ROS_DISTRO${NC}"

# Navigate to ros2_ws or create it
if [ ! -d "$WS_ROOT/ros2_ws" ]; then
    echo -e "${YELLOW}Creating ros2_ws directory...${NC}"
    mkdir -p "$WS_ROOT/ros2_ws/src"
fi

cd "$WS_ROOT/ros2_ws"

# Create symlinks to packages if they don't exist
if [ ! -L "src/uq_msgs" ]; then
    echo -e "${YELLOW}Creating symlink for uq_msgs...${NC}"
    ln -sf "$SCRIPT_DIR/uq_msgs" src/uq_msgs
fi

if [ ! -L "src/uq_inference" ]; then
    echo -e "${YELLOW}Creating symlink for uq_inference...${NC}"
    ln -sf "$SCRIPT_DIR/uq_inference" src/uq_inference
fi

echo ""
echo -e "${GREEN}Step 1: Building uq_msgs (custom messages)${NC}"
echo "========================================"
colcon build --packages-select uq_msgs --symlink-install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ uq_msgs built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build uq_msgs${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Sourcing uq_msgs...${NC}"
source install/setup.bash

echo ""
echo -e "${GREEN}Step 2: Building uq_inference (main package)${NC}"
echo "========================================"
colcon build --packages-select uq_inference --symlink-install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ uq_inference built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build uq_inference${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}To use the packages, source the workspace:${NC}"
echo -e "  ${GREEN}source $WS_ROOT/ros2_ws/install/setup.bash${NC}"
echo ""
echo -e "${YELLOW}To run the node:${NC}"
echo -e "  ${GREEN}ros2 launch uq_inference pose_pipeline.launch.py${NC}"
echo ""
