#!/bin/bash

# Script to build the Docker image for uncertainty quantification

set -e

# Export current user information for docker-compose
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export USERNAME=$(whoami)

echo "Building Docker image for uncertainty quantification with CUDA and ROS2 Jazzy..."
echo "User: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build using docker-compose
cd "$SCRIPT_DIR"
docker-compose build

echo ""
echo "Build complete!"
echo "To run the container, use: ./run.sh"
