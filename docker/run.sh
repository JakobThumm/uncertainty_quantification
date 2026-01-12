#!/bin/bash

# Script to run the Docker container for uncertainty quantification

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export current user information for docker-compose
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export USERNAME=$(whoami)

echo "Starting Docker container for uncertainty quantification..."
echo "User: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"

# Run using docker-compose
cd "$SCRIPT_DIR"
docker-compose up -d

echo ""
echo "Container started with Zenoh Router active (Listening on 0.0.0.0:7447)!"
echo ""
echo "To access the container shell, use:"
echo "  docker exec -it uq-ros2-cuda bash"
echo ""
echo "To view router logs:"
echo "  docker logs uq-ros2-cuda"
echo ""
echo "To stop the container:"
echo "  docker-compose down"