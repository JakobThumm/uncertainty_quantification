#!/bin/bash

# Script to open a shell in the running Docker container

set -e

# Get username (default to current user)
USERNAME=${USERNAME:-$(whoami)}

echo "Opening shell in Docker container as user '$USERNAME'..."

docker exec -it -u "$USERNAME" uq-ros2-cuda bash
