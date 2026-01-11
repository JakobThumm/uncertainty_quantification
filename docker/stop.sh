#!/bin/bash

# Script to stop and remove the Docker container for uncertainty quantification

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Stopping and removing Docker container..."

# Stop and remove using docker-compose
cd "$SCRIPT_DIR"
docker-compose down

echo ""
echo "Container stopped and removed successfully!"
echo ""
echo "To start the container again, use:"
echo "  ./run.sh"
echo ""
