#!/bin/bash

# Script to set up the Python virtual environment inside the Docker container
# Run this script INSIDE the container after starting it

set -e

echo "Setting up Python virtual environment 'unc'..."

# Check if we're in the workspace directory
if [ ! -f "train_model.py" ]; then
    echo "Error: This script should be run from the /workspace directory"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "unc" ]; then
    echo "Removing existing virtual environment..."
    rm -rf unc
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv unc

# Activate virtual environment
source unc/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt or setup.py exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing package with setup.py or pyproject.toml..."
    pip install -e .
else
    echo "No requirements.txt or setup.py found."
    echo "Installing common dependencies for JAX/Flax and PyTorch..."

    # Install JAX with CUDA support
    echo "Installing JAX with CUDA support..."
    pip install --upgrade "jax[cuda12]"

    # Install other common dependencies
    echo "Installing other dependencies..."
    pip install flax optax torch torchvision numpy scipy matplotlib
fi

# Always install the package in editable mode if pyproject.toml exists
# This ensures human_pose_pipeline and other modules are importable
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    echo "Installing package in editable mode..."
    pip install -e .
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source unc/bin/activate"
echo ""
echo "Or add this to your .bashrc in the container:"
echo "  echo 'source /workspace/unc/bin/activate' >> ~/.bashrc"
