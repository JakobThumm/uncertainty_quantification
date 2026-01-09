# Docker Setup for Uncertainty Quantification

This directory contains Docker configuration files for running the uncertainty quantification codebase with NVIDIA CUDA GPU support and ROS2 Jazzy.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose v2.0 or later
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support

### Installing NVIDIA Docker Runtime

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

## Quick Start

1. **Build the Docker image:**
   ```bash
   cd docker
   chmod +x build.sh
   ./build.sh
   ```

2. **Start the container:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

3. **Open a shell in the container:**
   ```bash
   chmod +x shell.sh
   ./shell.sh
   ```

4. **Set up the Python environment (inside the container):**
   ```bash
   # Inside the container
   bash docker/setup_env.sh

   # Or use the existing setup script
   bash bash/setup.sh

   # IMPORTANT: After setting up the environment, install the package in editable mode
   # This makes human_pose_pipeline and other modules importable
   source unc/bin/activate
   pip install -e .
   ```

5. **Activate the environment (automatic):**

   The virtual environment is **automatically activated** when you open a shell in the container. You should see `(unc)` in your prompt.

   If for some reason you need to manually activate it:
   ```bash
   source unc/bin/activate
   ```

   **Note:** If you see `ModuleNotFoundError: No module named 'human_pose_pipeline'`, you need to install the package:
   ```bash
   pip install -e .
   ```

## User Permissions

**Important:** The container runs as your host user (not root) to avoid permission issues with files created in the mounted workspace.

When you build and run the container:
- The build and run scripts automatically detect your user ID (UID), group ID (GID), and username
- A matching user is created inside the container with the same UID/GID
- All files created in the mounted `/workspace` directory will be owned by your host user
- The container user has sudo privileges (no password required) for installing additional packages

This means you can seamlessly edit files both inside the container and on your host system without permission conflicts.

## What's Included

The Docker image includes:

- **Base:** NVIDIA CUDA 12.6.0 with cuDNN on Ubuntu 24.04
- **ROS2 Jazzy:** Full desktop installation with RViz, demos, and tutorials
- **Development tools:** Build essentials, Python 3, git, vim, sudo, etc.
- **Python environment:** Virtual environment setup for JAX/Flax and PyTorch
- **User configuration:** Automatic user creation matching host UID/GID

## Usage

### Working with the Container

The codebase is mounted at `/workspace` in the container, so all changes you make are reflected on your host system.

```bash
# Enter the container
./shell.sh

# The virtual environment is automatically activated!
# You should see (unc) in your prompt

# Run training
python train_model.py --dataset MNIST --likelihood classification --model MLP --default_hyperparams

# Run scoring
python score_model.py --ID_dataset FMNIST --OOD_datasets MNIST FMNIST-R --model LeNet --score local_ensemble
```

### ROS2 Usage

ROS2 Jazzy is automatically sourced in the container. You can use ROS2 commands directly:

```bash
# Check ROS2 installation
ros2 --help

# Run RViz (requires X11 forwarding - see GUI section below)
rviz2
```

### GPU Verification

Verify GPU access inside the container:

```bash
nvidia-smi

# For JAX
python -c "import jax; print(jax.devices())"

# For PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Stopping the Container

```bash
cd docker
docker-compose down
```

## GUI Applications (RViz, etc.)

To use GUI applications like RViz, you need to enable X11 forwarding:

1. **On your host machine:**
   ```bash
   xhost +local:docker
   ```

2. **Uncomment the X11 volumes in docker-compose.yml:**
   ```yaml
   volumes:
     - /tmp/.X11-unix:/tmp/.X11-unix:rw
     - $HOME/.Xauthority:/home/${USERNAME:-user}/.Xauthority:rw
   ```

3. **Rebuild and restart the container:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## File Structure

```
docker/
├── Dockerfile           # Main Docker image definition
├── docker-compose.yml   # Container orchestration configuration
├── entrypoint.sh       # Container entrypoint for user setup
├── build.sh            # Script to build the image
├── run.sh              # Script to start the container
├── shell.sh            # Script to open a shell in the container
├── setup_env.sh        # Script to set up Python environment inside container
├── .dockerignore       # Files to exclude from Docker build context
└── README.md           # This file
```

### How It Works

1. **build.sh** - Detects your host user credentials (UID, GID, username) and passes them to Docker Compose as build arguments
2. **Dockerfile** - Installs system dependencies, ROS2 Jazzy, and copies the entrypoint script
3. **entrypoint.sh** - Runs when the container starts, creates a user matching your host credentials, and sets up the environment
4. **run.sh** - Starts the container with your user credentials passed as environment variables
5. **shell.sh** - Opens an interactive shell in the running container (as your user, not root)

## Troubleshooting

### GPU Not Detected

If `nvidia-smi` doesn't work in the container:
- Ensure nvidia-docker2 is installed on the host
- Restart Docker daemon: `sudo systemctl restart docker`
- Check Docker supports GPU: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`

### JAX CUDA Issues

If JAX doesn't detect GPU:
```bash
# Inside container
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### CuDNN Version Issues

The image includes cuDNN. If you need a specific version:
```bash
# Check installed version
dpkg -l | grep cudnn

# The base image should have cuDNN 9.x compatible with CUDA 12.6
```

### Permission Issues

The container automatically runs as your host user, so permission issues should not occur. However, if you do encounter permission problems:

1. **Verify user setup:** Inside the container, run `id` and compare with `id` on your host
2. **Check file ownership:** `ls -la /workspace` should show files owned by your user
3. **Rebuild if needed:** If user setup failed, try rebuilding:
   ```bash
   docker-compose down
   docker-compose build --no-cache
   ./run.sh
   ```

4. **Manual fix:** If files have wrong ownership on the host:
   ```bash
   # On host system
   sudo chown -R $(id -u):$(id -g) /path/to/workspace
   ```

### Memory Issues

For large models, you may need to increase Docker's memory limit:
- Docker Desktop: Settings → Resources → Memory
- Linux: Modify `/etc/docker/daemon.json`

## Advanced Configuration

### Custom Python Packages

Add packages to `requirements.txt` in the repo root, then:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Multiple GPUs

The configuration uses all available GPUs by default. To specify GPUs:

Edit docker-compose.yml:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1  # Use only GPUs 0 and 1
```

### Persistent Bash History

The container preserves bash history in a Docker volume named `bash_history`.

### Network Configuration

The container uses `network_mode: host` for ROS2 communication. If you need different networking:
```yaml
network_mode: bridge
ports:
  - "8888:8888"  # Example: Jupyter notebook
```

## Development Workflow

Recommended workflow:

1. Start container: `./run.sh`
2. Open shell: `./shell.sh` (virtual environment auto-activates)
3. Make changes in your IDE on the host (changes are reflected immediately)
4. Run/test in the container
5. Commit changes on host or in container (git is available in both)

**Note:** Since the container runs as your host user, you have full read/write access to all files in the workspace from both inside and outside the container. No permission conflicts!

## Cleaning Up

```bash
# Stop and remove container
docker-compose down

# Remove image
docker rmi uncertainty-quantification:latest

# Remove all unused Docker resources
docker system prune -a
```
