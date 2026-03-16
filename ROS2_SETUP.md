# ROS2 Integration Setup

This document explains the ROS2 integration for the uncertainty quantification project.

## Overview

The project now uses a monorepo structure that includes both the research codebase and ROS2 packages for deployment:

```
uncertainty_quantification/
├── src/                          # Original UQ research code
├── ros2_packages/                # ROS2 packages for deployment
│   └── uq_inference/             # Main inference package
│       ├── package.xml
│       ├── setup.py
│       ├── config/
│       │   └── default_params.yaml
│       └── uq_inference/
│           ├── uq_wrapper.py     # Wrapper around src/
│           └── image_processor_node.py
├── ros2_ws/                      # ROS2 workspace (build artifacts)
│   ├── src/                      # Symlinks to ros2_packages/
│   ├── build/                    # Build output (git-ignored)
│   ├── install/                  # Install space (git-ignored)
│   └── setup_workspace.sh
├── docker/                       # Docker configuration
└── ...                          # Other project files
```

## Key Design Decisions

1. **Monorepo Approach**: ROS2 packages live alongside research code, making it easier to maintain both together.

2. **Separation via ros2_packages/**: ROS2 packages are in a separate directory to keep them distinct from research code.

3. **ros2_ws/ for Build Artifacts**: The ROS2 workspace uses symlinks to `ros2_packages/`, keeping build artifacts separate from source code.

4. **UQ Wrapper Pattern**: The `uq_wrapper.py` module provides a clean interface between ROS2 and the research codebase, preventing circular dependencies.

## Quick Start

### 1. Build Docker Container

```bash
cd docker
./build.sh
./run.sh
./shell.sh
```

### 2. Set Up ROS2 Workspace

Inside the container:
```bash
cd /workspace/ros2_ws
bash setup_workspace.sh
source /workspace/ros2_ws/install/setup.bash
```

The workspace will be automatically sourced in new shells.

### 3. Run the Pose Pipeline Node
Run:
```bash
ros2 launch uq_inference pose_pipeline.launch.py
```