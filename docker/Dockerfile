FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale for ROS2
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    build-essential \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-zstandard \
    sudo \
    gosu \
    net-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Enable Ubuntu Universe repository
RUN add-apt-repository universe

# Install ROS 2 Jazzy apt repository
RUN apt-get update && apt-get install -y curl \
    && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm /tmp/ros2-apt-source.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update apt cache and upgrade
RUN apt-get update && apt-get upgrade -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Jazzy Desktop (includes RViz, demos, tutorials)
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions \
    ros-jazzy-cv-bridge \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rmw-zenoh-cpp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create Python virtual environment directory
# The environment will be created at runtime or by setup script
ENV VIRTUAL_ENV=/workspace/unc
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set default user/group IDs (will be overridden at runtime)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# Create entrypoint script for user setup
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy CycloneDDS configuration
# COPY docker/cyclonedds.xml /tmp/cyclonedds.xml

# Install additional Python dependencies that might be needed for JAX GPU support
RUN apt-get update && apt-get install -y \
    python3-numpy \
    python3-scipy \
    libopenblas-dev \
    liblapack-dev \
    iputils-ping \
    netcat-openbsd \
    net-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

# Set entrypoint to handle user creation and setup
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set default command
CMD ["/bin/bash"]
