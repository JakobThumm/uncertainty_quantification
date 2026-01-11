#!/bin/bash
set -e

# Get user information from environment variables
USER_ID=${LOCAL_USER_ID:-1000}
GROUP_ID=${LOCAL_GROUP_ID:-1000}
USERNAME=${LOCAL_USERNAME:-user}

# Create group if it doesn't exist
if ! getent group $GROUP_ID > /dev/null 2>&1; then
    groupadd -g $GROUP_ID $USERNAME
fi

# Get the group name (in case it already existed with a different name)
GROUP_NAME=$(getent group $GROUP_ID | cut -d: -f1)

# Create user if it doesn't exist
if ! id -u $USERNAME > /dev/null 2>&1; then
    useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USERNAME

    # Add user to sudo group and allow passwordless sudo
    usermod -aG sudo $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

    # Add user to video group for GPU access
    usermod -aG video $USERNAME
fi

# Set up user's home directory
USER_HOME="/home/$USERNAME"

# Create bashrc if it doesn't exist
if [ ! -f "$USER_HOME/.bashrc" ]; then
    cp /etc/skel/.bashrc "$USER_HOME/.bashrc"
fi

# Add ROS2 setup to bashrc if not already there
if ! grep -q "source /opt/ros/jazzy/setup.bash" "$USER_HOME/.bashrc"; then
    echo "" >> "$USER_HOME/.bashrc"
    echo "# ROS2 Jazzy setup" >> "$USER_HOME/.bashrc"
    echo "source /opt/ros/jazzy/setup.bash" >> "$USER_HOME/.bashrc"
    echo "" >> "$USER_HOME/.bashrc"
    echo "# Source ROS2 workspace if it exists" >> "$USER_HOME/.bashrc"
    echo "if [ -f /workspace/ros2_ws/install/setup.bash ]; then" >> "$USER_HOME/.bashrc"
    echo "    source /workspace/ros2_ws/install/setup.bash" >> "$USER_HOME/.bashrc"
    echo "fi" >> "$USER_HOME/.bashrc"
fi

# Add virtual environment activation to bashrc if not already enabled
if ! grep -q "if \[ -f /workspace/unc/bin/activate \]; then" "$USER_HOME/.bashrc"; then
    echo "" >> "$USER_HOME/.bashrc"
    echo "# Activate Python virtual environment" >> "$USER_HOME/.bashrc"
    echo "if [ -f /workspace/unc/bin/activate ]; then" >> "$USER_HOME/.bashrc"
    echo "    source /workspace/unc/bin/activate" >> "$USER_HOME/.bashrc"
    echo "fi" >> "$USER_HOME/.bashrc"
elif grep -q "# if \[ -f /workspace/unc/bin/activate \]; then" "$USER_HOME/.bashrc"; then
    # If it exists but is commented out, uncomment it
    sed -i 's/# if \[ -f \/workspace\/unc\/bin\/activate \]; then/if [ -f \/workspace\/unc\/bin\/activate ]; then/' "$USER_HOME/.bashrc"
    sed -i 's/#     source \/workspace\/unc\/bin\/activate/    source \/workspace\/unc\/bin\/activate/' "$USER_HOME/.bashrc"
    sed -i 's/# fi$/fi/' "$USER_HOME/.bashrc"
fi

# Fix ownership of home directory
chown -R $USER_ID:$GROUP_ID "$USER_HOME"

# Export environment variables for the user
export HOME="$USER_HOME"

# Execute command as the specified user
exec gosu $USERNAME "$@"
