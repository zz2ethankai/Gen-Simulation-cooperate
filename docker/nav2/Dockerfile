FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-lc"]

RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list; \
    fi && \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -print0 | xargs -0 -r sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|http://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g' \
      -e 's|https://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g'

RUN mkdir -p /root/.pip && \
    cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

RUN apt-get update && apt-get install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3-description \
    ros-humble-tf-transformations \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update || true

# Runtime startup selection is env-driven. Compose should override these values
# per stack/session instead of relying on project-specific defaults in the
# entrypoint script.
ENV INTERNDATA_AUTOSTART_ROS_STACK=1 \
    INTERNDATA_NAV2_ROBOT_CONFIG=workflows/simbox/core/configs/robots/split_aloha.yaml \
    INTERNDATA_NAV2_ROBOT_NAME=split_aloha \
    INTERNDATA_NAV2_SESSION_UUID=nav2_default \
    INTERNDATA_NAV2_STACK_CONFIG_DIR= \
    INTERNDATA_NAV2_BOOTSTRAP_MAP_DIR= \
    INTERNDATA_NAV2_PARAMS_FILE=

COPY docker/nav2/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
