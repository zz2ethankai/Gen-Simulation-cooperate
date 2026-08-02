FROM ros:humble

ARG NAV2_VERSION=1.1.20

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

# Build only RotationShim against the matching Humble Nav2 release. The rest
# of Nav2 continues to come from the apt-installed underlay.
RUN mkdir -p /opt/navigation2_ws/src && \
    git clone --depth 1 --branch "${NAV2_VERSION}" \
      https://github.com/ros-navigation/navigation2.git \
      /opt/navigation2_ws/src/navigation2

COPY docker/nav2/patches/0001-rotation-shim-stop-before-overshoot.patch /tmp/

RUN git -C /opt/navigation2_ws/src/navigation2 apply \
      /tmp/0001-rotation-shim-stop-before-overshoot.patch && \
    source /opt/ros/humble/setup.bash && \
    cd /opt/navigation2_ws && \
    colcon build \
      --merge-install \
      --packages-select nav2_rotation_shim_controller \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    rm -rf /opt/navigation2_ws/build /opt/navigation2_ws/log \
      /tmp/0001-rotation-shim-stop-before-overshoot.patch

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
