#!/usr/bin/env bash
set -euo pipefail

TASK_PATH="${1:-${WEBRTC_TASK_PATH:-download/01_kitchen/task.yaml}}"
if [ "$#" -gt 0 ]; then
  shift
fi
IMAGE="${ISAAC_WEBRTC_IMAGE:-local/isaac-sim-4.1.0-curobo-app:latest}"
NAME="${ISAAC_WEBRTC_CONTAINER:-isaac-webrtc-viewer}"
GPU_DEVICES="${ISAAC_WEBRTC_GPU_DEVICES:-0}"
TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${NAME}" \
  --network host \
  --gpus "device=${GPU_DEVICES}" \
  --shm-size 8gb \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e ISAAC_SIM_PATH=/isaac-sim \
  -e WEBRTC_HTTP_PORT="${WEBRTC_HTTP_PORT:-8211}" \
  -e WEBRTC_RTC_PORT="${WEBRTC_RTC_PORT:-49100}" \
  -v "$(pwd)":/workspace:rw \
  -v "$(pwd)/.docker/isaac-sim/cache/main":/root/.cache/ov:rw \
  -v "$(pwd)/.docker/isaac-sim/cache/computecache":/root/.nv/ComputeCache:rw \
  -v "$(pwd)/.docker/isaac-sim/logs":/root/.nvidia-omniverse/logs:rw \
  -v "$(pwd)/.docker/isaac-sim/config":/root/.nvidia-omniverse/config:rw \
  -v "$(pwd)/.docker/isaac-sim/data":/root/.local/share/ov/data:rw \
  -v "$(pwd)/.docker/isaac-sim/pkg":/root/.local/share/ov/pkg:rw \
  -w /workspace \
  --entrypoint /isaac-sim/python.sh \
  "${IMAGE}" \
  webrtc/viewer.py \
  --/app/livestream/enabled=true \
  --/app/window/drawMouse=true \
  --/exts/omni.services.transport.server.http/port="${WEBRTC_HTTP_PORT:-8211}" \
  --/app/livestream/port="${WEBRTC_RTC_PORT:-49100}" \
  --task "${TASK_PATH}" \
  "$@"
