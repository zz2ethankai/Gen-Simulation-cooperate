#!/usr/bin/env bash
set -euo pipefail

TASK_PATH="${1:-${TASK_RENDER_TASK_PATH:-download/02_bookroom/task.yaml}}"
if [ "$#" -gt 0 ]; then
  shift
fi
IMAGE="${ISAAC_RENDER_IMAGE:-local/isaac-sim-4.1.0-curobo-app:latest}"
NAME="${ISAAC_RENDER_CONTAINER:-isaac-task-renderer}"
GPU_DEVICES="${ISAAC_RENDER_GPU_DEVICES:-0}"
OUTPUT_DIR="${TASK_RENDER_OUTPUT_DIR:-outputs/task_views}"
TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${NAME}" \
  --network none \
  --gpus "device=${GPU_DEVICES}" \
  --shm-size 8gb \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e ISAAC_SIM_PATH=/isaac-sim \
  -e TASK_RENDER_WIDTH="${TASK_RENDER_WIDTH:-2560}" \
  -e TASK_RENDER_HEIGHT="${TASK_RENDER_HEIGHT:-1440}" \
  -e TASK_RENDER_RT_SUBFRAMES="${TASK_RENDER_RT_SUBFRAMES:-32}" \
  -e TASK_RENDER_SETTLE_SECONDS="${TASK_RENDER_SETTLE_SECONDS:-1.0}" \
  -e TASK_RENDER_NO_PHYSICS="${TASK_RENDER_NO_PHYSICS:-}" \
  -e TASK_RENDER_NO_SNAP_TO_SUPPORTS="${TASK_RENDER_NO_SNAP_TO_SUPPORTS:-}" \
  -e TASK_RENDER_RENDERER="${TASK_RENDER_RENDERER:-RayTracedLighting}" \
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
  webrtc/render_views.py \
  --task "${TASK_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
