#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/simbox/simbox_plan_with_render_multi_gpu.sh <task_yaml|task_list|task_dir> [random_num] [gpu_list] [random_seed_base] [scene_info]

Compatibility wrapper for scripts/simbox/simbox_parallel_generate.sh --backend local.

Parameters:
  task_yaml        Single task yaml to run
  task_list        Text file with one task yaml per line; blank lines and # comments are ignored
  task_dir         Directory; all *.yaml/*.yml files under it will be queued
  random_num       Samples per task yaml, default: 10
  gpu_list         Physical GPU ids, comma-separated, default: 0,1,2,3
  random_seed_base Optional. If set, task seed = random_seed_base + queue index
  scene_info       Optional scene info key, e.g. living_room_scene_info

Environment:
  ISAACSIM_PYTHON_EXE     Isaac Sim python launcher, default: /home/bld/ykqin/isaacsim/python.sh
  INTERNDATA_DE_CONFIG    Data engine config, default: configs/simbox/de_plan_with_render_template.yaml
  INTERNDATA_DATASET_ROOT Shared dataset root, default: output/simbox_plan_with_render_dataset

Examples:
  bash scripts/simbox/simbox_plan_with_render_multi_gpu.sh workflows/simbox/core/configs/tasks/example 1 0,1
  bash scripts/simbox/simbox_plan_with_render_multi_gpu.sh /tmp/task_list.txt 20 0,1,2,3 1000
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -lt 1 ]; then
    usage
    exit 2
fi

task_source="$1"
random_num="${2:-10}"
gpu_list="${3:-0,1,2,3}"
random_seed_base="${4:-}"
scene_info="${5:-}"

cmd=(
    bash "${SCRIPT_DIR}/simbox_parallel_generate.sh"
    --backend local
    --gpus "$gpu_list"
    --workers-per-gpu 1
    --random-num "$random_num"
    --de-config "${INTERNDATA_DE_CONFIG:-configs/simbox/de_plan_with_render_template.yaml}"
    --dataset-root "${INTERNDATA_DATASET_ROOT:-output/simbox_plan_with_render_dataset}"
    --isaac-python "${ISAACSIM_PYTHON_EXE:-/home/bld/ykqin/isaacsim/python.sh}"
)

if [ -n "$random_seed_base" ]; then
    cmd+=(--random-seed-base "$random_seed_base")
fi

if [ -n "$scene_info" ]; then
    cmd+=(--scene-info "$scene_info")
fi

cmd+=("$task_source")

exec "${cmd[@]}"
