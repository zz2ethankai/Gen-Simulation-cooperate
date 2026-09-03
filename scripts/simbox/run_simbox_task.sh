#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-interndata-isaac6}"
GPU_ID="${GPU_ID:-0}"
RANDOM_NUM="${RANDOM_NUM:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"
TASK_CONFIG="${TASK_CONFIG:?TASK_CONFIG is required}"
LAUNCH_TEMPLATE="${LAUNCH_TEMPLATE:-configs/de_plan_with_render_template.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${TASK_CONFIG}")")/$(basename "${TASK_CONFIG}" .yaml)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SEQ_OUTPUT_DIR="${SEQ_OUTPUT_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_PREFLIGHT_ONLY="${CONDA_PREFLIGHT_ONLY:-0}"

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || {
  printf 'Invalid GPU_ID: %s\n' "${GPU_ID}" >&2
  exit 2
}
[[ "${RANDOM_NUM}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'Invalid RANDOM_NUM: %s\n' "${RANDOM_NUM}" >&2
  exit 2
}
[[ "${RANDOM_SEED}" =~ ^-?[0-9]+$ ]] || {
  printf 'Invalid RANDOM_SEED: %s\n' "${RANDOM_SEED}" >&2
  exit 2
}

resolve_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "$1"
  fi
}

TASK_PATH="$(resolve_path "${TASK_CONFIG}")"
TEMPLATE_PATH="$(resolve_path "${LAUNCH_TEMPLATE}")"
[[ -f "${TASK_PATH}" ]] || {
  printf 'Task config not found: %s\n' "${TASK_PATH}" >&2
  exit 2
}
[[ -f "${TEMPLATE_PATH}" ]] || {
  printf 'Launch template not found: %s\n' "${TEMPLATE_PATH}" >&2
  exit 2
}

COMMAND=(
  python launcher.py
  --config "${TEMPLATE_PATH}"
  --name="${RUN_NAME}"
  --random_seed="${RANDOM_SEED}"
  --load_stage.scene_loader.args.cfg_path="${TASK_PATH}"
  --load_stage.scene_loader.args.simulator.active_gpu="${GPU_ID}"
  --load_stage.scene_loader.args.simulator.physics_gpu="${GPU_ID}"
  --load_stage.scene_loader.args.simulator.cuda_device="${GPU_ID}"
  --load_stage.layout_random_generator.args.random_num="${RANDOM_NUM}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMAND+=(--store_stage.writer.args.output_dir="${OUTPUT_DIR}")
fi
if [[ -n "${SEQ_OUTPUT_DIR}" ]]; then
  COMMAND+=(--store_stage.writer.args.seq_output_dir="${SEQ_OUTPUT_DIR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_COMMAND=(
    env
    -u CUDA_VISIBLE_DEVICES
    CONDA_ENV="${CONDA_ENV}"
    conda run --no-capture-output -n "${CONDA_ENV}"
    "${COMMAND[@]}"
  )
  printf '%q ' "${DRY_COMMAND[@]}"
  printf '\n'
  exit 0
fi

export CONDA_ENV
set +u
source "${REPO_ROOT}/scripts/conda/activate_isaac6_env.sh"
set -u

command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'nvidia-smi was not found.\n' >&2
  exit 2
}
nvidia-smi --query-gpu=index --format=csv,noheader,nounits |
  tr -d ' ' |
  grep -qx "${GPU_ID}" || {
    printf 'Physical GPU %s does not exist.\n' "${GPU_ID}" >&2
    exit 2
  }

# Native Isaac Sim/Omniverse selects the renderer GPU by the physical
# nvidia-smi index.  CUDA_VISIBLE_DEVICES renumbers CUDA devices without
# equivalently renumbering Vulkan devices, which can leave RTX/Hydra without
# a matching CUDA device on non-zero GPUs.  Keep the full device inventory
# visible and pass GPU_ID as the absolute index to rendering, PhysX and CUDA.
unset CUDA_VISIBLE_DEVICES
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACSIM_ACCEPT_EULA="${ISAACSIM_ACCEPT_EULA:-yes}"
python "${REPO_ROOT}/scripts/conda/verify_isaac6_env.py" --repo-root "${REPO_ROOT}"

if [[ "${CONDA_PREFLIGHT_ONLY}" == "1" ]]; then
  exit 0
fi

cd "${REPO_ROOT}"
exec env PYTHONUNBUFFERED=1 "${COMMAND[@]}"
