#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-interndata}"
GPU_ID="${GPU_ID:-0}"
RANDOM_NUM="${RANDOM_NUM:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"
TASK_CONFIG="${TASK_CONFIG:?TASK_CONFIG is required}"
LAUNCH_TEMPLATE="${LAUNCH_TEMPLATE:-configs/simbox/de_plan_with_render_template.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${TASK_CONFIG}")")/$(basename "${TASK_CONFIG}" .yaml)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SEQ_OUTPUT_DIR="${SEQ_OUTPUT_DIR:-}"

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || { printf 'Invalid GPU_ID: %s\n' "${GPU_ID}" >&2; exit 2; }
[[ "${RANDOM_NUM}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid RANDOM_NUM: %s\n' "${RANDOM_NUM}" >&2; exit 2; }
[[ "${RANDOM_SEED}" =~ ^-?[0-9]+$ ]] || { printf 'Invalid RANDOM_SEED: %s\n' "${RANDOM_SEED}" >&2; exit 2; }

resolve_path() {
    if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "${REPO_ROOT}" "$1"; fi
}
TASK_PATH="$(resolve_path "${TASK_CONFIG}")"
TEMPLATE_PATH="$(resolve_path "${LAUNCH_TEMPLATE}")"
[[ -f "${TASK_PATH}" ]] || { printf 'Task config not found: %s\n' "${TASK_PATH}" >&2; exit 2; }
[[ -f "${TEMPLATE_PATH}" ]] || { printf 'Launch template not found: %s\n' "${TEMPLATE_PATH}" >&2; exit 2; }

CURRENT_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ "${CURRENT_CONDA_ENV}" != "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -x /opt/anaconda3/bin/conda ]]; then
        CONDA_BASE="$(/opt/anaconda3/bin/conda info --base)"
    else
        printf 'conda was not found.\n' >&2
        exit 2
    fi
    set +u
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
fi

command -v nvidia-smi >/dev/null 2>&1 || { printf 'nvidia-smi was not found.\n' >&2; exit 2; }
nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ' | grep -qx "${GPU_ID}" || {
    printf 'Physical GPU %s does not exist.\n' "${GPU_ID}" >&2
    exit 2
}

EXPECTED_CUROBO_ROOT="${REPO_ROOT}/InternDataAssets/curobo"
CUROBO_MODULE_PATH="$(python -c 'from pathlib import Path; import curobo; print(Path(curobo.__file__).resolve())')"
[[ "${CUROBO_MODULE_PATH}" == "${EXPECTED_CUROBO_ROOT}/"* ]] || {
    printf 'Unexpected CuRobo checkout: %s\n' "${CUROBO_MODULE_PATH}" >&2
    exit 2
}
python -c 'import isaacsim; import omni.isaac.kit' >/dev/null 2>&1 || {
    printf 'Isaac Sim modules are unavailable in %s.\n' "${CONDA_ENV}" >&2
    exit 2
}
[[ "$(python -c 'import scipy; print(scipy.__version__)')" == "1.14.1" ]] || {
    printf 'Expected scipy==1.14.1 in %s.\n' "${CONDA_ENV}" >&2
    exit 2
}

cd "${REPO_ROOT}"
COMMAND=(
    python launcher.py
    --config "${TEMPLATE_PATH}"
    --name="${RUN_NAME}"
    --random_seed="${RANDOM_SEED}"
    --load_stage.scene_loader.args.cfg_path="${TASK_PATH}"
    --load_stage.scene_loader.args.simulator.active_gpu="${GPU_ID}"
    --load_stage.scene_loader.args.simulator.physics_gpu=0
    --load_stage.scene_loader.args.simulator.cuda_device=0
    --load_stage.layout_random_generator.args.random_num="${RANDOM_NUM}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    COMMAND+=(--store_stage.writer.args.output_dir="${OUTPUT_DIR}")
fi
if [[ -n "${SEQ_OUTPUT_DIR}" ]]; then
    COMMAND+=(--store_stage.writer.args.seq_output_dir="${SEQ_OUTPUT_DIR}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${COMMAND[@]}"
