#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_MODE="${RUN_MODE:-observe}"
WAIT_STEPS="${WAIT_STEPS:-300}"
TASK_CONFIG="${TASK_CONFIG:-InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_phone_placement/simbox_task.yaml}"
[[ "${RUN_MODE}" == "observe" || "${RUN_MODE}" == "skill" ]] || {
    printf 'RUN_MODE must be observe or skill, got %s\n' "${RUN_MODE}" >&2
    exit 2
}
[[ "${WAIT_STEPS}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid WAIT_STEPS: %s\n' "${WAIT_STEPS}" >&2; exit 2; }

if [[ "${TASK_CONFIG}" = /* ]]; then TASK_PATH="${TASK_CONFIG}"; else TASK_PATH="${REPO_ROOT}/${TASK_CONFIG}"; fi
[[ -f "${TASK_PATH}" ]] || { printf 'Task config not found: %s\n' "${TASK_PATH}" >&2; exit 2; }
TASK_NAME="$(basename "$(dirname "${TASK_PATH}")")"
RUN_NAME="${RUN_NAME:-bench21/${TASK_NAME}/${RUN_MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/${RUN_NAME}}"

if [[ "${RUN_MODE}" == "observe" ]]; then
    OBSERVE_CONFIG="${OBSERVE_CONFIG:-${REPO_ROOT}/output/.observe_configs/${TASK_NAME}/simbox_task.yaml}"
    python "${REPO_ROOT}/scripts/simbox/prepare_bench_observe_config.py" \
        --input "${TASK_PATH}" \
        --output "${OBSERVE_CONFIG}" \
        --wait-steps "${WAIT_STEPS}"
    LAUNCH_TASK="${OBSERVE_CONFIG}"
else
    LAUNCH_TASK="${TASK_PATH}"
fi

exec env \
    TASK_CONFIG="${LAUNCH_TASK}" \
    LAUNCH_TEMPLATE="configs/simbox/de_plan_with_render_template.yaml" \
    RUN_NAME="${RUN_NAME}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${REPO_ROOT}/scripts/docker/run_simbox_task.sh"
