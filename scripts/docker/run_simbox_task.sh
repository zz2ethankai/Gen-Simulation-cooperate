#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_ID="${GPU_ID:-0}"
RANDOM_NUM="${RANDOM_NUM:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"
TASK_CONFIG="${TASK_CONFIG:?TASK_CONFIG is required}"
LAUNCH_TEMPLATE="${LAUNCH_TEMPLATE:-configs/de_plan_with_render_template.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${TASK_CONFIG}")")/$(basename "${TASK_CONFIG}" .yaml)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SEQ_OUTPUT_DIR="${SEQ_OUTPUT_DIR:-}"
EPISODE_EVENT_PATH="${INTERNDATA_EPISODE_EVENT_PATH:-}"
metadata_path="${INTERNDATA_DOCKER_METADATA_PATH:-output/docker_runtime/$(basename "${TASK_CONFIG}" .yaml)-$$/docker_runtime.json}"
if [[ "${metadata_path}" != /* ]]; then
    metadata_path="${REPO_ROOT}/${metadata_path}"
fi
mkdir -p "$(dirname "${metadata_path}")"
container_log_path="${metadata_path%.json}.isaac.log"
DEBUG_OUTPUT_DIR="${SIMBOX_DEBUG_OUTPUT_DIR:-$(dirname "${metadata_path}")/simbox_debug}"
COMPOSE_FILE="${INTERNDATA_COMPOSE_FILE:-${REPO_ROOT}/docker/docker-compose.yml}"
if [[ "${COMPOSE_FILE}" != /* ]]; then
    COMPOSE_FILE="${REPO_ROOT}/${COMPOSE_FILE}"
fi

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || { printf 'Invalid GPU_ID: %s\n' "${GPU_ID}" >&2; exit 2; }
[[ "${RANDOM_NUM}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid RANDOM_NUM: %s\n' "${RANDOM_NUM}" >&2; exit 2; }
[[ "${RANDOM_SEED}" =~ ^-?[0-9]+$ ]] || { printf 'Invalid RANDOM_SEED: %s\n' "${RANDOM_SEED}" >&2; exit 2; }

contract="$(${REPO_ROOT}/scripts/docker/prepare_simbox_run.py \
    --task-config "${TASK_CONFIG}" \
    --launcher-config "${LAUNCH_TEMPLATE}" \
    --run-name "${RUN_NAME}" \
    --random-num "${RANDOM_NUM}" \
    --random-seed "${RANDOM_SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --seq-output-dir "${SEQ_OUTPUT_DIR}" \
    --episode-event-path "${EPISODE_EVENT_PATH}" \
    --debug-output-dir "${DEBUG_OUTPUT_DIR}")"

json_field() {
    python3 -c 'import json,sys; value=json.load(sys.stdin); result=value[sys.argv[1]]; print(json.dumps(result, ensure_ascii=False) if isinstance(result, (list, dict, bool)) else result)' "$1" <<<"${contract}"
}

task_container="$(json_field task_container)"
launcher_container="$(json_field launcher_container)"
launcher_args_json="$(json_field launcher_args)"
event_container="$(json_field episode_event_container)"
debug_container="$(json_field debug_output_container)"

sanitize_id() {
    local value="$1"
    value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '-')"
    value="${value#-}"
    value="${value%-}"
    printf '%s' "${value:-run}"
}

stack_id="$(sanitize_id "${INTERNDATA_STACK_ID:-${RUN_NAME}-$$}")"
compose_project="simbox-isaac-${stack_id}"
isaac_container="isaac-${stack_id}"

write_metadata() {
    local status="$1"
    local exit_code="${2:-}"
    python3 - \
        "${metadata_path}" \
        "${status}" \
        "${exit_code}" \
        "${stack_id}" \
        "${compose_project}" \
        "${isaac_container}" \
        "${GPU_ID}" \
        "${task_container}" \
        "${container_log_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]) if sys.argv[3] else None,
    "stack_id": sys.argv[4],
    "compose_project": sys.argv[5],
    "isaac_container": sys.argv[6],
    "host_gpu_id": int(sys.argv[7]),
    "task_container": sys.argv[8],
    "isaac_log_path": sys.argv[9],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

export INTERNDATA_STACK_ID="${stack_id}"
export INTERNDATA_COMPOSE_PROJECT="${compose_project}"
export INTERNDATA_ISAAC_CONTAINER_NAME="${isaac_container}"
export INTERNDATA_COMPOSE_FILE="${COMPOSE_FILE}"
export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${GPU_ID}"
export INTERNDATA_LAUNCHER_CONFIG="${launcher_container}"
export INTERNDATA_LAUNCHER_ARGS_JSON="${launcher_args_json}"
export INTERNDATA_TASK_CONFIG="${task_container}"
export INTERNDATA_EPISODE_EVENT_PATH="${event_container}"
export INTERNDATA_RANDOM_SEED="${RANDOM_SEED}"
export INTERNDATA_GPU="${GPU_ID}"
export INTERNDATA_TASK_PATH="${task_container}"
export SIMBOX_DEBUG_OUTPUT_DIR="${debug_container}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    write_metadata dry_run 0
    printf 'INTERNDATA_STACK_ID=%q INTERNDATA_ISAAC_GPU_DEVICE_IDS=%q ' "${stack_id}" "${GPU_ID}"
    printf '%q ' "${SCRIPT_DIR}/up_simbox_isaac.sh" --stack-id "${stack_id}" --gpu "${GPU_ID}" --launcher-config "${launcher_container}"
    printf '\nlauncher_args_json=%s\n' "${launcher_args_json}"
    exit 0
fi

runtime_unavailable() {
    write_metadata runtime_unavailable 3
    printf 'DOCKER_RUNTIME_UNAVAILABLE: %s\n' "$1" >&2
    exit 3
}

[[ -f "${COMPOSE_FILE}" ]] || runtime_unavailable "Compose file not found: ${COMPOSE_FILE}"
command -v docker >/dev/null 2>&1 || runtime_unavailable "Docker CLI was not found."
docker compose version >/dev/null 2>&1 || runtime_unavailable "Docker Compose plugin is unavailable."
docker info >/dev/null 2>&1 || runtime_unavailable "Docker daemon is unavailable."

set +e
required_images="$(docker compose -f "${COMPOSE_FILE}" config --images isaac)"
compose_config_status=$?
set -e
if [[ "${compose_config_status}" -ne 0 || -z "${required_images}" ]]; then
    write_metadata start_failed 4
    printf 'DOCKER_START_FAILED: could not resolve the Isaac image.\n' >&2
    exit 4
fi
while IFS= read -r image; do
    [[ -n "${image}" ]] || continue
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        write_metadata image_missing 5
        printf 'DOCKER_IMAGE_MISSING: required image is not available locally: %s.\n' "${image}" >&2
        exit 5
    fi
done <<<"${required_images}"

started=0
log_pid=""
cleanup() {
    local cleanup_status=$?
    set +e
    if [[ -n "${log_pid}" ]]; then
        kill "${log_pid}" >/dev/null 2>&1 || true
        wait "${log_pid}" >/dev/null 2>&1 || true
    fi
    if [[ "${started}" == "1" ]]; then
        docker stop "${isaac_container}" >/dev/null 2>&1 || true
        docker compose -f "${COMPOSE_FILE}" -p "${compose_project}" down --remove-orphans >/dev/null 2>&1 || true
    fi
    return "${cleanup_status}"
}
handle_signal() {
    local exit_code="$1"
    write_metadata interrupted "${exit_code}"
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

write_metadata starting
started=1
    "${SCRIPT_DIR}/up_simbox_isaac.sh" \
        --stack-id "${stack_id}" \
        --gpu "${GPU_ID}" \
        --launcher-config "${launcher_container}" || {
    printf 'DOCKER_START_FAILED: stack %s did not start.\n' "${stack_id}" >&2
    write_metadata start_failed 4
    exit 4
}
write_metadata running

docker logs -f "${isaac_container}" 2>&1 | tee "${container_log_path}" &
log_pid=$!
set +e
wait_output="$(docker wait "${isaac_container}")"
wait_status=$?
set -e
wait "${log_pid}" >/dev/null 2>&1 || true
log_pid=""

if [[ "${wait_status}" -ne 0 || ! "${wait_output}" =~ ^[0-9]+$ ]]; then
    exit_code=125
    printf 'DOCKER_WAIT_FAILED: could not read Isaac exit status for %s.\n' "${isaac_container}" >&2
else
    exit_code="${wait_output}"
fi
if [[ "${exit_code}" -ne 0 ]]; then
    printf 'ISAAC_CONTAINER_FAILED: %s exited with status %s.\n' "${isaac_container}" "${exit_code}" >&2
fi
application_status="finished"
if [[ "${exit_code}" -eq 0 ]]; then
    validation_error=""
    if [[ ! -s "${container_log_path}" ]]; then
        validation_error="Isaac container produced no captured logs"
    elif ! grep -Fq "Task is successful, mode=plan_with_render" "${container_log_path}"; then
        validation_error="missing Task is successful, mode=plan_with_render marker"
    elif grep -Fq "[LmdbLogger] Episode failed" "${container_log_path}"; then
        validation_error="found [LmdbLogger] Episode failed in Isaac logs"
    elif grep -Fq "Traceback (most recent call last):" "${container_log_path}"; then
        validation_error="found a traceback in Isaac logs"
    fi
    if [[ -n "${validation_error}" ]]; then
        printf 'ISAAC_APPLICATION_FAILED: %s. Log: %s\n' \
            "${validation_error}" "${container_log_path}" >&2
        application_status="application_failed"
        exit_code=20
    fi
fi
write_metadata "${application_status}" "${exit_code}"
exit "${exit_code}"
