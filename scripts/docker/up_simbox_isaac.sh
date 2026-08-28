#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_GPU_ID="0"
DEFAULT_COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yml"

COMPOSE_FILE="${INTERNDATA_COMPOSE_FILE:-${DEFAULT_COMPOSE_FILE}}"
STACK_ID="${INTERNDATA_STACK_ID:-}"
GPU_ID="${INTERNDATA_ISAAC_GPU_DEVICE_IDS:-${DEFAULT_GPU_ID}}"
ISAAC_CPUS="${INTERNDATA_ISAAC_CPUS:-}"
LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${DEFAULT_LAUNCHER_CONFIG}}"
LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
LAUNCHER_EXTRA_ARGS_EXPLICIT=0
TASK_CONFIG="${TASK_CONFIG:-}"
LAUNCH_TEMPLATE="${LAUNCH_TEMPLATE:-${LAUNCHER_CONFIG}}"
RUN_NAME="${RUN_NAME:-}"
RANDOM_NUM="${RANDOM_NUM:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SEQ_OUTPUT_DIR="${SEQ_OUTPUT_DIR:-}"
EPISODE_EVENT_PATH="${INTERNDATA_EPISODE_EVENT_PATH:-}"
DEBUG_OUTPUT_DIR="${SIMBOX_DEBUG_OUTPUT_DIR:-}"
TASK_MODE=0
CPU_LIMITS_COMPOSE_FILE=""
services=()

usage() {
  cat <<'USAGE'
Usage: scripts/docker/up_simbox_isaac.sh [options] [isaac]

Start Isaac for a SimBox task. With TASK_CONFIG set, this command also
prepares, monitors, validates, and cleans up the task stack.

Options:
  --task-config PATH             Set the SimBox task config.
  --stack-id ID                  Set the stack and container names.
  --gpu ID                       Set the Isaac GPU device ID.
  --compose-file PATH            Use another Docker Compose file.
  --launcher-config PATH         Set the launcher config path.
  --launcher-extra-args ARGS     Set INTERNDATA_LAUNCHER_EXTRA_ARGS.
  --run-name NAME                Set the run name in task mode.
  --random-num COUNT             Set the layout randomization count.
  --random-seed SEED             Set the random seed.
  --output-dir PATH              Set the task output directory.
  --seq-output-dir PATH          Set the sequence output directory.
  --episode-event-path PATH      Set the episode event path.
  --debug-output-dir PATH        Set the debug output directory.
  --isaac-cpus COUNT             Limit the Isaac container to COUNT CPUs.
  -h, --help                     Show this help.

Task-mode values may also be supplied through the corresponding environment
variables. Without TASK_CONFIG, the command starts Isaac in the background.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --task-config)
      TASK_CONFIG="${2:?--task-config requires a value}"
      shift 2
      ;;
    --stack-id)
      STACK_ID="${2:?--stack-id requires a value}"
      shift 2
      ;;
    --gpu)
      GPU_ID="${2:?--gpu requires a value}"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="${2:?--compose-file requires a value}"
      shift 2
      ;;
    --launcher-config)
      LAUNCHER_CONFIG="${2:?--launcher-config requires a value}"
      LAUNCH_TEMPLATE="${LAUNCHER_CONFIG}"
      shift 2
      ;;
    --launcher-extra-args)
      LAUNCHER_EXTRA_ARGS="${2:?--launcher-extra-args requires a value}"
      LAUNCHER_EXTRA_ARGS_EXPLICIT=1
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?--run-name requires a value}"
      shift 2
      ;;
    --random-num)
      RANDOM_NUM="${2:?--random-num requires a value}"
      shift 2
      ;;
    --random-seed)
      RANDOM_SEED="${2:?--random-seed requires a value}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --seq-output-dir)
      SEQ_OUTPUT_DIR="${2:?--seq-output-dir requires a value}"
      shift 2
      ;;
    --episode-event-path)
      EPISODE_EVENT_PATH="${2:?--episode-event-path requires a value}"
      shift 2
      ;;
    --debug-output-dir)
      DEBUG_OUTPUT_DIR="${2:?--debug-output-dir requires a value}"
      shift 2
      ;;
    --isaac-cpus)
      ISAAC_CPUS="${2:?--isaac-cpus requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      services+=("$@")
      break
      ;;
    isaac)
      services+=("$1")
      shift
      ;;
    *)
      printf 'Unknown option or service: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "${#services[@]}" -eq 0 ]; then
  services=(isaac)
fi
for service in "${services[@]}"; do
  if [ "${service}" != "isaac" ]; then
    printf 'Unsupported service: %s\n' "${service}" >&2
    exit 2
  fi
done

if [[ "${COMPOSE_FILE}" != /* ]]; then
  COMPOSE_FILE="${REPO_ROOT}/${COMPOSE_FILE}"
fi

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || {
  printf 'Invalid GPU ID: %s\n' "${GPU_ID}" >&2
  exit 2
}
if [ -n "${ISAAC_CPUS}" ] && \
   { ! [[ "${ISAAC_CPUS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN { exit !(${ISAAC_CPUS} > 0) }"; }; then
  printf 'INTERNDATA_ISAAC_CPUS must be a positive number, got %s\n' "${ISAAC_CPUS}" >&2
  exit 2
fi

CONTAINER_UID="${INTERNDATA_CONTAINER_UID:-$(id -u)}"
CONTAINER_GID="${INTERNDATA_CONTAINER_GID:-$(id -g)}"
[[ "${CONTAINER_UID}" =~ ^[0-9]+$ ]] || {
  printf 'INTERNDATA_CONTAINER_UID must be a non-negative integer, got %s\n' "${CONTAINER_UID}" >&2
  exit 2
}
[[ "${CONTAINER_GID}" =~ ^[0-9]+$ ]] || {
  printf 'INTERNDATA_CONTAINER_GID must be a non-negative integer, got %s\n' "${CONTAINER_GID}"
  exit 2
}

sanitize_id() {
  local value="$1"
  value="$(printf '%s' "${value}" | LC_ALL=C tr '[:upper:]' '[:lower:]' | LC_ALL=C tr -cs '[:alnum:]_-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-default}"
}

json_field() {
  python3 -c 'import json,sys; value=json.load(sys.stdin); result=value[sys.argv[1]]; print(json.dumps(result, ensure_ascii=False) if isinstance(result, (list, dict, bool)) else result)' "$1"
}

if [ -n "${TASK_CONFIG}" ]; then
  TASK_MODE=1
  RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${TASK_CONFIG}")")/$(basename "${TASK_CONFIG}" .yaml)}"
  DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-output/docker_runtime/$(basename "${TASK_CONFIG}" .yaml)-$$/simbox_debug}"

  [[ "${RANDOM_NUM}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Invalid RANDOM_NUM: %s\n' "${RANDOM_NUM}" >&2
    exit 2
  }
  [[ "${RANDOM_SEED}" =~ ^-?[0-9]+$ ]] || {
    printf 'Invalid RANDOM_SEED: %s\n' "${RANDOM_SEED}" >&2
    exit 2
  }

  metadata_path="${INTERNDATA_DOCKER_METADATA_PATH:-output/docker_runtime/$(basename "${TASK_CONFIG}" .yaml)-$$/docker_runtime.json}"
  if [[ "${metadata_path}" != /* ]]; then
    metadata_path="${REPO_ROOT}/${metadata_path}"
  fi
  mkdir -p "$(dirname "${metadata_path}")"
  container_log_path="${metadata_path%.json}.isaac.log"

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

  task_container="$(json_field task_container <<<"${contract}")"
  launcher_container="$(json_field launcher_container <<<"${contract}")"
  launcher_args_json="$(json_field launcher_args <<<"${contract}")"
  event_container="$(json_field episode_event_container <<<"${contract}")"
  debug_container="$(json_field debug_output_container <<<"${contract}")"

  stack_id="$(sanitize_id "${STACK_ID:-${RUN_NAME}-$$}")"
  compose_project="simbox-isaac-${stack_id}"
  isaac_container="isaac-${stack_id}"
  default_cache_root="${REPO_ROOT}/.docker/isaac-sim/${stack_id}"

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
else
  if [ -n "${STACK_ID}" ]; then
    safe_stack_id="$(sanitize_id "${STACK_ID}")"
    export INTERNDATA_STACK_ID="${safe_stack_id}"
    default_cache_root="${REPO_ROOT}/.docker/isaac-sim/${safe_stack_id}"
    export INTERNDATA_COMPOSE_PROJECT="${INTERNDATA_COMPOSE_PROJECT:-simbox-isaac-${safe_stack_id}}"
    export INTERNDATA_ISAAC_CONTAINER_NAME="${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac-${safe_stack_id}}"
  else
    default_cache_root="${REPO_ROOT}/.docker/isaac-sim"
    export INTERNDATA_COMPOSE_PROJECT="${INTERNDATA_COMPOSE_PROJECT:-simbox-isaac}"
    export INTERNDATA_ISAAC_CONTAINER_NAME="${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac}"
  fi
fi

if [ "${TASK_MODE}" = "1" ]; then
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
else
  export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${GPU_ID}"
  export INTERNDATA_LAUNCHER_CONFIG="${LAUNCHER_CONFIG}"
  export INTERNDATA_LAUNCHER_EXTRA_ARGS="${LAUNCHER_EXTRA_ARGS}"
fi
if [ "${TASK_MODE}" = "1" ]; then
  export INTERNDATA_LAUNCHER_EXTRA_ARGS="${LAUNCHER_EXTRA_ARGS}"
fi
export INTERNDATA_CONTAINER_UID="${CONTAINER_UID}"
export INTERNDATA_CONTAINER_GID="${CONTAINER_GID}"

if [ "${TASK_MODE}" = "1" ] && [ "${DRY_RUN:-0}" = "1" ]; then
  write_metadata dry_run 0
  printf 'INTERNDATA_STACK_ID=%q INTERNDATA_ISAAC_GPU_DEVICE_IDS=%q ' "${stack_id}" "${GPU_ID}"
  printf '%q ' "${SCRIPT_DIR}/up_simbox_isaac.sh" --task-config "${TASK_CONFIG}" --stack-id "${stack_id}" --gpu "${GPU_ID}" --launcher-config "${LAUNCH_TEMPLATE}" "${services[@]}"
  printf '\nlauncher_args_json=%s\n' "${launcher_args_json}"
  exit 0
fi

runtime_unavailable() {
  if [ "${TASK_MODE}" = "1" ]; then
    write_metadata runtime_unavailable 3
  fi
  printf 'DOCKER_RUNTIME_UNAVAILABLE: %s\n' "$1" >&2
  exit 3
}

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  runtime_unavailable "Compose file not found: ${COMPOSE_FILE}"
fi

require_docker() {
  local docker_runtimes

  if ! command -v docker >/dev/null 2>&1; then
    runtime_unavailable "Docker CLI was not found."
  fi
  if ! docker compose version >/dev/null 2>&1; then
    runtime_unavailable "Docker Compose v2 is unavailable."
  fi
  if ! docker info >/dev/null 2>&1; then
    runtime_unavailable "Docker daemon is unavailable."
  fi
  if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    runtime_unavailable "NVIDIA Container Toolkit is unavailable: nvidia-container-runtime was not found."
  fi
  docker_runtimes="$(docker info --format '{{json .Runtimes}}')"
  if [[ "${docker_runtimes}" != *'"nvidia":'* ]]; then
    runtime_unavailable "Docker has not registered the NVIDIA container runtime. Configure it with nvidia-ctk runtime configure --runtime=docker."
  fi
}

require_docker

export ISAAC_CACHE_MAIN="${ISAAC_CACHE_MAIN:-${default_cache_root}/cache/main}"
export ISAAC_CACHE_COMPUTE="${ISAAC_CACHE_COMPUTE:-${default_cache_root}/cache/computecache}"
export ISAAC_LOGS="${ISAAC_LOGS:-${default_cache_root}/logs}"
export ISAAC_CONFIG="${ISAAC_CONFIG:-${default_cache_root}/config}"
export ISAAC_DATA="${ISAAC_DATA:-${default_cache_root}/data}"
export ISAAC_PKGS="${ISAAC_PKGS:-${default_cache_root}/pkg}"
mkdir -p \
  -- \
  "${ISAAC_CACHE_MAIN}" \
  "${ISAAC_CACHE_COMPUTE}" \
  "${ISAAC_LOGS}" \
  "${ISAAC_CONFIG}" \
  "${ISAAC_DATA}" \
  "${ISAAC_PKGS}"

# An explicit shell-style launcher argument override must clear a stale JSON
# value because entrypoint.sh gives the JSON channel precedence.
if [ "${LAUNCHER_EXTRA_ARGS_EXPLICIT}" = "1" ]; then
  export INTERNDATA_LAUNCHER_ARGS_JSON='[]'
fi

if [ -n "${ISAAC_CPUS}" ]; then
  CPU_LIMITS_COMPOSE_FILE="$(mktemp "${TMPDIR:-/tmp}/interndata-compose-cpu.XXXXXX.yml")"
  printf 'services:\n  isaac:\n    cpus: "%s"\n' "${ISAAC_CPUS}" >"${CPU_LIMITS_COMPOSE_FILE}"
fi

compose_args=(-f "${COMPOSE_FILE}" -p "${INTERNDATA_COMPOSE_PROJECT}")
if [ -n "${CPU_LIMITS_COMPOSE_FILE}" ]; then
  compose_args+=(-f "${CPU_LIMITS_COMPOSE_FILE}")
fi

if [ "${TASK_MODE}" = "1" ]; then
  set +e
  required_images="$(docker compose "${compose_args[@]}" config --images "${services[@]}")"
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
    if [[ "${started}" = "1" ]]; then
      docker stop "${isaac_container}" >/dev/null 2>&1 || true
      docker compose "${compose_args[@]}" down --remove-orphans >/dev/null 2>&1 || true
    fi
    if [[ -n "${CPU_LIMITS_COMPOSE_FILE}" ]]; then
      rm -f "${CPU_LIMITS_COMPOSE_FILE}"
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
  if ! docker compose "${compose_args[@]}" up -d --no-recreate "${services[@]}"; then
    printf 'DOCKER_START_FAILED: stack %s did not start.\n' "${stack_id}" >&2
    write_metadata start_failed 4
    exit 4
  fi
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

  application_status=finished
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
      application_status=application_failed
      exit_code=20
    fi
  fi
  write_metadata "${application_status}" "${exit_code}"
  exit "${exit_code}"
fi

trap 'rm -f "${CPU_LIMITS_COMPOSE_FILE}"' EXIT
printf 'Starting Isaac container %s on GPU %s\n' "${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac}" "${GPU_ID}"
docker compose "${compose_args[@]}" up -d --no-recreate "${services[@]}"
