#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yml"
DEFAULT_STACK_ID="dev"
DEFAULT_GPU_ID="0"
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"

ACTION="shell"
ACTION_EXPLICIT=0
GPU_ID="${INTERNDATA_DEV_GPU:-${DEFAULT_GPU_ID}}"
STACK_ID="${INTERNDATA_DEV_STACK_ID:-${DEFAULT_STACK_ID}}"
ISAAC_CPUS="${INTERNDATA_DEV_ISAAC_CPUS:-}"
COMPOSE_FILE="${INTERNDATA_DEV_COMPOSE_FILE:-${DEFAULT_COMPOSE_FILE}}"
BUILD_IMAGE=0
FOLLOW_LOGS=0
EXEC_COMMAND=()

usage() {
  cat <<'USAGE'
Usage: scripts/docker/isaac_dev.sh [action] [options]

Start or enter an Isaac Sim Bash development container. The existing Isaac
image and repository mounts are reused; launcher.py is never auto-started.

Actions:
  shell                          Start the dev container and open Bash (default).
  start                          Start the dev container in the background.
  exec -- COMMAND [ARGS...]      Run a command inside /workspace.
  status                         Show the dev container status.
  logs                           Show the dev container logs.
  stop                           Stop and remove the dev Compose project.

Options:
  --gpu ID                       Select one NVIDIA GPU (default: 0).
  --stack-id ID                  Isolate container/cache names (default: dev).
  --isaac-cpus COUNT             Set a Docker CPU quota.
  --compose-file PATH            Use another Compose file.
  --build                        Build the Isaac image before starting.
  --follow                       Follow logs for the logs action.
  -h, --help                     Show this help.

Examples:
  scripts/docker/isaac_dev.sh shell --gpu 0 --build
  scripts/docker/isaac_dev.sh start --gpu 1
  scripts/docker/isaac_dev.sh exec -- python -c 'import torch; print(torch.__version__)'
  scripts/docker/isaac_dev.sh stop
USAGE
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    shell|start|exec|status|logs|stop)
      ACTION="$1"
      ACTION_EXPLICIT=1
      shift
      ;;
  esac
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    shell|start|exec|status|logs|stop)
      if [ "${ACTION_EXPLICIT}" = "1" ]; then
        if [ "${ACTION}" = "exec" ]; then
          EXEC_COMMAND+=("$1")
          shift
        else
          printf 'Only one action may be specified.\n' >&2
          exit 2
        fi
      else
        ACTION="$1"
        ACTION_EXPLICIT=1
        shift
      fi
      ;;
    --gpu)
      GPU_ID="${2:?--gpu requires a value}"
      shift 2
      ;;
    --stack-id)
      STACK_ID="${2:?--stack-id requires a value}"
      shift 2
      ;;
    --isaac-cpus)
      ISAAC_CPUS="${2:?--isaac-cpus requires a value}"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="${2:?--compose-file requires a value}"
      shift 2
      ;;
    --build)
      BUILD_IMAGE=1
      shift
      ;;
    --follow)
      FOLLOW_LOGS=1
      shift
      ;;
    --)
      shift
      EXEC_COMMAND=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ "${ACTION}" = "exec" ]; then
        EXEC_COMMAND+=("$1")
        shift
      else
        printf 'Unknown option or argument: %s\n' "$1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || {
  printf 'Invalid GPU ID: %s\n' "${GPU_ID}" >&2
  exit 2
}
if [ -n "${ISAAC_CPUS}" ] && \
   { ! [[ "${ISAAC_CPUS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN { exit !(${ISAAC_CPUS} > 0) }"; }; then
  printf 'ISAAC CPU quota must be a positive number, got %s\n' "${ISAAC_CPUS}" >&2
  exit 2
fi

sanitize_id() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-default}"
}

safe_stack_id="$(sanitize_id "${STACK_ID}")"
compose_project="${INTERNDATA_DEV_COMPOSE_PROJECT:-simbox-isaac-dev-${safe_stack_id}}"
container_name="${INTERNDATA_DEV_CONTAINER_NAME:-isaac-dev-${safe_stack_id}}"
dev_cache_root="${INTERNDATA_DEV_CACHE_ROOT:-${REPO_ROOT}/output/isaac-dev/${safe_stack_id}}"

# Keep the development stack independent from task-generation stacks while
# preserving the same image, source checkout, assets, and Isaac runtime.
export INTERNDATA_COMPOSE_FILE="${COMPOSE_FILE}"
export INTERNDATA_COMPOSE_PROJECT="${compose_project}"
export INTERNDATA_ISAAC_CONTAINER_NAME="${container_name}"
export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${GPU_ID}"
export INTERNDATA_CONTAINER_UID="${INTERNDATA_CONTAINER_UID:-$(id -u)}"
export INTERNDATA_CONTAINER_GID="${INTERNDATA_CONTAINER_GID:-$(id -g)}"
export INTERNDATA_STACK_ID="${safe_stack_id}"

# A developer shell must remain a shell. In particular, do not inherit a
# host setting that would make the existing service launch launcher.py.
export INTERNDATA_AUTOSTART_LAUNCHER=0
export INTERNDATA_RUN_IMPORT_CHECKS=0
export LIVESTREAM=0
export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${DEFAULT_LAUNCHER_CONFIG}}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS=""
export INTERNDATA_LAUNCHER_ARGS_JSON="[]"

export ISAAC_CACHE_MAIN="${ISAAC_CACHE_MAIN:-${dev_cache_root}/cache/main}"
export ISAAC_CACHE_COMPUTE="${ISAAC_CACHE_COMPUTE:-${dev_cache_root}/cache/computecache}"
export ISAAC_LOGS="${ISAAC_LOGS:-${dev_cache_root}/logs}"
export ISAAC_CONFIG="${ISAAC_CONFIG:-${dev_cache_root}/config}"
export ISAAC_DATA="${ISAAC_DATA:-${dev_cache_root}/data}"
export ISAAC_PKGS="${ISAAC_PKGS:-${dev_cache_root}/pkg}"

compose=(docker compose -f "${COMPOSE_FILE}" -p "${compose_project}")

require_docker() {
  command -v docker >/dev/null 2>&1 || {
    printf 'Docker CLI was not found.\n' >&2
    exit 3
  }
  docker compose version >/dev/null 2>&1 || {
    printf 'Docker Compose v2 is required.\n' >&2
    exit 3
  }
  docker info >/dev/null 2>&1 || {
    printf 'Docker daemon is unavailable.\n' >&2
    exit 3
  }
}

prepare_host_dirs() {
  mkdir -p \
    "${ISAAC_CACHE_MAIN}" \
    "${ISAAC_CACHE_COMPUTE}" \
    "${ISAAC_LOGS}" \
    "${ISAAC_CONFIG}" \
    "${ISAAC_DATA}" \
    "${ISAAC_PKGS}"
}

start_container() {
  require_docker
  prepare_host_dirs

  if [ "${BUILD_IMAGE}" = "1" ]; then
    "${compose[@]}" build isaac
  fi

  up_args=(--stack-id "${safe_stack_id}" --gpu "${GPU_ID}")
  if [ -n "${ISAAC_CPUS}" ]; then
    up_args+=(--isaac-cpus "${ISAAC_CPUS}")
  fi
  "${SCRIPT_DIR}/up_simbox_isaac.sh" "${up_args[@]}"
}

wait_until_running() {
  local state=""
  local attempt
  for attempt in $(seq 1 60); do
    state="$(docker inspect --format '{{.State.Status}}' "${container_name}" 2>/dev/null || true)"
    case "${state}" in
      running)
        return 0
        ;;
      exited|dead)
        printf 'Isaac dev container entered state %s:\n' "${state}" >&2
        docker logs --tail 80 "${container_name}" >&2 || true
        return 1
        ;;
    esac
    sleep 1
  done
  printf 'Timed out waiting for %s to become running.\n' "${container_name}" >&2
  docker logs --tail 80 "${container_name}" >&2 || true
  return 1
}

run_shell() {
  local tty_args=()
  if [ -t 0 ] && [ -t 1 ]; then
    tty_args=(-i -t)
  else
    tty_args=(-i)
  fi
  # Run the entrypoint once more for the attached shell so its Isaac/CUDA
  # runtime exports are present even when this is a docker exec process.
  exec docker exec "${tty_args[@]}" -w /workspace "${container_name}" \
    /entrypoint.sh bash --login
}

run_command() {
  if [ "${#EXEC_COMMAND[@]}" -eq 0 ]; then
    printf 'exec requires a command, for example: exec -- python -V\n' >&2
    exit 2
  fi
  exec docker exec -w /workspace "${container_name}" \
    /entrypoint.sh "${EXEC_COMMAND[@]}"
}

require_docker
case "${ACTION}" in
  start)
    start_container
    wait_until_running
    printf 'Isaac dev container is running: %s\n' "${container_name}"
    ;;
  shell)
    start_container
    wait_until_running
    run_shell
    ;;
  exec)
    start_container
    wait_until_running
    run_command
    ;;
  status)
    state="$(docker inspect --format '{{.State.Status}}' "${container_name}" 2>/dev/null || true)"
    if [ -n "${state}" ]; then
      printf '%s\n' "${container_name}: ${state}"
    else
      printf '%s: not created\n' "${container_name}"
    fi
    ;;
  logs)
    log_args=(logs --tail="${INTERNDATA_DEV_LOG_TAIL:-100}")
    if [ "${FOLLOW_LOGS}" = "1" ]; then
      log_args+=(--follow)
    fi
    "${compose[@]}" "${log_args[@]}" isaac
    ;;
  stop)
    "${compose[@]}" down --remove-orphans
    ;;
  *)
    printf 'Unsupported action: %s\n' "${ACTION}" >&2
    exit 2
    ;;
esac
