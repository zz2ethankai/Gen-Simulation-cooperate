#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_GPU_ID="0"

COMPOSE_FILE="${INTERNDATA_COMPOSE_FILE:-${REPO_ROOT}/docker/docker-compose.yml}"
STACK_ID="${INTERNDATA_STACK_ID:-}"
GPU_ID="${INTERNDATA_ISAAC_GPU_DEVICE_IDS:-${DEFAULT_GPU_ID}}"
ISAAC_CPUS="${INTERNDATA_ISAAC_CPUS:-}"
CPU_LIMITS_COMPOSE_FILE=""

usage() {
  cat <<'USAGE'
Usage: scripts/docker/up_simbox_isaac.sh [options]

Start the Isaac container used by a SimBox task. Navigation runs inside Isaac;
there are no external ROS or navigation stack services.

Options:
  --stack-id ID                  Set the stack and container names.
  --gpu ID                       Set the Isaac GPU device ID.
  --launcher-config PATH         Set INTERNDATA_LAUNCHER_CONFIG.
  --launcher-extra-args ARGS     Set INTERNDATA_LAUNCHER_EXTRA_ARGS.
  --isaac-cpus COUNT             Limit the Isaac container to COUNT CPUs.
  -h, --help                     Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stack-id)
      STACK_ID="${2:?--stack-id requires a value}"
      shift 2
      ;;
    --gpu)
      GPU_ID="${2:?--gpu requires a value}"
      shift 2
      ;;
    --launcher-config)
      export INTERNDATA_LAUNCHER_CONFIG="${2:?--launcher-config requires a value}"
      shift 2
      ;;
    --launcher-extra-args)
      export INTERNDATA_LAUNCHER_EXTRA_ARGS="${2:?--launcher-extra-args requires a value}"
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
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]] || {
  printf 'Invalid GPU ID: %s\n' "${GPU_ID}" >&2
  exit 2
}
if [ -n "${ISAAC_CPUS}" ] && \
   { ! [[ "${ISAAC_CPUS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN { exit !(${ISAAC_CPUS} > 0) }"; }; then
  printf 'INTERNDATA_ISAAC_CPUS must be a positive number, got %s\n' "${ISAAC_CPUS}" >&2
  exit 2
fi

sanitize_id() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-default}"
}

export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${GPU_ID}"
export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${DEFAULT_LAUNCHER_CONFIG}}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"

if [ -n "${STACK_ID}" ]; then
  safe_stack_id="$(sanitize_id "${STACK_ID}")"
  export INTERNDATA_STACK_ID="${safe_stack_id}"
  export INTERNDATA_COMPOSE_PROJECT="${INTERNDATA_COMPOSE_PROJECT:-simbox-isaac-${safe_stack_id}}"
  export INTERNDATA_ISAAC_CONTAINER_NAME="${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac-${safe_stack_id}}"
  export ISAAC_CACHE_MAIN="${ISAAC_CACHE_MAIN:-../.docker/isaac-sim/${safe_stack_id}/cache/main}"
  export ISAAC_CACHE_COMPUTE="${ISAAC_CACHE_COMPUTE:-../.docker/isaac-sim/${safe_stack_id}/cache/computecache}"
  export ISAAC_LOGS="${ISAAC_LOGS:-../.docker/isaac-sim/${safe_stack_id}/logs}"
  export ISAAC_CONFIG="${ISAAC_CONFIG:-../.docker/isaac-sim/${safe_stack_id}/config}"
  export ISAAC_DATA="${ISAAC_DATA:-../.docker/isaac-sim/${safe_stack_id}/data}"
  export ISAAC_PKGS="${ISAAC_PKGS:-../.docker/isaac-sim/${safe_stack_id}/pkg}"
fi

if [ -n "${ISAAC_CPUS}" ]; then
  CPU_LIMITS_COMPOSE_FILE="$(mktemp "${TMPDIR:-/tmp}/interndata-compose-cpu.XXXXXX.yml")"
  printf 'services:\n  isaac:\n    cpus: "%s"\n' "${ISAAC_CPUS}" >"${CPU_LIMITS_COMPOSE_FILE}"
  trap 'rm -f "${CPU_LIMITS_COMPOSE_FILE}"' EXIT
fi

compose_args=(-f "${COMPOSE_FILE}")
if [ -n "${CPU_LIMITS_COMPOSE_FILE}" ]; then
  compose_args+=(-f "${CPU_LIMITS_COMPOSE_FILE}")
fi

printf 'Starting Isaac container %s on GPU %s\n' "${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac}" "${GPU_ID}"
docker compose "${compose_args[@]}" up -d isaac
