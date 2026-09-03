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
LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${LAUNCH_TEMPLATE:-${DEFAULT_LAUNCHER_CONFIG}}}"
LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
LAUNCHER_EXTRA_ARGS_EXPLICIT=0
CPU_LIMITS_COMPOSE_FILE=""

usage() {
  cat <<'USAGE'
Usage: scripts/docker/up_simbox_isaac.sh [options]

Start the Isaac Docker container.

Options:
  --stack-id ID                  Set the stack and container names.
  --gpu ID                       Set the Isaac GPU device ID.
  --compose-file PATH            Use another Docker Compose file.
  --launcher-config PATH         Set the launcher config path.
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
    --compose-file)
      COMPOSE_FILE="${2:?--compose-file requires a value}"
      shift 2
      ;;
    --launcher-config)
      LAUNCHER_CONFIG="${2:?--launcher-config requires a value}"
      shift 2
      ;;
    --launcher-extra-args)
      LAUNCHER_EXTRA_ARGS="${2:?--launcher-extra-args requires a value}"
      LAUNCHER_EXTRA_ARGS_EXPLICIT=1
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

if [[ "${COMPOSE_FILE}" != /* ]]; then
  COMPOSE_FILE="${REPO_ROOT}/${COMPOSE_FILE}"
fi

CONTAINER_UID="${INTERNDATA_CONTAINER_UID:-$(id -u)}"
CONTAINER_GID="${INTERNDATA_CONTAINER_GID:-$(id -g)}"

sanitize_id() {
  local value="$1"
  value="$(printf '%s' "${value}" | LC_ALL=C tr '[:upper:]' '[:lower:]' | LC_ALL=C tr -cs '[:alnum:]_-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-default}"
}

export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${GPU_ID}"
export INTERNDATA_LAUNCHER_CONFIG="${LAUNCHER_CONFIG}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS="${LAUNCHER_EXTRA_ARGS}"
export INTERNDATA_CONTAINER_UID="${CONTAINER_UID}"
export INTERNDATA_CONTAINER_GID="${CONTAINER_GID}"

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

# entrypoint.sh gives the JSON argument channel precedence over the shell
# argument channel. An explicit shell override must therefore clear it.
if [ "${LAUNCHER_EXTRA_ARGS_EXPLICIT}" = "1" ]; then
  export INTERNDATA_LAUNCHER_ARGS_JSON='[]'
fi

if [ -n "${ISAAC_CPUS}" ]; then
  CPU_LIMITS_COMPOSE_FILE="$(mktemp "${TMPDIR:-/tmp}/interndata-compose-cpu.XXXXXX.yml")"
  printf 'services:\n  isaac:\n    cpus: "%s"\n' "${ISAAC_CPUS}" >"${CPU_LIMITS_COMPOSE_FILE}"
  trap 'rm -f "${CPU_LIMITS_COMPOSE_FILE}"' EXIT
fi

compose_args=(-f "${COMPOSE_FILE}" -p "${INTERNDATA_COMPOSE_PROJECT}")
if [ -n "${CPU_LIMITS_COMPOSE_FILE}" ]; then
  compose_args+=(-f "${CPU_LIMITS_COMPOSE_FILE}")
fi

printf 'Starting Isaac container %s on GPU %s\n' \
  "${INTERNDATA_ISAAC_CONTAINER_NAME}" "${GPU_ID}"
docker compose "${compose_args[@]}" up -d isaac
