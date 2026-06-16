#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default launch settings. Edit these values for the common local workflow; all
# of them can still be overridden from the environment when needed.
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_SINGLE_GPU_DEVICE_IDS="0"
DEFAULT_ROS_DOMAIN_ID="0"
DEFAULT_SERVICES=(isaac nav2)
DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="1"

COMPOSE_FILE="${INTERNDATA_COMPOSE_FILE:-${REPO_ROOT}/docker/docker-compose.yml}"
STACK_ID="${INTERNDATA_STACK_ID:-}"

sanitize_id() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -cs '[:alnum:]_.-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-default}"
}

generate_uuid() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr '[:upper:]' '[:lower:]'
    return
  fi
  if [ -r /proc/sys/kernel/random/uuid ]; then
    tr '[:upper:]' '[:lower:]' </proc/sys/kernel/random/uuid
    return
  fi
  python3 - <<'PY'
import uuid
print(uuid.uuid4().hex)
PY
}

SESSION_UUID="${INTERNDATA_NAV2_SESSION_UUID:-}"
if [ -z "${SESSION_UUID}" ]; then
  if [ -n "${STACK_ID}" ]; then
    SESSION_UUID="nav2_$(sanitize_id "${STACK_ID}")"
  else
    SESSION_UUID="$(generate_uuid)"
  fi
fi
export INTERNDATA_NAV2_SESSION_UUID="${SESSION_UUID}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}"

if [ -n "${STACK_ID}" ]; then
  SAFE_STACK_ID="$(sanitize_id "${STACK_ID}")"
  export INTERNDATA_COMPOSE_PROJECT="${INTERNDATA_COMPOSE_PROJECT:-isaac-nav2-stack-${SAFE_STACK_ID}}"
  export INTERNDATA_ISAAC_CONTAINER_NAME="${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac-${SAFE_STACK_ID}}"
  export INTERNDATA_NAV2_CONTAINER_NAME="${INTERNDATA_NAV2_CONTAINER_NAME:-nav2-${SAFE_STACK_ID}}"
  export ISAAC_CACHE_MAIN="${ISAAC_CACHE_MAIN:-../.docker/isaac-sim/${SAFE_STACK_ID}/cache/main}"
  export ISAAC_CACHE_COMPUTE="${ISAAC_CACHE_COMPUTE:-../.docker/isaac-sim/${SAFE_STACK_ID}/cache/computecache}"
  export ISAAC_LOGS="${ISAAC_LOGS:-../.docker/isaac-sim/${SAFE_STACK_ID}/logs}"
  export ISAAC_CONFIG="${ISAAC_CONFIG:-../.docker/isaac-sim/${SAFE_STACK_ID}/config}"
  export ISAAC_DATA="${ISAAC_DATA:-../.docker/isaac-sim/${SAFE_STACK_ID}/data}"
  export ISAAC_PKGS="${ISAAC_PKGS:-../.docker/isaac-sim/${SAFE_STACK_ID}/pkg}"
fi

export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${INTERNDATA_ISAAC_GPU_DEVICE_IDS:-${DEFAULT_SINGLE_GPU_DEVICE_IDS}}"
export INTERNDATA_LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
STOP_NAV2_WHEN_ISAAC_EXITS="${INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS:-${DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS}}"

export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${DEFAULT_LAUNCHER_CONFIG}}"

# INTERNDATA_NAV2_SKILL_OVERRIDES_JSON is intentionally left unset by default so
# the footprint/inflation values from the robot base YAML are used. Set it
# externally only when you need a temporary runtime override.

if [ "$#" -eq 0 ]; then
  set -- "${DEFAULT_SERVICES[@]}"
fi

echo "Using INTERNDATA_NAV2_SESSION_UUID=${INTERNDATA_NAV2_SESSION_UUID}"
echo "Using INTERNDATA_NAV2_SKILL_OVERRIDES_JSON=${INTERNDATA_NAV2_SKILL_OVERRIDES_JSON:-}"
echo "Using INTERNDATA_STACK_ID=${STACK_ID:-}"
echo "Using INTERNDATA_COMPOSE_PROJECT=${INTERNDATA_COMPOSE_PROJECT:-isaac-nav2-stack}"
echo "Using INTERNDATA_ISAAC_CONTAINER_NAME=${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac}"
echo "Using INTERNDATA_NAV2_CONTAINER_NAME=${INTERNDATA_NAV2_CONTAINER_NAME:-nav2}"
echo "Using INTERNDATA_ISAAC_GPU_DEVICE_IDS=${INTERNDATA_ISAAC_GPU_DEVICE_IDS}"
echo "Using ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "Using INTERNDATA_LAUNCHER_CONFIG=${INTERNDATA_LAUNCHER_CONFIG}"
echo "Using INTERNDATA_LAUNCHER_EXTRA_ARGS=${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
echo "Using INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS=${STOP_NAV2_WHEN_ISAAC_EXITS}"
echo "Compose file: ${COMPOSE_FILE}"
echo "Services: $*"

docker compose -f "${COMPOSE_FILE}" up -d "$@"

if [ "${STOP_NAV2_WHEN_ISAAC_EXITS}" = "1" ]; then
  start_isaac_nav2_watchdog=false
  has_isaac=false
  has_nav2=false
  for service in "$@"; do
    if [ "${service}" = "isaac" ]; then
      has_isaac=true
    elif [ "${service}" = "nav2" ]; then
      has_nav2=true
    fi
  done
  if [ "${has_isaac}" = "true" ] && [ "${has_nav2}" = "true" ]; then
    start_isaac_nav2_watchdog=true
  fi

  if [ "${start_isaac_nav2_watchdog}" = "true" ]; then
    isaac_container="${INTERNDATA_ISAAC_CONTAINER_NAME:-isaac}"
    nav2_container="${INTERNDATA_NAV2_CONTAINER_NAME:-nav2}"
    echo "Watching ${isaac_container}; ${nav2_container} will be stopped after Isaac exits."
    docker wait "${isaac_container}" >/dev/null 2>&1 || true
    docker stop "${nav2_container}" >/dev/null 2>&1 || true
  fi
fi
