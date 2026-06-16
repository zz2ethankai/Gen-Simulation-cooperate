#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parallel launch settings. Edit these values to change the direct
# script behavior; environment variables with the same names still override them.
DEFAULT_LAUNCHER_CONFIG="configs/de_plan_with_render_template.yaml"
DEFAULT_PARALLEL_GPU_COUNT="4"
DEFAULT_PARALLEL_GPUS=""             # Empty means generate 0..GPU_COUNT-1.
DEFAULT_STACKS_PER_GPU="2"
DEFAULT_STACK_PREFIX="gpu"
DEFAULT_ROS_DOMAIN_BASE="10"
DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS="1"
DEFAULT_START_DELAY_SEC="0"
DEFAULT_SERVICES=(isaac nav2)

GPU_COUNT="${INTERNDATA_PARALLEL_GPU_COUNT:-${DEFAULT_PARALLEL_GPU_COUNT}}"
GPU_LIST="${INTERNDATA_PARALLEL_GPUS:-${DEFAULT_PARALLEL_GPUS}}"
STACKS_PER_GPU="${INTERNDATA_STACKS_PER_GPU:-${DEFAULT_STACKS_PER_GPU}}"
STACK_PREFIX="${INTERNDATA_PARALLEL_STACK_PREFIX:-${DEFAULT_STACK_PREFIX}}"
ROS_DOMAIN_BASE="${INTERNDATA_PARALLEL_ROS_DOMAIN_BASE:-${DEFAULT_ROS_DOMAIN_BASE}}"
STOP_NAV2_WHEN_ISAAC_EXITS="${INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS:-${DEFAULT_STOP_NAV2_WHEN_ISAAC_EXITS}}"
START_DELAY_SEC="${INTERNDATA_PARALLEL_START_DELAY_SEC:-${DEFAULT_START_DELAY_SEC}}"
USER_LAUNCHER_EXTRA_ARGS="${INTERNDATA_LAUNCHER_EXTRA_ARGS:-}"
LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-${DEFAULT_LAUNCHER_CONFIG}}"

if [ "$#" -eq 0 ]; then
  set -- "${DEFAULT_SERVICES[@]}"
fi

if [ -z "${GPU_LIST}" ]; then
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [ "${GPU_COUNT}" -lt 1 ]; then
    echo "ERROR: INTERNDATA_PARALLEL_GPU_COUNT must be a positive integer, got '${GPU_COUNT}'" >&2
    exit 2
  fi
  GPU_IDS=()
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index += 1)); do
    GPU_IDS+=("${gpu_index}")
  done
  GPU_LIST="${GPU_IDS[*]}"
else
  IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
fi

if ! [[ "${STACKS_PER_GPU}" =~ ^[0-9]+$ ]] || [ "${STACKS_PER_GPU}" -lt 1 ]; then
  echo "ERROR: INTERNDATA_STACKS_PER_GPU must be a positive integer, got '${STACKS_PER_GPU}'" >&2
  exit 2
fi

started=0
for raw_gpu_id in "${GPU_IDS[@]}"; do
  gpu_id="$(printf '%s' "${raw_gpu_id}" | xargs)"
  if [ -z "${gpu_id}" ]; then
    continue
  fi

  stack_gpu_id="$(printf '%s' "${gpu_id}" | tr -cs '[:alnum:]_.-' '-')"
  stack_gpu_id="${stack_gpu_id#-}"
  stack_gpu_id="${stack_gpu_id%-}"

  for ((stack_index = 0; stack_index < STACKS_PER_GPU; stack_index += 1)); do
    if [ "${STACKS_PER_GPU}" -eq 1 ]; then
      stack_id="${STACK_PREFIX}${stack_gpu_id:-${started}}"
    else
      stack_id="${STACK_PREFIX}${stack_gpu_id:-${started}}_${stack_index}"
    fi
    ros_domain_id="$((ROS_DOMAIN_BASE + started))"

    launcher_extra_args=()
    case "${LAUNCHER_CONFIG}" in
      *de_plan_with_render_template.yaml|*de_pipe_template.yaml)
        ;;
      *)
        echo "ERROR: INTERNDATA_LAUNCHER_CONFIG must point to plan_with_render or pipeline template, got '${LAUNCHER_CONFIG}'" >&2
        exit 2
        ;;
    esac
    if [ -n "${USER_LAUNCHER_EXTRA_ARGS}" ]; then
      launcher_extra_args+=(${USER_LAUNCHER_EXTRA_ARGS})
    fi

    echo "Starting stack '${stack_id}' on host GPU '${gpu_id}' with ROS_DOMAIN_ID=${ros_domain_id}"
    env \
      INTERNDATA_STACK_ID="${stack_id}" \
      INTERNDATA_ISAAC_GPU_DEVICE_IDS="${gpu_id}" \
      ROS_DOMAIN_ID="${ros_domain_id}" \
      INTERNDATA_LAUNCHER_CONFIG="${LAUNCHER_CONFIG}" \
      INTERNDATA_LAUNCHER_EXTRA_ARGS="${launcher_extra_args[*]}" \
      INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS=0 \
      "${SCRIPT_DIR}/up_nav2_stack_single_gpu.sh" "$@"

    if [ "${STOP_NAV2_WHEN_ISAAC_EXITS}" = "1" ]; then
      isaac_container="isaac-${stack_id}"
      nav2_container="nav2-${stack_id}"
      (
        echo "Watching ${isaac_container}; ${nav2_container} will be stopped after Isaac exits."
        docker wait "${isaac_container}" >/dev/null 2>&1 || exit 0
        docker stop "${nav2_container}" >/dev/null 2>&1 || true
      ) &
    fi

    started=$((started + 1))
    total_requested_stacks="$((${#GPU_IDS[@]} * STACKS_PER_GPU))"
    if [ "${START_DELAY_SEC}" != "0" ] && [ "${started}" -lt "${total_requested_stacks}" ]; then
      sleep "${START_DELAY_SEC}"
    fi
  done
done

echo "Started ${started} stack(s), launcher_config=${LAUNCHER_CONFIG}, gpus=${GPU_LIST}, stacks_per_gpu=${STACKS_PER_GPU}"
