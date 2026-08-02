#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${INTERNDATA_NAV2_STACK_MODE:-single}"

usage() {
  cat <<'USAGE'
Usage: scripts/docker/up_nav2_stack.sh [options] [isaac] [nav2]

Options:
  --single                       Use the single-GPU launcher (default).
  --multi                        Use the multi-GPU launcher.
  --stack-id ID                  Set INTERNDATA_STACK_ID.
  --gpu ID                       Set INTERNDATA_ISAAC_GPU_DEVICE_IDS for single mode.
  --launcher-config PATH         Set INTERNDATA_LAUNCHER_CONFIG.
  --launcher-extra-args ARGS     Set INTERNDATA_LAUNCHER_EXTRA_ARGS.
  --isaac-cpus COUNT             Limit the Isaac container to COUNT CPUs.
  --nav2-cpus COUNT              Limit the Nav2 container to COUNT CPUs.
  --keep-nav2                    Keep Nav2 running after Isaac exits.
  -h, --help                     Show this help.
USAGE
}

services=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --single)
      MODE="single"
      shift
      ;;
    --multi)
      MODE="multi"
      shift
      ;;
    --stack-id)
      export INTERNDATA_STACK_ID="${2:?--stack-id requires a value}"
      shift 2
      ;;
    --gpu)
      export INTERNDATA_ISAAC_GPU_DEVICE_IDS="${2:?--gpu requires a value}"
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
      export INTERNDATA_ISAAC_CPUS="${2:?--isaac-cpus requires a value}"
      shift 2
      ;;
    --nav2-cpus)
      export INTERNDATA_NAV2_CPUS="${2:?--nav2-cpus requires a value}"
      shift 2
      ;;
    --keep-nav2)
      export INTERNDATA_STOP_NAV2_WHEN_ISAAC_EXITS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        services+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      services+=("$1")
      shift
      ;;
  esac
done

case "${MODE}" in
  single)
    exec "${SCRIPT_DIR}/up_nav2_stack_single_gpu.sh" "${services[@]}"
    ;;
  multi)
    exec "${SCRIPT_DIR}/up_nav2_stack_multi_gpu.sh" "${services[@]}"
    ;;
  *)
    echo "Unsupported INTERNDATA_NAV2_STACK_MODE=${MODE}" >&2
    exit 2
    ;;
esac
