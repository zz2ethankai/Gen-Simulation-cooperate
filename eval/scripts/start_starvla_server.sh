#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/home/bld/ykqin/starvla}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH=/path/to/starvla/checkpoint}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"
USE_BF16="${USE_BF16:-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

cmd=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT_PATH}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  cmd+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
