#!/usr/bin/env bash
set -euo pipefail

cd /data1/yifei/workspace/InterndataEngine

mkdir -p output/download_scene_conversion

PID_FILE=output/download_scene_conversion/grasp_labels.pid
LOG_FILE=output/download_scene_conversion/grasp_labels.log

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Existing grasp label generation is still running: PID $old_pid"
    echo "Stop it first with: kill $old_pid"
    echo "Then rerun this script."
    exit 1
  fi
fi

nohup /home/dyf/miniconda3/envs/anygrasp/bin/python \
  scripts/prepare_assets_addition_grasps.py \
  --source download-small-objects \
  --skip-existing \
  --jobs 4 \
  --gpus 0,1,2,3 \
  > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "Started parallel grasp label generation: PID $(cat "$PID_FILE")"
echo "Workers: 4"
echo "GPUs: 0,1,2,3"
echo "Log: $LOG_FILE"
echo "Follow with: tail -f $LOG_FILE"
