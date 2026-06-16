#!/usr/bin/env bash
set -euo pipefail

# Default safety setting. Keep this at 0 to stop only InterndataEngine containers
# created by scripts/docker/up_nav2_stack*.sh. Set to 1 to stop every running
# Docker container on the host.
DEFAULT_STOP_EVERY_RUNNING_CONTAINER="0"

STOP_EVERY_RUNNING_CONTAINER="${INTERNDATA_STOP_EVERY_RUNNING_CONTAINER:-${DEFAULT_STOP_EVERY_RUNNING_CONTAINER}}"

if [ "${STOP_EVERY_RUNNING_CONTAINER}" = "1" ]; then
  mapfile -t containers < <(docker ps --format '{{.Names}}')
else
  mapfile -t containers < <(
    docker ps --format '{{.Names}}' \
      | sort \
      | awk '
          $0 == "isaac" || $0 == "nav2" || $0 ~ /^isaac-/ || $0 ~ /^nav2-/ {
            print
          }
        '
  )
fi

if [ "${#containers[@]}" -eq 0 ]; then
  echo "No matching running containers found."
  exit 0
fi

echo "Stopping containers:"
printf '  %s\n' "${containers[@]}"
docker stop "${containers[@]}"
