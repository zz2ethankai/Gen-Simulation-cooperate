#!/usr/bin/env bash
set -euo pipefail

# Default safety setting. Keep this at 0 to stop only InterndataEngine containers
# created by scripts/docker/up_simbox_isaac.sh. Set to 1 to stop every running
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
          $0 == "isaac" || $0 ~ /^isaac-/ {
            print
          }
        '
  )
fi

if [ "${#containers[@]}" -eq 0 ]; then
  echo "No matching running containers found."
else
  echo "Stopping containers:"
  printf '  %s\n' "${containers[@]}"
  docker stop "${containers[@]}"
fi

# Remove every container that is not running, including containers stopped
# above and containers that were only created. This does not remove images.
# mapfile -t stopped_containers < <(
#   docker ps --all --quiet \
#     --filter status=created \
#     --filter status=exited \
#     --filter status=dead
# )

# if [ "${#stopped_containers[@]}" -eq 0 ]; then
#   echo "No stopped containers found."
# else
#   echo "Removing stopped containers:"
#   printf '  %s\n' "${stopped_containers[@]}"
#   docker rm "${stopped_containers[@]}"
# fi
