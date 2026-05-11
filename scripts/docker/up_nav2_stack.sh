#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${INTERNDATA_COMPOSE_FILE:-${REPO_ROOT}/docker/docker-compose.yml}"

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
  SESSION_UUID="$(generate_uuid)"
fi
export INTERNDATA_NAV2_SESSION_UUID="${SESSION_UUID}"

if [ "$#" -eq 0 ]; then
  set -- isaac nav2
fi

echo "Using INTERNDATA_NAV2_SESSION_UUID=${INTERNDATA_NAV2_SESSION_UUID}"
echo "Compose file: ${COMPOSE_FILE}"
echo "Services: $*"

exec docker compose -f "${COMPOSE_FILE}" up -d "$@"
