#!/bin/bash
# List and inspect Codex CLI conversation history.
#
# Usage:
#   bash scripts/codex_history.sh
#   bash scripts/codex_history.sh list [limit]
#   bash scripts/codex_history.sh show <index|session_id>
#   bash scripts/codex_history.sh open <index|session_id>
#   bash scripts/codex_history.sh export <index|session_id>
#   bash scripts/codex_history.sh raw <index|session_id>
#   bash scripts/codex_history.sh path <index|session_id>
#   bash scripts/codex_history.sh resume <index|session_id>
#
# Notes:
# - Index values come from the `list` output.
# - `show` / `export` creates a readable markdown transcript file.
# - `open` creates the markdown transcript and tries to open it with your editor or desktop opener.
# - `raw` opens the raw jsonl in `less` if available, otherwise prints it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
HISTORY_FILE="${CODEX_HOME}/history.jsonl"
SESSIONS_DIR="${CODEX_HOME}/sessions"
ARCHIVE_DIR="${CODEX_HOME}/archived_sessions"
EXPORT_DIR="${REPO_ROOT}/output/codex_history"
DEFAULT_LIMIT=20

usage() {
    echo "Usage:"
    echo "  bash $0"
    echo "  bash $0 list [limit]"
    echo "  bash $0 show <index|session_id>"
    echo "  bash $0 open <index|session_id>"
    echo "  bash $0 export <index|session_id>"
    echo "  bash $0 raw <index|session_id>"
    echo "  bash $0 path <index|session_id>"
    echo "  bash $0 resume <index|session_id>"
    exit 1
}

ensure_history_exists() {
    if [[ ! -f "$HISTORY_FILE" ]]; then
        echo "Error: history file not found: $HISTORY_FILE"
        exit 1
    fi
}

resolve_session_id() {
    local token="$1"
    if [[ "$token" =~ ^[0-9]+$ ]]; then
        python3 - "$HISTORY_FILE" "$token" <<'PY'
import json
import sys

history_path = sys.argv[1]
target_index = int(sys.argv[2])
latest = {}

with open(history_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        session_id = item.get("session_id")
        if not session_id:
            continue
        ts = item.get("ts", 0)
        text = " ".join(str(item.get("text", "")).split())
        latest[session_id] = {"session_id": session_id, "ts": ts, "text": text}

items = sorted(latest.values(), key=lambda x: x["ts"], reverse=True)
if target_index < 1 or target_index > len(items):
    raise SystemExit(f"Error: index out of range: {target_index} (available 1..{len(items)})")
print(items[target_index - 1]["session_id"])
PY
        return
    fi
    echo "$token"
}

find_session_file() {
    local session_id="$1"
    local found=""
    if [[ -d "$SESSIONS_DIR" ]]; then
        found="$(find "$SESSIONS_DIR" -type f -name "*${session_id}*.jsonl" | sort | tail -n 1 || true)"
    fi
    if [[ -z "$found" && -d "$ARCHIVE_DIR" ]]; then
        found="$(find "$ARCHIVE_DIR" -type f -name "*${session_id}*.jsonl" | sort | tail -n 1 || true)"
    fi
    if [[ -z "$found" ]]; then
        echo "Error: session file not found for session_id=$session_id" >&2
        exit 1
    fi
    echo "$found"
}

list_sessions() {
    local limit="${1:-$DEFAULT_LIMIT}"
    python3 - "$HISTORY_FILE" "$limit" <<'PY'
import datetime as dt
import json
import sys

history_path = sys.argv[1]
limit = int(sys.argv[2])
latest = {}

with open(history_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        session_id = item.get("session_id")
        if not session_id:
            continue
        ts = int(item.get("ts", 0))
        text = " ".join(str(item.get("text", "")).split())
        latest[session_id] = {"session_id": session_id, "ts": ts, "text": text}

items = sorted(latest.values(), key=lambda x: x["ts"], reverse=True)[:limit]

print(f"{'Idx':<4} {'Time':<19} {'Session ID':<36} Preview")
for idx, item in enumerate(items, start=1):
    when = dt.datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
    preview = item["text"][:100]
    print(f"{idx:<4} {when:<19} {item['session_id']:<36} {preview}")
PY
}

render_session_to_file() {
    local session_id="$1"
    local session_file="$2"
    mkdir -p "$EXPORT_DIR"
    local output_file="${EXPORT_DIR}/${session_id}.md"
    python3 - "$session_id" "$session_file" "$output_file" <<'PY'
import json
import sys

session_id = sys.argv[1]
session_file = sys.argv[2]
output_file = sys.argv[3]

def normalize_text(payload):
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = []
        for item in payload:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(payload, dict):
        text = payload.get("text")
        if text:
            return str(text)
    return ""

lines = []
lines.append(f"# Codex Session {session_id}")
lines.append("")
lines.append(f"- Session file: `{session_file}`")
lines.append("")
last_emitted = None

def emit(role, timestamp, text):
    global last_emitted
    text = text.strip()
    if not text:
        return
    dedup_key = (role, text)
    if dedup_key == last_emitted:
        return
    lines.append(f"## {role} [{timestamp}]")
    lines.append("")
    lines.append(text)
    lines.append("")
    last_emitted = dedup_key

with open(session_file, "r", encoding="utf-8") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        item = json.loads(raw)
        payload = item.get("payload", {})
        item_type = item.get("type")
        timestamp = item.get("timestamp", "")

        if item_type == "response_item" and payload.get("type") == "message":
            role = payload.get("role", "unknown")
            if role not in {"assistant", "user"}:
                continue
            content = payload.get("content", [])
            text_parts = []
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if text:
                        text_parts.append(str(text))
            text = "\n".join(text_parts).strip()
            emit(role.upper(), timestamp, text)
        elif item_type == "event_msg" and payload.get("type") == "agent_message":
            text = normalize_text(payload.get("message", ""))
            emit("ASSISTANT", timestamp, text)
        elif item_type == "event_msg" and payload.get("type") == "user_message":
            text = normalize_text(payload.get("message", ""))
            emit("USER", timestamp, text)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines).rstrip() + "\n")

print(output_file)
PY
}

open_raw_session() {
    local session_file="$1"
    if command -v less >/dev/null 2>&1; then
        less "$session_file"
    else
        sed -n '1,200p' "$session_file"
    fi
}

resume_session() {
    local session_id="$1"
    codex resume "$session_id"
}

open_exported_file() {
    local output_file="$1"
    if [[ -n "${EDITOR:-}" ]]; then
        "$EDITOR" "$output_file"
        return
    fi
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$output_file" >/dev/null 2>&1 &
        return
    fi
    if command -v open >/dev/null 2>&1; then
        open "$output_file" >/dev/null 2>&1 &
        return
    fi
    echo "$output_file"
}

ensure_history_exists

COMMAND="${1:-list}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$COMMAND" in
    list)
        list_sessions "${1:-$DEFAULT_LIMIT}"
        ;;
    show|export|open)
        [[ $# -ge 1 ]] || usage
        SESSION_ID="$(resolve_session_id "$1")"
        SESSION_FILE="$(find_session_file "$SESSION_ID")"
        OUTPUT_FILE="$(render_session_to_file "$SESSION_ID" "$SESSION_FILE")"
        if [[ "$COMMAND" == "open" ]]; then
            echo "Transcript file: $OUTPUT_FILE"
            open_exported_file "$OUTPUT_FILE"
        else
            echo "$OUTPUT_FILE"
        fi
        ;;
    raw)
        [[ $# -ge 1 ]] || usage
        SESSION_ID="$(resolve_session_id "$1")"
        SESSION_FILE="$(find_session_file "$SESSION_ID")"
        open_raw_session "$SESSION_FILE"
        ;;
    path)
        [[ $# -ge 1 ]] || usage
        SESSION_ID="$(resolve_session_id "$1")"
        find_session_file "$SESSION_ID"
        ;;
    resume)
        [[ $# -ge 1 ]] || usage
        SESSION_ID="$(resolve_session_id "$1")"
        resume_session "$SESSION_ID"
        ;;
    *)
        usage
        ;;
esac
