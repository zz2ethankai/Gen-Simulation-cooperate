#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/simbox/simbox_docker_smoke.sh [options]

Options:
  --run-id ID             Report run id, default: docker_smoke_TIMESTAMP_PID
  --compose-file PATH     Compose file, default: docker/docker-compose.simbox.yml
  --de-config PATH        Data engine config, default: configs/simbox/de_plan_with_render_template.yaml
  --dataset-root PATH     Optional. Override de_config writer output_dir.
                          If omitted, output_dir is read from de_config;
                          ${name} is resolved with de_config.name.
  --task PATH             Smoke task yaml, default: workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml
  --gpus LIST             GPU ids for dry-run/real smoke, default: 0
  --workers-per-gpu N     Workers per GPU for dry-run/real smoke, default: 1
  --random-num N          Samples for real smoke, default: 1
  --task-timeout-sec N    Per-task timeout for real smoke, default: 600
  --build                 Build the Isaac image before import checks
  --no-build              Do not build image, default
  --import-check          Run container import check, default
  --no-import-check       Skip container import check
  --dry-run-only          Do not run a real generation task
  --no-permission-fix     Do not attempt sudo usermod when Docker socket is blocked
  -h, --help              Show this help
EOF
}

run_id=""
compose_file="docker/docker-compose.simbox.yml"
de_config="configs/simbox/de_plan_with_render_template.yaml"
dataset_root=""
dataset_root_source="de_config"
effective_dataset_root=""
task_path="workflows/simbox/core/configs/tasks/example/sort_the_rubbish.yaml"
gpus="0"
workers_per_gpu="1"
random_num="1"
task_timeout_sec="600"
build_image="0"
import_check="1"
dry_run_only="0"
permission_fix="1"
isaac_image="${INTERNDATA_ISAAC_IMAGE:-local/interdata-isaac-sim-4.1.0-curobo:latest}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id) run_id="${2:-}"; shift 2 ;;
        --compose-file) compose_file="${2:-}"; shift 2 ;;
        --de-config) de_config="${2:-}"; shift 2 ;;
        --dataset-root) dataset_root="${2:-}"; dataset_root_source="cli"; shift 2 ;;
        --task) task_path="${2:-}"; shift 2 ;;
        --gpus) gpus="${2:-}"; shift 2 ;;
        --workers-per-gpu) workers_per_gpu="${2:-}"; shift 2 ;;
        --random-num) random_num="${2:-}"; shift 2 ;;
        --task-timeout-sec) task_timeout_sec="${2:-}"; shift 2 ;;
        --build) build_image="1"; shift ;;
        --no-build) build_image="0"; shift ;;
        --import-check) import_check="1"; shift ;;
        --no-import-check) import_check="0"; shift ;;
        --dry-run-only) dry_run_only="1"; shift ;;
        --no-permission-fix) permission_fix="0"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

cd "$REPO_ROOT" || exit 1

if [ -z "$run_id" ]; then
    run_id="docker_smoke_$(date +%Y%m%d_%H%M%S)_$$"
fi
safe_run_id="$(printf '%s' "$run_id" | tr -cs '[:alnum:]_.-' '-')"
safe_run_id="${safe_run_id#-}"
safe_run_id="${safe_run_id%-}"
safe_run_id="${safe_run_id:-docker_smoke}"

run_dir="output/_parallel_runs/${safe_run_id}"
log_dir="${run_dir}/logs"
steps_file="${run_dir}/docker_smoke_steps.jsonl"
report_json="${run_dir}/docker_smoke_report.json"
report_md="${run_dir}/docker_smoke_report.md"
mkdir -p "$log_dir"
: > "$steps_file"

first_gpu="${gpus%%,*}"
first_gpu="$(printf '%s' "$first_gpu" | xargs)"
first_gpu="${first_gpu:-0}"

read_de_config_output_dir() {
    local config_path="$1"
    python3 - "$config_path" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    print(f"Error: PyYAML is required to read output_dir from {sys.argv[1]}: {exc}", file=sys.stderr)
    raise SystemExit(1)

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

config_name = data.get("name")
node = data
for key in ("store_stage", "writer", "args", "output_dir"):
    if not isinstance(node, dict) or key not in node:
        print(f"Error: missing store_stage.writer.args.output_dir in {path}", file=sys.stderr)
        raise SystemExit(1)
    node = node[key]

if not isinstance(node, str) or not node.strip():
    print(f"Error: store_stage.writer.args.output_dir must be a non-empty string in {path}", file=sys.stderr)
    raise SystemExit(1)

output_dir = node.strip()
if "${name}" in output_dir:
    if not isinstance(config_name, str) or not config_name.strip():
        print(f"Error: {path} uses ${{name}} in output_dir but has no top-level name", file=sys.stderr)
        raise SystemExit(1)
    output_dir = output_dir.replace("${name}", config_name.strip())

print(output_dir.rstrip("/"))
PY
}

if [ -n "$dataset_root" ]; then
    effective_dataset_root="${dataset_root%/}"
else
    effective_dataset_root="$(read_de_config_output_dir "$de_config")" || exit 1
fi

record_step() {
    local name="$1"
    local status="$2"
    local exit_code="$3"
    local log_file="$4"
    local detail="$5"
    python3 - "$steps_file" "$name" "$status" "$exit_code" "$log_file" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, name, status, exit_code, log_file, detail = sys.argv[1:]
record = {
    "time": datetime.now(timezone.utc).isoformat(),
    "name": name,
    "status": status,
    "exit_code": int(exit_code),
    "log_file": log_file,
    "detail": detail,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

run_logged() {
    local name="$1"
    shift
    local log_file="${log_dir}/${name}.log"
    echo "[$name] $*"
    "$@" > "$log_file" 2>&1
    local status=$?
    if [ "$status" -eq 0 ]; then
        record_step "$name" "ok" "$status" "$log_file" ""
    else
        record_step "$name" "failed" "$status" "$log_file" ""
    fi
    return "$status"
}

run_logged_shell() {
    local name="$1"
    local script="$2"
    local log_file="${log_dir}/${name}.log"
    echo "[$name] $script"
    bash -lc "$script" > "$log_file" 2>&1
    local status=$?
    if [ "$status" -eq 0 ]; then
        record_step "$name" "ok" "$status" "$log_file" ""
    else
        record_step "$name" "failed" "$status" "$log_file" ""
    fi
    return "$status"
}

write_reports() {
    local final_status="$1"
    local blocker="$2"
    python3 - "$steps_file" "$report_json" "$report_md" "$final_status" "$blocker" "$safe_run_id" "$compose_file" "$de_config" "$effective_dataset_root" "$dataset_root_source" "$task_path" "$gpus" "$workers_per_gpu" "$random_num" "$task_timeout_sec" "$build_image" "$import_check" "$dry_run_only" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    steps_path,
    json_path,
    md_path,
    final_status,
    blocker,
    run_id,
    compose_file,
    de_config,
    dataset_root,
    dataset_root_source,
    task_path,
    gpus,
    workers_per_gpu,
    random_num,
    task_timeout_sec,
    build_image,
    import_check,
    dry_run_only,
) = sys.argv[1:]

steps = []
path = Path(steps_path)
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(json.loads(line))

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "status": final_status,
    "blocker": blocker,
    "compose_file": compose_file,
    "de_config": de_config,
    "dataset_root": dataset_root,
    "dataset_root_source": dataset_root_source,
    "task_path": task_path,
    "gpus": gpus,
    "workers_per_gpu": workers_per_gpu,
    "random_num": random_num,
    "task_timeout_sec": int(task_timeout_sec),
    "build_image": build_image == "1",
    "import_check": import_check == "1",
    "dry_run_only": dry_run_only == "1",
    "steps": steps,
}
Path(json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# SimBox Docker Smoke Report",
    "",
    f"- Run id: `{run_id}`",
    f"- Status: `{final_status}`",
    f"- Blocker: `{blocker or 'none'}`",
    f"- Compose file: `{compose_file}`",
    f"- Data engine config: `{de_config}`",
    f"- Dataset root: `{dataset_root}`",
    f"- Dataset root source: `{dataset_root_source}`",
    f"- Task: `{task_path}`",
    f"- GPUs: `{gpus}`",
    f"- Workers per GPU: `{workers_per_gpu}`",
    f"- Random num: `{random_num}`",
    f"- Task timeout sec: `{task_timeout_sec}`",
    "",
    "| Step | Status | Exit | Log | Detail |",
    "| --- | --- | ---: | --- | --- |",
]
for step in steps:
    lines.append(
        "| {name} | {status} | {exit_code} | `{log_file}` | {detail} |".format(
            name=step.get("name", ""),
            status=step.get("status", ""),
            exit_code=step.get("exit_code", ""),
            log_file=step.get("log_file", ""),
            detail=str(step.get("detail", "")).replace("|", "\\|"),
        )
    )
Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

collect_host_info() {
    {
        echo "pwd=$(pwd)"
        echo "user=$(id -un)"
        echo "id=$(id)"
        echo "groups=$(groups)"
        if [ -S /var/run/docker.sock ]; then
            ls -l /var/run/docker.sock
        else
            echo "/var/run/docker.sock not found"
        fi
        command -v docker || true
        docker context ls || true
        docker compose version || true
        command -v nvidia-smi || true
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    } > "${log_dir}/host_info.log" 2>&1
    record_step "host_info" "ok" 0 "${log_dir}/host_info.log" ""
}

attempt_docker_permission_fix() {
    local log_file="${log_dir}/docker_permission_fix.log"
    (
        echo "Docker daemon access failed."
        echo "Current identity:"
        id
        groups
        ls -l /var/run/docker.sock || true
        if [ "$permission_fix" != "1" ]; then
            echo "Permission fix disabled by --no-permission-fix."
            exit 2
        fi
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is not available."
            exit 3
        fi
        if ! sudo -n true >/dev/null 2>&1; then
            echo "passwordless sudo is not available."
            exit 4
        fi
        if ! getent group docker >/dev/null 2>&1; then
            echo "docker group does not exist."
            exit 5
        fi
        sudo usermod -aG docker "$(id -un)"
        echo "Added $(id -un) to docker group. Re-login or run 'newgrp docker' before retrying."
        exit 10
    ) > "$log_file" 2>&1
    local status=$?
    if [ "$status" -eq 10 ]; then
        record_step "docker_permission_fix" "needs_relogin" "$status" "$log_file" "user_added_to_docker_group"
    else
        record_step "docker_permission_fix" "failed" "$status" "$log_file" "run: sudo usermod -aG docker $(id -un), then re-login"
    fi
}

collect_host_info

overall_status=0
blocker=""

if ! run_logged "compose_config" docker compose -f "$compose_file" config; then
    overall_status=1
fi

if ! run_logged "docker_info" docker info; then
    attempt_docker_permission_fix
    blocker="cannot_connect_to_docker_daemon"
    write_reports "blocked" "$blocker"
    echo "Docker daemon is not accessible. Report: $report_md"
    exit 1
fi

if [ "$build_image" = "1" ]; then
    if ! run_logged "compose_build_isaac" docker compose -f "$compose_file" build isaac; then
        overall_status=1
    fi
elif ! run_logged "image_exists" docker image inspect "$isaac_image"; then
    blocker="missing_docker_image"
    write_reports "blocked" "$blocker"
    echo "Docker image is missing. Build first: docker compose -f $compose_file build isaac"
    echo "Docker smoke report: $report_md"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    run_logged "host_nvidia_smi" nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || overall_status=1
else
    record_step "host_nvidia_smi" "skipped" 0 "" "nvidia-smi_not_found"
fi

if [ "$import_check" = "1" ]; then
    compose_build_flag=""
    if [ "$build_image" = "1" ]; then
        compose_build_flag="--build"
    fi
    import_cmd="INTERNDATA_RUN_IMPORT_CHECKS=1 INTERNDATA_ISAAC_GPU_DEVICE_IDS=${first_gpu} docker compose -f ${compose_file@Q} -p ${safe_run_id@Q} run --rm ${compose_build_flag} --no-deps isaac bash -lc 'cd /workspace && /isaac-sim/python.sh -c \"import launcher; import nimbus; from nimbus.utils.utils import init_env; init_env(); import core.loggers.lmdb_logger; print(\\\"intern data import ok\\\")\"'"
    if ! run_logged_shell "container_import_check" "$import_cmd"; then
        overall_status=1
    fi
fi

dry_run_id="${safe_run_id}_parallel_dryrun"
dry_run_cmd=(
    bash scripts/simbox/simbox_parallel_generate.sh
    --backend docker
    --gpus "$gpus"
    --workers-per-gpu "$workers_per_gpu"
    --random-num "$random_num"
    --de-config "$de_config"
    --task-timeout-sec "$task_timeout_sec"
    --run-id "$dry_run_id"
    --dry-run
)
if [ -n "$dataset_root" ]; then
    dry_run_cmd+=(--dataset-root "$dataset_root")
fi
dry_run_cmd+=("$task_path")
if ! run_logged "parallel_dry_run" "${dry_run_cmd[@]}"; then
    overall_status=1
fi

if [ "$dry_run_only" != "1" ]; then
    real_run_id="${safe_run_id}_real"
    real_run_cmd=(
        bash scripts/simbox/simbox_parallel_generate.sh
        --backend docker
        --gpus "$gpus"
        --workers-per-gpu "$workers_per_gpu"
        --random-num "$random_num"
        --de-config "$de_config"
        --task-timeout-sec "$task_timeout_sec"
        --run-id "$real_run_id"
    )
    if [ -n "$dataset_root" ]; then
        real_run_cmd+=(--dataset-root "$dataset_root")
    fi
    real_run_cmd+=("$task_path")
    if ! run_logged "parallel_real_smoke" "${real_run_cmd[@]}"; then
        overall_status=1
    fi
fi

if [ -d "$effective_dataset_root" ]; then
    run_logged "dataset_stats" python3 scripts/simbox/simbox_dataset_stats.py "$effective_dataset_root" --output-dir "${run_dir}/dataset_stats" || overall_status=1
else
    record_step "dataset_stats" "skipped" 0 "" "dataset_root_not_found"
fi

if [ "$overall_status" -eq 0 ]; then
    write_reports "ok" ""
else
    write_reports "failed" ""
fi

echo "Docker smoke report: $report_md"
exit "$overall_status"
