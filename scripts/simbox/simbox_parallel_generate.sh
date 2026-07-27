#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/simbox/simbox_parallel_generate.sh [options] <task_yaml|task_list|task_dir>

Options:
  --backend docker|local       Execution backend, default: docker
  --gpus 0,1,2,3              Physical GPU ids, default: 0,1,2,3
  --workers-per-gpu N         Queue workers per GPU, default: 1
  --random-num N              Samples per task yaml, default: 10
  --split-random-num          Split each task yaml's random_num across queue
                              workers, so one yaml can use multiple GPUs
  --random-seed-base N        Optional. Task seed = base + queue index
  --scene-info KEY            Optional scene info key
  --de-config PATH            Data engine config, default: configs/simbox/de_plan_with_render_template.yaml
  --dataset-root PATH         Optional. Override de_config writer output_dir.
                              If omitted, output_dir is read from de_config;
                              ${name} is resolved with de_config.name, not
                              the per-task parallel run name.
  --run-id ID                 Optional run id, default: timestamp_pid
  --compose-file PATH         Docker compose file, default: docker/docker-compose.simbox.yml
  --isaac-python PATH         Local Isaac Python, default: /home/bld/ykqin/isaacsim/python.sh
  --estimate-mem-gb N         GPU preflight estimate per worker, default: 16
  --min-free-mem-gb N         Warn below this free memory, default: 12
  --max-gpu-util N            Warn above this utilization percent, default: 70
  --task-timeout-sec N        Per-task timeout, 0 disables it, default: 0
  --skip-task-preflight       Skip static task YAML path checks
  --stats-after-run           Generate dataset stats after real runs, default
  --no-stats-after-run        Skip dataset stats after real runs
  --dry-run                   Print planned commands and write manifest without executing tasks
  -h, --help                  Show this help

Environment:
  INTERNDATA_NVIDIA_SMI_BIN   Optional nvidia-smi command override for preflight tests

Examples:
  bash scripts/simbox/simbox_parallel_generate.sh --backend docker --gpus 0,1,2,3 --random-num 20 /tmp/tasks.txt
  bash scripts/simbox/simbox_parallel_generate.sh --backend docker --gpus 0,1,2,3 --random-num 20 --split-random-num task.yaml
  bash scripts/simbox/simbox_parallel_generate.sh --backend local --gpus 0,1 --random-num 1 --dry-run workflows/simbox/core/configs/tasks/example
  bash scripts/simbox/simbox_parallel_generate.sh --backend docker --gpus 0 --workers-per-gpu 2 task.yaml
EOF
}

backend="docker"
gpu_list="0,1,2,3"
workers_per_gpu="1"
random_num="10"
split_random_num="0"
random_seed_base=""
scene_info=""
de_config="configs/simbox/de_plan_with_render_template.yaml"
dataset_root=""
dataset_root_source="de_config"
run_id=""
compose_file="docker/docker-compose.simbox.yml"
isaac_image="${INTERNDATA_ISAAC_IMAGE:-local/interdata-isaac-sim-4.1.0-curobo:latest}"
isaac_python="/home/bld/ykqin/isaacsim/python.sh"
estimate_mem_gb="16"
min_free_mem_gb="12"
max_gpu_util="70"
task_timeout_sec="0"
task_preflight="1"
dry_run="0"
stats_after_run="1"
task_source=""
nvidia_smi_bin="${INTERNDATA_NVIDIA_SMI_BIN:-nvidia-smi}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --backend)
            backend="${2:-}"
            shift 2
            ;;
        --gpus)
            gpu_list="${2:-}"
            shift 2
            ;;
        --workers-per-gpu)
            workers_per_gpu="${2:-}"
            shift 2
            ;;
        --random-num)
            random_num="${2:-}"
            shift 2
            ;;
        --split-random-num)
            split_random_num="1"
            shift
            ;;
        --no-split-random-num)
            split_random_num="0"
            shift
            ;;
        --random-seed-base)
            random_seed_base="${2:-}"
            shift 2
            ;;
        --scene-info)
            scene_info="${2:-}"
            shift 2
            ;;
        --de-config)
            de_config="${2:-}"
            shift 2
            ;;
        --dataset-root)
            dataset_root="${2:-}"
            dataset_root_source="cli"
            shift 2
            ;;
        --run-id)
            run_id="${2:-}"
            shift 2
            ;;
        --compose-file)
            compose_file="${2:-}"
            shift 2
            ;;
        --isaac-python)
            isaac_python="${2:-}"
            shift 2
            ;;
        --estimate-mem-gb)
            estimate_mem_gb="${2:-}"
            shift 2
            ;;
        --min-free-mem-gb)
            min_free_mem_gb="${2:-}"
            shift 2
            ;;
        --max-gpu-util)
            max_gpu_util="${2:-}"
            shift 2
            ;;
        --task-timeout-sec)
            task_timeout_sec="${2:-}"
            shift 2
            ;;
        --skip-task-preflight)
            task_preflight="0"
            shift
            ;;
        --stats-after-run)
            stats_after_run="1"
            shift
            ;;
        --no-stats-after-run)
            stats_after_run="0"
            shift
            ;;
        --dry-run)
            dry_run="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            if [ -n "$task_source" ]; then
                echo "Error: multiple task sources provided: '$task_source' and '$1'" >&2
                exit 2
            fi
            task_source="$1"
            shift
            ;;
    esac
done

if [ -z "$task_source" ]; then
    usage
    exit 2
fi

if [ "$backend" != "docker" ] && [ "$backend" != "local" ]; then
    echo "Error: --backend must be docker or local, got '$backend'" >&2
    exit 2
fi

if ! [[ "$workers_per_gpu" =~ ^[0-9]+$ ]] || [ "$workers_per_gpu" -lt 1 ]; then
    echo "Error: --workers-per-gpu must be a positive integer" >&2
    exit 2
fi

if ! [[ "$random_num" =~ ^[0-9]+$ ]] || [ "$random_num" -lt 1 ]; then
    echo "Error: --random-num must be a positive integer" >&2
    exit 2
fi

if [ -n "$random_seed_base" ] && ! [[ "$random_seed_base" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --random-seed-base must be an integer" >&2
    exit 2
fi

if ! [[ "$task_timeout_sec" =~ ^[0-9]+$ ]]; then
    echo "Error: --task-timeout-sec must be a non-negative integer" >&2
    exit 2
fi

if [ "$task_timeout_sec" -gt 0 ] && ! command -v timeout >/dev/null 2>&1; then
    echo "Error: --task-timeout-sec requires the timeout command" >&2
    exit 1
fi

cd "$REPO_ROOT" || exit 1

if [ -z "$run_id" ]; then
    run_id="$(date +%Y%m%d_%H%M%S)_$$"
fi
safe_run_id="$(printf '%s' "$run_id" | tr -cs '[:alnum:]_.-' '-')"
safe_run_id="${safe_run_id#-}"
safe_run_id="${safe_run_id%-}"
safe_run_id="${safe_run_id:-run}"

run_root="output/_parallel_runs"
run_dir="${run_root}/${safe_run_id}"
manifest_file="${run_dir}/manifest.jsonl"
manifest_lock="${run_dir}/manifest.lock"
failure_file="${run_dir}/failures.tsv"
queue_file="$(mktemp /tmp/interndata_parallel_queue.XXXXXX)"
task_file="$(mktemp /tmp/interndata_parallel_tasks.XXXXXX)"
queue_lock="$(mktemp /tmp/interndata_parallel_queue.lock.XXXXXX)"
running_containers_file="$(mktemp /tmp/interndata_parallel_containers.XXXXXX)"
pids=()

mkdir -p "$run_dir"
: > "$manifest_file"
: > "$failure_file"

cleanup() {
    rm -f "$queue_file" "$task_file" "$queue_lock" "$running_containers_file"
}

stop_children() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [ -s "$running_containers_file" ] && command -v docker >/dev/null 2>&1; then
        while IFS= read -r cid; do
            [ -n "$cid" ] || continue
            docker rm -f "$cid" >/dev/null 2>&1 || true
        done < "$running_containers_file"
    fi
    cleanup
}

trap 'stop_children; exit 130' INT TERM
trap cleanup EXIT

append_manifest() {
    local event="$1"
    shift
    flock "$manifest_lock" python3 - "$manifest_file" "$event" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path = sys.argv[1]
event = sys.argv[2]
pairs = sys.argv[3:]
record = {
    "event": event,
    "time": datetime.now(timezone.utc).isoformat(),
}
for pair in pairs:
    key, _, value = pair.partition("=")
    record[key] = value
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

normalize_repo_path() {
    local path="$1"
    if [[ "$path" == "$REPO_ROOT/"* ]]; then
        printf '%s\n' "${path#$REPO_ROOT/}"
    else
        printf '%s\n' "$path"
    fi
}

resolve_host_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$REPO_ROOT/$path"
    fi
}

ensure_repo_visible_for_docker() {
    local rel_path="$1"
    if [[ "$rel_path" = /* ]]; then
        echo "Error: Docker backend cannot see absolute path outside repo mount: $rel_path" >&2
        exit 2
    fi
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

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

validate_task_configs() {
    local task_list_file="$1"
    python3 - "$task_list_file" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    print(f"Error: PyYAML is required for task preflight: {exc}", file=sys.stderr)
    raise SystemExit(1)

task_list = Path(sys.argv[1])
errors: list[str] = []


def should_check(path_value: object) -> bool:
    return isinstance(path_value, str) and path_value.strip() and "${" not in path_value


def check_path(task_yaml: Path, field: str, value: object) -> None:
    if not should_check(value):
        return
    raw = str(value).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        errors.append(f"{task_yaml}: missing {field}: {raw}")


for line in task_list.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    task_yaml = Path(line.strip())
    try:
        with task_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - report all config load failures together.
        errors.append(f"{task_yaml}: failed to parse YAML: {type(exc).__name__}: {exc}")
        continue

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        errors.append(f"{task_yaml}: expected top-level 'tasks' list")
        continue

    for task_idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"{task_yaml}: tasks[{task_idx}] is not a mapping")
            continue
        check_path(task_yaml, f"tasks[{task_idx}].arena_file", task.get("arena_file"))
        for robot_idx, robot in enumerate(task.get("robots") or []):
            if isinstance(robot, dict):
                check_path(
                    task_yaml,
                    f"tasks[{task_idx}].robots[{robot_idx}].robot_config_file",
                    robot.get("robot_config_file"),
                )
        for camera_idx, camera in enumerate(task.get("cameras") or []):
            if isinstance(camera, dict):
                check_path(
                    task_yaml,
                    f"tasks[{task_idx}].cameras[{camera_idx}].camera_file",
                    camera.get("camera_file"),
                )

if errors:
    print("Task preflight failed:", file=sys.stderr)
    for error in errors[:50]:
        print(f"  - {error}", file=sys.stderr)
    if len(errors) > 50:
        print(f"  ... {len(errors) - 50} more error(s)", file=sys.stderr)
    raise SystemExit(1)
PY
}

de_config="$(normalize_repo_path "$de_config")"
compose_file="$(normalize_repo_path "$compose_file")"

if [ ! -f "$de_config" ]; then
    echo "Error: data engine config not found: $de_config" >&2
    exit 1
fi

if [ -z "$dataset_root" ]; then
    dataset_root="$(read_de_config_output_dir "$de_config")" || exit 1
    dataset_root_source="de_config"
fi
dataset_root="$(normalize_repo_path "$dataset_root")"
dataset_root="${dataset_root%/}"
dataset_root_host="$(resolve_host_path "$dataset_root")"

if [ "$backend" = "docker" ]; then
    ensure_repo_visible_for_docker "$de_config"
    ensure_repo_visible_for_docker "$dataset_root"
    if [ ! -f "$compose_file" ]; then
        echo "Error: compose file not found: $compose_file" >&2
        exit 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker is not available" >&2
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "Error: docker compose plugin is not available" >&2
        exit 1
    fi
    if [ "$dry_run" != "1" ] && ! docker info >/dev/null 2>&1; then
        echo "Error: cannot connect to Docker daemon. Check whether Docker is running and this user can access /var/run/docker.sock." >&2
        exit 1
    fi
    if [ "$dry_run" != "1" ] && ! docker image inspect "$isaac_image" >/dev/null 2>&1; then
        echo "Error: Docker image not found: $isaac_image" >&2
        echo "Build it first with: docker compose -f $compose_file build isaac" >&2
        exit 1
    fi
else
    if [ ! -x "$isaac_python" ]; then
        echo "Error: Isaac Sim python launcher not executable: $isaac_python" >&2
        exit 1
    fi
fi

mkdir -p "$dataset_root_host"

IFS="," read -r -a gpus <<< "$gpu_list"
if [ "${#gpus[@]}" -eq 0 ]; then
    echo "Error: --gpus is empty" >&2
    exit 2
fi

gpu_count=0
for raw_gpu in "${gpus[@]}"; do
    gpu="$(trim "$raw_gpu")"
    [ -n "$gpu" ] || continue
    gpu_count=$((gpu_count + 1))
done
if [ "$gpu_count" -eq 0 ]; then
    echo "Error: --gpus is empty" >&2
    exit 2
fi
total_workers=$((gpu_count * workers_per_gpu))

case "$task_source" in
    *.yaml|*.yml)
        task_path="$(normalize_repo_path "$task_source")"
        if [ ! -f "$task_path" ]; then
            echo "Error: task yaml not found: $task_source" >&2
            exit 1
        fi
        printf "%s\n" "$task_path" > "$task_file"
        ;;
    *)
        if [ -d "$task_source" ]; then
            find "$task_source" -type f \( -name "*.yaml" -o -name "*.yml" \) | sort | while IFS= read -r path; do
                normalize_repo_path "$path"
            done > "$task_file"
        elif [ -f "$task_source" ]; then
            while IFS= read -r path; do
                case "$path" in
                    ''|\#*) continue ;;
                esac
                normalize_repo_path "$path"
            done < "$task_source" > "$task_file"
        else
            echo "Error: task source is not a yaml, list file, or directory: $task_source" >&2
            exit 1
        fi
        ;;
esac

if [ ! -s "$task_file" ]; then
    echo "Error: no task yaml files found from: $task_source" >&2
    exit 1
fi

if [ "$task_preflight" = "1" ]; then
    validate_task_configs "$task_file" || exit 1
fi

idx=0
task_yaml_count=0
while IFS= read -r task_path; do
    if [ ! -f "$task_path" ]; then
        echo "Error: queued task yaml does not exist: $task_path" >&2
        exit 1
    fi
    if [ "$backend" = "docker" ]; then
        ensure_repo_visible_for_docker "$task_path"
    fi
    task_yaml_count=$((task_yaml_count + 1))
    if [ "$split_random_num" = "1" ]; then
        shard_count="$total_workers"
        if [ "$random_num" -lt "$shard_count" ]; then
            shard_count="$random_num"
        fi
        base_random_num=$((random_num / shard_count))
        remainder_random_num=$((random_num % shard_count))
        for ((shard_idx = 0; shard_idx < shard_count; shard_idx += 1)); do
            job_random_num="$base_random_num"
            if [ "$shard_idx" -lt "$remainder_random_num" ]; then
                job_random_num=$((job_random_num + 1))
            fi
            printf "%s\t%s\t%s\t%s\t%s\n" "$idx" "$task_path" "$job_random_num" "$shard_idx" "$shard_count" >> "$queue_file"
            idx=$((idx + 1))
        done
    else
        printf "%s\t%s\t%s\t%s\t%s\n" "$idx" "$task_path" "$random_num" "0" "1" >> "$queue_file"
        idx=$((idx + 1))
    fi
done < "$task_file"
total_tasks="$idx"

if [ "$split_random_num" != "1" ] && [ "$task_yaml_count" -lt "$total_workers" ] && [ "$random_num" -gt 1 ]; then
    echo "Warning: queued task YAMLs ($task_yaml_count) fewer than workers ($total_workers). Only $task_yaml_count worker(s) can be busy."
    echo "         Use --split-random-num to split random samples from each YAML across GPUs."
    append_manifest "schedule_warning" "reason=fewer_task_yamls_than_workers" \
        "task_yaml_count=$task_yaml_count" "total_workers=$total_workers" "random_num=$random_num"
fi

slugify_task() {
    local task_path="$1"
    local slug hash
    slug="$task_path"
    slug="${slug#./}"
    slug="${slug#workflows/simbox/core/configs/tasks/}"
    slug="${slug%.yaml}"
    slug="${slug%.yml}"
    slug="$(printf '%s' "$slug" | tr '/ :' '___' | tr -cs '[:alnum:]_.-' '-')"
    slug="${slug#-}"
    slug="${slug%-}"
    hash="$(printf '%s' "$task_path" | sha1sum | awk '{print substr($1,1,8)}')"
    slug="${slug:0:80}"
    printf '%s_%s' "${slug:-task}" "$hash"
}

print_preflight() {
    append_manifest "preflight_start" "backend=$backend" "gpus=$gpu_list" "workers_per_gpu=$workers_per_gpu" \
        "estimate_mem_gb=$estimate_mem_gb" "min_free_mem_gb=$min_free_mem_gb" "max_gpu_util=$max_gpu_util"

    if ! command -v "$nvidia_smi_bin" >/dev/null 2>&1; then
        echo "GPU preflight: nvidia-smi not found; skipping load-based advice."
        append_manifest "preflight_warning" "reason=nvidia-smi_not_found"
        return
    fi

    echo "GPU preflight:"
    "$nvidia_smi_bin" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
        while IFS=',' read -r idx_raw name_raw used_raw total_raw util_raw; do
            gpu_idx="$(trim "$idx_raw")"
            name="$(trim "$name_raw")"
            used_mb="$(trim "$used_raw")"
            total_mb="$(trim "$total_raw")"
            util_pct="$(trim "$util_raw")"

            selected="0"
            for selected_gpu_raw in "${gpus[@]}"; do
                selected_gpu="$(trim "$selected_gpu_raw")"
                if [ "$gpu_idx" = "$selected_gpu" ]; then
                    selected="1"
                    break
                fi
            done
            [ "$selected" = "1" ] || continue

            python3 - "$gpu_idx" "$name" "$used_mb" "$total_mb" "$util_pct" "$estimate_mem_gb" "$min_free_mem_gb" "$max_gpu_util" <<'PY'
import math
import sys

idx, name, used_mb, total_mb, util_pct, estimate_gb, min_free_gb, max_util = sys.argv[1:]
used_mb = float(used_mb)
total_mb = float(total_mb)
util_pct = float(util_pct)
estimate_gb = float(estimate_gb)
min_free_gb = float(min_free_gb)
max_util = float(max_util)
free_gb = max((total_mb - used_mb) / 1024.0, 0.0)
suggested = int(math.floor(free_gb / estimate_gb)) if estimate_gb > 0 else 1
if free_gb < min_free_gb or util_pct > max_util:
    suggested = 0
print(f"  GPU {idx} {name}: used={used_mb/1024.0:.1f}GB total={total_mb/1024.0:.1f}GB free={free_gb:.1f}GB util={util_pct:.0f}% suggested_workers={max(suggested, 0)}")
PY
            append_manifest "preflight_gpu" "gpu=$gpu_idx" "name=$name" "memory_used_mb=$used_mb" \
                "memory_total_mb=$total_mb" "utilization_gpu_percent=$util_pct"
        done
}

build_launcher_args() {
    local queue_idx="$1"
    local cfg_path="$2"
    local name="$3"
    local seed="$4"
    local job_random_num="$5"
    LAUNCHER_ARGS=(
        "--name=$name"
        "--load_stage.scene_loader.args.cfg_path=$cfg_path"
        "--load_stage.scene_loader.args.simulator.active_gpu=0"
        "--load_stage.scene_loader.args.simulator.physics_gpu=0"
        "--load_stage.scene_loader.args.simulator.multi_gpu=false"
        "--load_stage.scene_loader.args.simulator.max_gpu_count=1"
        "--load_stage.layout_random_generator.args.random_num=$job_random_num"
        "--store_stage.writer.args.output_dir=${dataset_root}/"
    )
    if [ -n "$scene_info" ]; then
        LAUNCHER_ARGS+=("--load_stage.scene_loader.args.scene_info=$scene_info")
    fi
    if [ -n "$seed" ]; then
        LAUNCHER_ARGS+=("--random_seed=$seed")
    fi
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_task() {
    local gpu="$1"
    local worker="$2"
    local queue_idx="$3"
    local cfg_path="$4"
    local job_random_num="$5"
    local shard_idx="$6"
    local shard_count="$7"
    local slug seed worker_name task_dir data_engine_name stdout_log status cid cid_file start_time end_time container_name project cache_root

    worker_name="gpu${gpu}_w${worker}"
    slug="$(slugify_task "$cfg_path")"
    task_dir="${run_dir}/${worker_name}/q${queue_idx}_${slug}"
    mkdir -p "$task_dir"
    stdout_log="${task_dir}/${backend}.log"
    cid_file="${task_dir}/container_id.txt"
    data_engine_name="_parallel_runs/${safe_run_id}/${worker_name}/q${queue_idx}_${slug}"
    status=125
    cid=""
    if [ -n "$random_seed_base" ]; then
        seed=$((random_seed_base + queue_idx))
    else
        seed=""
    fi

    build_launcher_args "$queue_idx" "$cfg_path" "$data_engine_name" "$seed" "$job_random_num"
    start_time="$(date -Iseconds)"
    append_manifest "task_start" "queue_idx=$queue_idx" "task_path=$cfg_path" "gpu=$gpu" "worker=$worker" \
        "backend=$backend" "data_engine_name=$data_engine_name" "dataset_root=$dataset_root" "seed=$seed" \
        "task_dir=$task_dir" "start_time=$start_time" "job_random_num=$job_random_num" \
        "shard_idx=$shard_idx" "shard_count=$shard_count"

    echo "[${worker_name}] start q${queue_idx}/${total_tasks}: $cfg_path (random_num=$job_random_num shard=$shard_idx/$shard_count)"

    if [ "$backend" = "local" ]; then
        local cmd
        cmd=("$isaac_python" launcher.py --config "$de_config" "${LAUNCHER_ARGS[@]}")
        if [ "$dry_run" = "1" ]; then
            echo "[dry-run][${worker_name}] CUDA_VISIBLE_DEVICES=$gpu $(print_command "${cmd[@]}")" | tee "$stdout_log"
            status=0
        else
            (
                export CUDA_VISIBLE_DEVICES="$gpu"
                if [ "$task_timeout_sec" -gt 0 ]; then
                    timeout --kill-after=30s "${task_timeout_sec}s" "${cmd[@]}"
                else
                    "${cmd[@]}"
                fi
            ) > "$stdout_log" 2>&1
            status=$?
            if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
                append_manifest "task_timeout" "queue_idx=$queue_idx" "task_path=$cfg_path" \
                    "gpu=$gpu" "worker=$worker" "timeout_sec=$task_timeout_sec"
            fi
        fi
    else
        container_name="interdata-${safe_run_id}-${worker_name}-q${queue_idx}-${slug}"
        container_name="${container_name:0:180}"
        project="interdata-simbox-${safe_run_id}"
        cache_root="${REPO_ROOT}/.docker/isaac-sim/${safe_run_id}/${worker_name}"
        mkdir -p "$cache_root/cache/main" "$cache_root/cache/computecache" "$cache_root/logs" "$cache_root/config" "$cache_root/data" "$cache_root/pkg"

        local compose_cmd
        compose_cmd=(
            docker compose -f "$compose_file" -p "$project" run -d --no-deps --no-TTY
            --name "$container_name"
            isaac
            /isaac-sim/python.sh launcher.py --config "$de_config" "${LAUNCHER_ARGS[@]}"
        )

        if [ "$dry_run" = "1" ]; then
            {
                echo "[dry-run][${worker_name}] env INTERNDATA_ISAAC_GPU_DEVICE_IDS=$gpu ISAAC_CACHE_MAIN=$cache_root/cache/main ..."
                print_command "${compose_cmd[@]}"
            } | tee "$stdout_log"
            printf '%s\n' "$container_name" > "$cid_file"
            status=0
        else
            cid="$(
                env \
                    INTERNDATA_ISAAC_GPU_DEVICE_IDS="$gpu" \
                    ISAAC_CACHE_MAIN="$cache_root/cache/main" \
                    ISAAC_CACHE_COMPUTE="$cache_root/cache/computecache" \
                    ISAAC_LOGS="$cache_root/logs" \
                    ISAAC_CONFIG="$cache_root/config" \
                    ISAAC_DATA="$cache_root/data" \
                    ISAAC_PKGS="$cache_root/pkg" \
                    INTERNDATA_RUN_IMPORT_CHECKS="${INTERNDATA_RUN_IMPORT_CHECKS:-0}" \
                    "${compose_cmd[@]}" 2> "${task_dir}/docker_start.err"
            )"
            status=$?
            if [ "$status" -eq 0 ] && [ -n "$cid" ]; then
                printf '%s\n' "$cid" > "$cid_file"
                flock "$manifest_lock" bash -c 'printf "%s\n" "$1" >> "$2"' bash "$cid" "$running_containers_file"
                append_manifest "container_start" "queue_idx=$queue_idx" "task_path=$cfg_path" "gpu=$gpu" \
                    "worker=$worker" "container_id=$cid" "container_name=$container_name"
                docker logs -f "$cid" > "$stdout_log" 2>&1 &
                local log_pid="$!"
                if [ "$task_timeout_sec" -gt 0 ]; then
                    local wait_output wait_status
                    wait_output="$(timeout --kill-after=30s "${task_timeout_sec}s" docker wait "$cid" 2>>"$stdout_log")"
                    wait_status=$?
                    if [ "$wait_status" -eq 0 ] && [ -n "$wait_output" ]; then
                        status="${wait_output//$'\n'/}"
                    elif [ "$wait_status" -eq 124 ] || [ "$wait_status" -eq 137 ]; then
                        status=124
                        append_manifest "task_timeout" "queue_idx=$queue_idx" "task_path=$cfg_path" \
                            "gpu=$gpu" "worker=$worker" "container_id=$cid" "timeout_sec=$task_timeout_sec"
                        {
                            echo ""
                            echo "[parallel] task timeout after ${task_timeout_sec}s; stopping container ${cid}"
                        } >> "$stdout_log"
                        docker rm -f "$cid" >> "$stdout_log" 2>&1 || true
                    else
                        status=125
                    fi
                else
                    status="$(docker wait "$cid" 2>>"$stdout_log" || printf '125')"
                fi
                wait "$log_pid" 2>/dev/null || true
                docker rm "$cid" >/dev/null 2>&1 || true
            else
                cat "${task_dir}/docker_start.err" > "$stdout_log" 2>/dev/null || true
            fi
        fi
    fi

    end_time="$(date -Iseconds)"
    append_manifest "task_finish" "queue_idx=$queue_idx" "task_path=$cfg_path" "gpu=$gpu" "worker=$worker" \
        "backend=$backend" "exit_code=$status" "container_id=$cid" "log_file=$stdout_log" \
        "end_time=$end_time" "job_random_num=$job_random_num" "shard_idx=$shard_idx" "shard_count=$shard_count"

    if [ "$status" = "0" ]; then
        echo "[${worker_name}] done q${queue_idx}: $cfg_path"
    else
        echo "[${worker_name}] failed q${queue_idx}: $cfg_path (exit $status, log: $stdout_log)"
        flock "$manifest_lock" bash -c 'printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" >> "$6"' \
            bash "$status" "$gpu" "$worker" "$cfg_path" "$stdout_log" "$failure_file"
    fi
}

worker_loop() {
    local gpu="$1"
    local worker="$2"
    local item queue_idx cfg_path job_random_num shard_idx shard_count

    while true; do
        item="$(
            flock "$queue_lock" bash -c '
                queue_file="$1"
                [ -s "$queue_file" ] || exit 1
                IFS= read -r first_line < "$queue_file" || exit 1
                tail -n +2 "$queue_file" > "${queue_file}.tmp"
                mv "${queue_file}.tmp" "$queue_file"
                printf "%s\n" "$first_line"
            ' bash "$queue_file"
        )" || break

        IFS=$'\t' read -r queue_idx cfg_path job_random_num shard_idx shard_count <<< "$item"
        run_task "$gpu" "$worker" "$queue_idx" "$cfg_path" "$job_random_num" "$shard_idx" "$shard_count"
    done
}

run_dataset_stats() {
    local stats_dir stats_log status
    stats_dir="${run_dir}/dataset_stats"
    stats_log="${run_dir}/dataset_stats.log"
    append_manifest "dataset_stats_start" "dataset_root=$dataset_root" "output_dir=$stats_dir"
    if python3 scripts/simbox/simbox_dataset_stats.py "$dataset_root" --output-dir "$stats_dir" > "$stats_log" 2>&1; then
        status=0
    else
        status=$?
    fi
    append_manifest "dataset_stats_finish" "dataset_root=$dataset_root" "output_dir=$stats_dir" \
        "exit_code=$status" "log_file=$stats_log"
    if [ "$status" -eq 0 ]; then
        echo "Dataset stats written: $stats_dir"
    else
        echo "Dataset stats failed (exit $status, log: $stats_log)"
    fi
    return "$status"
}

append_manifest "run_start" "run_id=$safe_run_id" "backend=$backend" "task_source=$task_source" \
    "task_yaml_count=$task_yaml_count" "total_tasks=$total_tasks" "total_workers=$total_workers" \
    "split_random_num=$split_random_num" "gpus=$gpu_list" "workers_per_gpu=$workers_per_gpu" \
    "random_num=$random_num" "dataset_root=$dataset_root" "de_config=$de_config" "dry_run=$dry_run" \
    "stats_after_run=$stats_after_run" "task_timeout_sec=$task_timeout_sec" \
    "dataset_root_source=$dataset_root_source"

echo "Run id: $safe_run_id"
echo "Backend: $backend"
echo "Queued task YAMLs: $task_yaml_count"
echo "Queued jobs: $total_tasks"
echo "GPU list: $gpu_list"
echo "Workers per GPU: $workers_per_gpu"
echo "Total workers: $total_workers"
echo "Random num per task: $random_num"
echo "Split random num: $split_random_num"
echo "Task timeout sec: $task_timeout_sec"
echo "Dataset root: $dataset_root (source: $dataset_root_source)"
echo "Run records: $run_dir"

print_preflight

for raw_gpu in "${gpus[@]}"; do
    gpu="$(trim "$raw_gpu")"
    [ -n "$gpu" ] || continue
    for ((worker = 0; worker < workers_per_gpu; worker += 1)); do
        worker_loop "$gpu" "$worker" &
        pids+=("$!")
    done
done

overall_status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_status=1
    fi
done

if [ -s "$failure_file" ]; then
    echo ""
    echo "Some tasks failed:"
    cat "$failure_file"
    overall_status=1
fi

if [ "$dry_run" != "1" ] && [ "$stats_after_run" = "1" ]; then
    if ! run_dataset_stats; then
        overall_status=1
    fi
fi

append_manifest "run_finish" "run_id=$safe_run_id" "exit_code=$overall_status"
echo "All workers finished. Records: $run_dir"
exit "$overall_status"
