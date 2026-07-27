#!/bin/bash
# Convert task YAML to TaskIR with concise defaults.
#
# Usage:
#   bash scripts/task_ir_export.sh <task_yaml> [task_index]
#
# Optional flags:
#   --output <path>           Custom output path (.json/.yaml/.yml)
#   --all                     Export all tasks in the yaml as a list
#   --no-validate             Skip validator
#   --skip-asset-checks       Skip filesystem asset checks
#
# Examples:
#   bash scripts/task_ir_export.sh workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml
#   bash scripts/task_ir_export.sh workflows/simbox/core/configs/tasks/art/franka/open_the_pot/open_the_pot.yaml 0
#   bash scripts/task_ir_export.sh workflows/simbox/core/configs/tasks/navigation/split_aloha/navigate_asset_obstacles.yaml --skip-asset-checks
#   bash scripts/task_ir_export.sh workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml --output output/task_ir/demo.json

set -euo pipefail

usage() {
    echo "Usage: bash $0 <task_yaml> [task_index] [--output <path>] [--all] [--no-validate] [--skip-asset-checks]"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_YAML="$1"
shift

TASK_INDEX="0"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    TASK_INDEX="$1"
    shift
fi

VALIDATE=true
SKIP_ASSET_CHECKS=false
CUSTOM_OUTPUT=""
EXPORT_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        --all)
            EXPORT_ALL=true
            shift
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        --skip-asset-checks)
            SKIP_ASSET_CHECKS=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ ! "$INPUT_YAML" = /* ]]; then
    INPUT_YAML="${REPO_ROOT}/${INPUT_YAML}"
fi

if [[ ! -f "$INPUT_YAML" ]]; then
    echo "Error: input yaml not found: $INPUT_YAML"
    exit 1
fi

if [[ -n "$CUSTOM_OUTPUT" ]]; then
    OUTPUT_PATH="$CUSTOM_OUTPUT"
    if [[ ! "$OUTPUT_PATH" = /* ]]; then
        OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_PATH}"
    fi
else
    INPUT_BASENAME="$(basename "$INPUT_YAML")"
    INPUT_STEM="${INPUT_BASENAME%.*}"
    OUTPUT_DIR="${REPO_ROOT}/output/task_ir"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PATH="${OUTPUT_DIR}/${INPUT_STEM}.task_ir.yaml"
fi

CMD=(python -m agent.task_ir.cli
    --input "$INPUT_YAML"
    --output "$OUTPUT_PATH"
    --repo-root "$REPO_ROOT"
)

if [[ "$EXPORT_ALL" == false ]]; then
    CMD+=(--task-index "$TASK_INDEX")
fi

if [[ "$VALIDATE" == true ]]; then
    CMD+=(--validate)
fi
if [[ "$SKIP_ASSET_CHECKS" == true ]]; then
    CMD+=(--skip-asset-checks)
fi

echo "Input       : $INPUT_YAML"
echo "Task index  : $TASK_INDEX"
echo "Export all  : $EXPORT_ALL"
echo "Output      : $OUTPUT_PATH"
echo "Validate    : $VALIDATE"
echo "Asset checks: $([[ "$SKIP_ASSET_CHECKS" == true ]] && echo "disabled" || echo "enabled")"
echo ""
echo "Running: ${CMD[*]}"
echo ""

(
    cd "$REPO_ROOT"
    "${CMD[@]}"
)

echo ""
echo "Done."
