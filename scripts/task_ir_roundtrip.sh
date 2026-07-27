#!/bin/bash
# One-shot roundtrip: task YAML -> TaskIR -> native YAML
#
# Usage:
#   bash scripts/task_ir_roundtrip.sh <task_yaml>
#
# Optional flags:
#   --output <path>         Final output YAML path (default: output/task_ir_roundtrip/<stem>.roundtrip.yaml)
#   --skip-asset-checks     Skip filesystem asset checks during export
#   --keep-ir               Keep the intermediate TaskIR file (default: delete it)
#
# Examples:
#   bash scripts/task_ir_roundtrip.sh workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml
#   bash scripts/task_ir_roundtrip.sh workflows/simbox/core/configs/tasks/navigation/split_aloha/navigate_asset_obstacles.yaml --skip-asset-checks
#   bash scripts/task_ir_roundtrip.sh workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml --output output/test_run.yaml --keep-ir

set -euo pipefail

usage() {
    echo "Usage: bash $0 <task_yaml> [--output <path>] [--skip-asset-checks] [--keep-ir]"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_YAML="$1"
shift

CUSTOM_OUTPUT=""
SKIP_ASSET_CHECKS=false
KEEP_IR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        --skip-asset-checks)
            SKIP_ASSET_CHECKS=true
            shift
            ;;
        --keep-ir)
            KEEP_IR=true
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

INPUT_BASENAME="$(basename "$INPUT_YAML")"
INPUT_STEM="${INPUT_BASENAME%.*}"

IR_DIR="${REPO_ROOT}/output/task_ir"
mkdir -p "$IR_DIR"
IR_PATH="${IR_DIR}/${INPUT_STEM}.task_ir.yaml"

if [[ -n "$CUSTOM_OUTPUT" ]]; then
    OUTPUT_PATH="$CUSTOM_OUTPUT"
    if [[ ! "$OUTPUT_PATH" = /* ]]; then
        OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_PATH}"
    fi
else
    OUTPUT_DIR="${REPO_ROOT}/output/task_ir_roundtrip"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PATH="${OUTPUT_DIR}/${INPUT_STEM}.roundtrip.yaml"
fi

echo "=== TaskIR Roundtrip ==="
echo "Input YAML : $INPUT_YAML"
echo "TaskIR     : $IR_PATH"
echo "Output YAML: $OUTPUT_PATH"
echo ""

# Step 1: YAML -> TaskIR
EXPORT_CMD=(python -m agent.task_ir.cli
    --input "$INPUT_YAML"
    --output "$IR_PATH"
    --repo-root "$REPO_ROOT"
    --task-index 0
    --validate
)
if [[ "$SKIP_ASSET_CHECKS" == true ]]; then
    EXPORT_CMD+=(--skip-asset-checks)
fi

echo "[1/2] YAML -> TaskIR ..."
(cd "$REPO_ROOT" && "${EXPORT_CMD[@]}")
echo ""

# Step 2: TaskIR -> YAML
ASSEMBLE_CMD=(python -m agent.task_ir.assemble_cli
    --input "$IR_PATH"
    --output "$OUTPUT_PATH"
)

echo "[2/2] TaskIR -> YAML ..."
(cd "$REPO_ROOT" && "${ASSEMBLE_CMD[@]}")
echo ""

# Cleanup
if [[ "$KEEP_IR" == false ]]; then
    rm -f "$IR_PATH"
    echo "Cleaned up intermediate TaskIR file."
fi

echo ""
echo "=== Done ==="
echo "Roundtrip output: $OUTPUT_PATH"
echo "You can use this file directly in launcher configs."
