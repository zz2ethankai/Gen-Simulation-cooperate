#!/bin/bash
# Convert TaskIR file back into native task YAML.
#
# Usage:
#   bash scripts/task_ir_to_yaml.sh <task_ir_file>
#
# Optional flags:
#   --output <path>   Custom output path (.yaml/.yml)
#
# Examples:
#   bash scripts/task_ir_to_yaml.sh output/task_ir/arrange_the_tableware_part0.task_ir.yaml
#   bash scripts/task_ir_to_yaml.sh output/task_ir/demo.json --output output/task_ir/demo.roundtrip.yaml

set -euo pipefail

usage() {
    echo "Usage: bash $0 <task_ir_file> [--output <path>]"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_FILE="$1"
shift

CUSTOM_OUTPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ ! "$INPUT_FILE" = /* ]]; then
    INPUT_FILE="${REPO_ROOT}/${INPUT_FILE}"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: task ir file not found: $INPUT_FILE"
    exit 1
fi

if [[ -n "$CUSTOM_OUTPUT" ]]; then
    OUTPUT_PATH="$CUSTOM_OUTPUT"
    if [[ ! "$OUTPUT_PATH" = /* ]]; then
        OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_PATH}"
    fi
else
    INPUT_BASENAME="$(basename "$INPUT_FILE")"
    INPUT_STEM="${INPUT_BASENAME%.*}"
    INPUT_STEM="${INPUT_STEM%.task_ir}"
    OUTPUT_DIR="${REPO_ROOT}/output/task_ir_roundtrip"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PATH="${OUTPUT_DIR}/${INPUT_STEM}.roundtrip.yaml"
fi

CMD=(python -m agent.task_ir.assemble_cli
    --input "$INPUT_FILE"
    --output "$OUTPUT_PATH"
)

echo "Input : $INPUT_FILE"
echo "Output: $OUTPUT_PATH"
echo ""
echo "Running: ${CMD[*]}"
echo ""

(
    cd "$REPO_ROOT"
    "${CMD[@]}"
)

echo ""
echo "Done."
