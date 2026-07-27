"""Assemble TaskIR back into native SimBox task YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .parser import assemble_task_ir_to_document, assemble_task_ir_to_task_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".json":
        value = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    document = (
        {"tasks": [assemble_task_ir_to_task_dict(item) for item in value]}
        if isinstance(value, list)
        else assemble_task_ir_to_document(value)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

