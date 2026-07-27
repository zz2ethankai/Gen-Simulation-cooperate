"""Export native SimBox YAML as TaskIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .parser import parse_task_yaml_to_ir, parse_tasks_yaml_to_ir, validate_task_ir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-asset-checks", action="store_true")
    args = parser.parse_args()
    value = (
        parse_task_yaml_to_ir(args.input, args.task_index)
        if args.task_index is not None
        else parse_tasks_yaml_to_ir(args.input)
    )
    if args.validate:
        items = value if isinstance(value, list) else [value]
        validations = [
            validate_task_ir(item, repo_root=args.repo_root, check_assets=not args.skip_asset_checks)
            for item in items
        ]
        if not all(item["schema_ok"] and item["references_ok"] and item["compatibility_ok"] for item in validations):
            raise ValueError(f"TaskIR validation failed: {validations}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        args.output.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
