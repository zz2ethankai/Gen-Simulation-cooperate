"""Batch round-trip validation for TaskIR parse/assemble pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_ir import assemble_task_ir_to_task_dict, parse_task_yaml_to_ir, validate_task_ir
from agent.task_ir.batch import discover_task_yamls


def deep_diff(
    left: Any,
    right: Any,
    path: str = "",
    max_diffs: int = 200,
    diffs: list[str] | None = None,
) -> list[str]:
    """Recursively compare two nested python objects and return differing paths."""
    if diffs is None:
        diffs = []
    if len(diffs) >= max_diffs:
        return diffs

    if type(left) is not type(right):  # noqa: E721
        diffs.append(f"{path or '$'}: type mismatch ({type(left).__name__} != {type(right).__name__})")
        return diffs

    if isinstance(left, dict):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        missing_keys = sorted(left_keys - right_keys)
        extra_keys = sorted(right_keys - left_keys)
        for key in missing_keys:
            diffs.append(f"{_join_key(path, key)}: missing in roundtrip")
            if len(diffs) >= max_diffs:
                return diffs
        for key in extra_keys:
            diffs.append(f"{_join_key(path, key)}: extra in roundtrip")
            if len(diffs) >= max_diffs:
                return diffs
        for key in sorted(left_keys & right_keys):
            deep_diff(left[key], right[key], _join_key(path, key), max_diffs=max_diffs, diffs=diffs)
            if len(diffs) >= max_diffs:
                return diffs
        return diffs

    if isinstance(left, list):
        if len(left) != len(right):
            diffs.append(f"{path or '$'}: list length mismatch ({len(left)} != {len(right)})")
            if len(diffs) >= max_diffs:
                return diffs
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            deep_diff(left_item, right_item, f"{path}[{idx}]" if path else f"[{idx}]", max_diffs=max_diffs, diffs=diffs)
            if len(diffs) >= max_diffs:
                return diffs
        return diffs

    if left != right:
        diffs.append(f"{path or '$'}: value mismatch ({left!r} != {right!r})")
    return diffs


def process_single_yaml(
    yaml_path: Path,
    repo_root: Path,
    check_assets: bool,
    max_diffs_per_task: int,
    fail_fast: bool,
) -> dict[str, Any]:
    """Process one yaml file and return per-task roundtrip results."""
    yaml_path = Path(yaml_path).resolve()

    try:
        original_tasks = _load_tasks_from_yaml(yaml_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(yaml_path),
            "task_count": 0,
            "status": "load_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "results": [],
        }

    results: list[dict[str, Any]] = []
    for task_index, original_task in enumerate(original_tasks):
        task_name = original_task.get("name")
        task_family = _infer_task_family_from_path(yaml_path)
        embodiment = None

        try:
            task_ir = parse_task_yaml_to_ir(yaml_path, task_index=task_index)
            embodiment = _extract_primary_embodiment(task_ir)
        except Exception as exc:  # noqa: BLE001
            result = {
                "task_index": task_index,
                "task_name": task_name,
                "task_family": task_family,
                "embodiment": embodiment,
                "status": "parse_error",
                "diffs": [],
                "validation_issues": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(result)
            if fail_fast:
                raise
            continue

        try:
            validation = validate_task_ir(task_ir, repo_root=repo_root, check_assets=check_assets)
            validation_issues = validation.get("issues", [])
        except Exception as exc:  # noqa: BLE001
            result = {
                "task_index": task_index,
                "task_name": task_name,
                "task_family": task_family,
                "embodiment": embodiment,
                "status": "parse_error",
                "diffs": [],
                "validation_issues": [],
                "error_type": type(exc).__name__,
                "error": f"validation_failed: {exc}",
                "traceback": traceback.format_exc(),
            }
            results.append(result)
            if fail_fast:
                raise
            continue

        try:
            roundtrip_task = assemble_task_ir_to_task_dict(task_ir)
        except Exception as exc:  # noqa: BLE001
            result = {
                "task_index": task_index,
                "task_name": task_name,
                "task_family": task_family,
                "embodiment": embodiment,
                "status": "assemble_error",
                "diffs": [],
                "validation_issues": validation_issues,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(result)
            if fail_fast:
                raise
            continue

        diffs = deep_diff(original_task, roundtrip_task, max_diffs=max_diffs_per_task)
        status = "pass" if not diffs else "semantic_diff"
        result = {
            "task_index": task_index,
            "task_name": task_name,
            "task_family": task_family,
            "embodiment": embodiment,
            "status": status,
            "diffs": diffs,
            "validation_issues": validation_issues,
            "error_type": None,
            "error": None,
            "traceback": None,
        }
        results.append(result)

        if fail_fast and status != "pass":
            raise RuntimeError(f"Fail-fast triggered by {status} in {yaml_path}#{task_index}")

    status_counts: dict[str, int] = dict(Counter(str(item["status"]) for item in results))
    return {
        "path": str(yaml_path),
        "task_count": len(original_tasks),
        "status": "ok",
        "error_type": None,
        "error": None,
        "traceback": None,
        "status_counts": status_counts,
        "results": results,
    }


def aggregate_report(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate file-level results into a summary report."""
    status_counts: Counter[str] = Counter()
    by_task_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_embodiment: dict[str, Counter[str]] = defaultdict(Counter)
    error_patterns: Counter[str] = Counter()
    error_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_files = len(all_results)
    load_error_files = 0
    total_tasks = 0

    for file_result in all_results:
        file_path = file_result.get("path")
        if file_result.get("status") == "load_error":
            load_error_files += 1
            status_counts["load_error"] += 1
            pattern = _normalize_error_pattern(
                status="load_error",
                error_type=file_result.get("error_type"),
                error=file_result.get("error"),
            )
            error_patterns[pattern] += 1
            if len(error_samples[pattern]) < 3:
                error_samples[pattern].append(
                    {
                        "path": file_path,
                        "task_index": None,
                        "task_name": None,
                        "example": file_result.get("error"),
                    }
                )
            continue

        task_results = file_result.get("results", [])
        total_tasks += len(task_results)
        for item in task_results:
            status = str(item.get("status"))
            status_counts[status] += 1

            family = str(item.get("task_family") or "unknown")
            by_task_family[family]["total"] += 1
            by_task_family[family][status] += 1

            embodiment = str(item.get("embodiment") or "unknown")
            by_embodiment[embodiment]["total"] += 1
            by_embodiment[embodiment][status] += 1

            if status in {"parse_error", "assemble_error"}:
                pattern = _normalize_error_pattern(
                    status=status,
                    error_type=item.get("error_type"),
                    error=item.get("error"),
                )
                error_patterns[pattern] += 1
                if len(error_samples[pattern]) < 3:
                    error_samples[pattern].append(
                        {
                            "path": file_path,
                            "task_index": item.get("task_index"),
                            "task_name": item.get("task_name"),
                            "example": item.get("error"),
                        }
                    )

            if status == "semantic_diff":
                first_diff = (item.get("diffs") or ["unknown_diff"])[0]
                pattern = f"semantic_diff::{first_diff}"
                error_patterns[pattern] += 1
                if len(error_samples[pattern]) < 3:
                    error_samples[pattern].append(
                        {
                            "path": file_path,
                            "task_index": item.get("task_index"),
                            "task_name": item.get("task_name"),
                            "example": first_diff,
                        }
                    )

    pass_count = status_counts.get("pass", 0)
    pass_rate = (pass_count / total_tasks) if total_tasks > 0 else 0.0

    top_error_patterns = []
    for pattern, count in error_patterns.most_common(10):
        top_error_patterns.append(
            {
                "pattern": pattern,
                "count": count,
                "samples": error_samples.get(pattern, []),
            }
        )

    return {
        "summary": {
            "total_files": total_files,
            "load_error_files": load_error_files,
            "total_tasks": total_tasks,
            "status_counts": dict(status_counts),
            "pass_rate": round(pass_rate, 6),
        },
        "by_task_family": _counter_dict_to_plain(by_task_family),
        "by_embodiment": _counter_dict_to_plain(by_embodiment),
        "top_error_patterns": top_error_patterns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch roundtrip validation for TaskIR.")
    parser.add_argument(
        "--tasks-root",
        default="workflows/simbox/core/configs/tasks",
        help="Root directory of task YAML files.",
    )
    parser.add_argument(
        "--output",
        default="output/task_ir_batch_report",
        help="Output directory for batch report.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used by validator.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional task category filter (e.g. pick_and_place).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of yaml files for quick dry-run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first error/diff.",
    )
    parser.add_argument(
        "--check-assets",
        action="store_true",
        help="Enable validator asset existence checks (disabled by default).",
    )
    parser.add_argument(
        "--max-diffs-per-task",
        type=int,
        default=200,
        help="Maximum number of diff paths recorded per task.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).resolve()
    output_dir = Path(args.output).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_paths = discover_task_yamls(tasks_root, category=args.category)
    if args.limit is not None:
        yaml_paths = yaml_paths[: max(args.limit, 0)]

    all_results: list[dict[str, Any]] = []
    for yaml_path in yaml_paths:
        file_result = process_single_yaml(
            yaml_path=yaml_path,
            repo_root=repo_root,
            check_assets=args.check_assets,
            max_diffs_per_task=args.max_diffs_per_task,
            fail_fast=args.fail_fast,
        )
        all_results.append(file_result)

    aggregate = aggregate_report(all_results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks_root": str(tasks_root),
        "repo_root": str(repo_root),
        "category": args.category,
        "limit": args.limit,
        **aggregate,
        "files": all_results,
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report["summary"]
    status_counts = summary["status_counts"]
    print(f"Scanned yaml files: {summary['total_files']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Status counts: {status_counts}")
    print(f"Report written to: {report_path}")


def _load_tasks_from_yaml(yaml_path: Path) -> list[dict[str, Any]]:
    yaml_conf = OmegaConf.load(str(yaml_path))
    if "tasks" not in yaml_conf:
        raise ValueError(f"Expected 'tasks' key in {yaml_path}")
    tasks = OmegaConf.to_container(yaml_conf["tasks"], resolve=True)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected 'tasks' to be list in {yaml_path}")
    task_list: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError(f"Each task should be dict in {yaml_path}, got {type(item)}")
        task_list.append(item)
    return task_list


def _join_key(path: str, key: Any) -> str:
    key_text = str(key)
    return f"{path}.{key_text}" if path else key_text


def _infer_task_family_from_path(yaml_path: Path) -> str:
    parts = [part.lower() for part in yaml_path.parts]
    if "tasks" not in parts:
        return "unknown"
    idx = parts.index("tasks")
    if idx + 1 >= len(parts):
        return "unknown"
    return parts[idx + 1]


def _extract_primary_embodiment(task_ir: dict[str, Any]) -> str | None:
    robots = task_ir.get("robots", [])
    if not isinstance(robots, list) or not robots:
        return None
    first_robot = robots[0]
    if not isinstance(first_robot, dict):
        return None
    embodiment = first_robot.get("embodiment")
    if embodiment in (None, ""):
        return None
    return str(embodiment)


def _normalize_error_pattern(status: str, error_type: Any, error: Any) -> str:
    error_type_text = str(error_type or "UnknownError")
    error_line = str(error or "").splitlines()[0] if error is not None else ""
    return f"{status}::{error_type_text}::{error_line}"


def _counter_dict_to_plain(counter_dict: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    plain: dict[str, dict[str, int]] = {}
    for key in sorted(counter_dict.keys()):
        plain[key] = dict(counter_dict[key])
    return plain


if __name__ == "__main__":
    main()
