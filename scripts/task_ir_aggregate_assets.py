"""Aggregate TaskIR objects into an Asset Registry."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_ir.batch import load_or_parse_task_irs


def extract_object_usage(task_ir: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-object usage records from one TaskIR item."""
    objects = task_ir.get("objects", [])
    if not isinstance(objects, list):
        return []

    skill_counter_by_object: dict[str, Counter[str]] = defaultdict(Counter)
    for step in task_ir.get("skill_steps", []):
        if not isinstance(step, dict):
            continue
        skill_name = str(step.get("skill_name") or "unknown")
        refs = step.get("object_refs", [])
        if not isinstance(refs, list):
            continue
        for obj_name in refs:
            skill_counter_by_object[str(obj_name)][skill_name] += 1

    source_path = str(task_ir.get("_source_path") or (task_ir.get("metadata") or {}).get("source_yaml") or "")
    source_task_index = int(task_ir.get("_source_task_index") or (task_ir.get("metadata") or {}).get("source_task_index") or 0)
    task_ref = f"{source_path}#{source_task_index}"
    task_family = str(task_ir.get("task_family") or "unknown")

    usages: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_name = str(obj.get("name") or "")
        skills_applied = dict(skill_counter_by_object.get(object_name, Counter()))
        usages.append(
            {
                "task_ref": task_ref,
                "task_family": task_family,
                "object_name": object_name,
                "asset_path": obj.get("asset_path"),
                "target_class": obj.get("target_class"),
                "object_type": obj.get("object_type"),
                "dataset": obj.get("dataset"),
                "category": obj.get("category"),
                "capabilities": obj.get("capabilities") if isinstance(obj.get("capabilities"), dict) else {},
                "scale": obj.get("scale"),
                "euler": obj.get("euler"),
                "skills_applied": skills_applied,
            }
        )
    return usages


def aggregate_by_asset_path(all_usages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate object usages by unique ``asset_path``."""
    grouped: dict[str, dict[str, Any]] = {}

    for usage in all_usages:
        asset_key = _asset_key(usage.get("asset_path"), usage.get("object_name"))
        if asset_key not in grouped:
            grouped[asset_key] = {
                "asset_path": usage.get("asset_path"),
                "names_used": set(),
                "task_refs": set(),
                "task_family_counter": Counter(),
                "target_class_counter": Counter(),
                "object_type_counter": Counter(),
                "dataset_counter": Counter(),
                "category_counter": Counter(),
                "skills_counter": Counter(),
                "scales_observed": set(),
                "eulers_observed": set(),
                "capabilities": {
                    "is_pickable": False,
                    "is_place_target": False,
                    "is_container": False,
                    "requires_grasp_annotation": False,
                },
            }

        bucket = grouped[asset_key]
        bucket["names_used"].add(str(usage.get("object_name") or ""))
        bucket["task_refs"].add(str(usage.get("task_ref") or ""))
        bucket["task_family_counter"].update([str(usage.get("task_family") or "unknown")])
        bucket["target_class_counter"].update([str(usage.get("target_class") or "unknown")])
        bucket["object_type_counter"].update([str(usage.get("object_type") or "unknown")])
        bucket["dataset_counter"].update([str(usage.get("dataset") or "unknown")])
        bucket["category_counter"].update([str(usage.get("category") or "unknown")])
        bucket["skills_counter"].update(
            {str(skill): int(count) for skill, count in dict(usage.get("skills_applied") or {}).items()}
        )

        capabilities = usage.get("capabilities")
        if isinstance(capabilities, dict):
            for key in bucket["capabilities"].keys():
                bucket["capabilities"][key] = bool(bucket["capabilities"][key] or capabilities.get(key, False))

        scale_key = _serialize_value(usage.get("scale"))
        if scale_key is not None:
            bucket["scales_observed"].add(scale_key)
        euler_key = _serialize_value(usage.get("euler"))
        if euler_key is not None:
            bucket["eulers_observed"].add(euler_key)

    assets: list[dict[str, Any]] = []
    summary_object_type = Counter()
    summary_dataset = Counter()
    for asset_key, bucket in grouped.items():
        object_type = _most_common_key(bucket["object_type_counter"])
        dataset = _most_common_key(bucket["dataset_counter"])
        summary_object_type.update([object_type])
        summary_dataset.update([dataset])

        assets.append(
            {
                "_asset_key": asset_key,
                "asset_path": bucket["asset_path"],
                "target_class": _most_common_key(bucket["target_class_counter"]),
                "object_type": object_type,
                "dataset": dataset,
                "category": _most_common_key(bucket["category_counter"]),
                "usage_count": len(bucket["task_refs"]),
                "names_used": sorted(name for name in bucket["names_used"] if name),
                "capabilities": bucket["capabilities"],
                "scales_observed": [_deserialize_value(item) for item in sorted(bucket["scales_observed"])],
                "eulers_observed": [_deserialize_value(item) for item in sorted(bucket["eulers_observed"])],
                "skills_applied": _counter_to_ranked_skill_list(bucket["skills_counter"]),
                "task_families": sorted(bucket["task_family_counter"].keys()),
            }
        )

    assets.sort(key=lambda item: (-int(item["usage_count"]), str(item.get("asset_path") or item["_asset_key"])))
    summary = {
        "total_unique_assets": len(assets),
        "by_object_type": _counter_to_dict(summary_object_type),
        "by_dataset": _counter_to_dict(summary_dataset),
    }
    return assets, summary


def compute_co_occurrence(all_task_irs: list[dict[str, Any]], top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Compute asset co-occurrence frequency across tasks."""
    co_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for task_ir in all_task_irs:
        objects = task_ir.get("objects", [])
        if not isinstance(objects, list):
            continue
        asset_keys = sorted(
            {
                _asset_key(obj.get("asset_path"), obj.get("name"))
                for obj in objects
                if isinstance(obj, dict)
            }
        )
        for first, second in combinations(asset_keys, 2):
            co_counter[first][second] += 1
            co_counter[second][first] += 1

    result: dict[str, list[dict[str, Any]]] = {}
    for asset_key, counter in co_counter.items():
        ranked = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
        result[asset_key] = [
            {
                "asset": other_asset_key,
                "count": int(count),
            }
            for other_asset_key, count in ranked[:top_n]
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TaskIR objects into Asset Registry.")
    parser.add_argument(
        "--tasks-root",
        default="workflows/simbox/core/configs/tasks",
        help="Root directory of task YAML files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="output/task_ir_cache",
        help="TaskIR cache directory used by shared batch loader.",
    )
    parser.add_argument(
        "--output",
        default="output/knowledge/asset_registry.yaml",
        help="Output path for aggregated Asset Registry YAML.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional filter by task_family.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of TaskIR entries.",
    )
    parser.add_argument(
        "--top-cooccurrence",
        type=int,
        default=10,
        help="Top N co-occurring assets retained per asset entry.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_irs = load_or_parse_task_irs(tasks_root=tasks_root, cache_dir=cache_dir)
    if args.category:
        category_text = str(args.category).strip().lower()
        task_irs = [item for item in task_irs if str(item.get("task_family", "")).lower() == category_text]
    if args.limit is not None:
        task_irs = task_irs[: max(args.limit, 0)]

    all_usages: list[dict[str, Any]] = []
    for task_ir in task_irs:
        all_usages.extend(extract_object_usage(task_ir))

    assets, summary = aggregate_by_asset_path(all_usages)
    co_occurrence = compute_co_occurrence(task_irs, top_n=max(0, args.top_cooccurrence))
    for item in assets:
        asset_key = item.pop("_asset_key")
        item["co_occurring_assets"] = co_occurrence.get(asset_key, [])

    payload = {
        "summary": {
            "total_task_irs": len(task_irs),
            **summary,
        },
        "assets": assets,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"TaskIR entries: {len(task_irs)}")
    print(f"Object usage records: {len(all_usages)}")
    print(f"Unique assets: {summary['total_unique_assets']}")
    print(f"Asset registry written to: {output_path}")


def _asset_key(asset_path: Any, object_name: Any) -> str:
    if asset_path not in (None, ""):
        return str(asset_path)
    name = str(object_name or "unknown_object")
    return f"__missing_asset_path__::{name}"


def _most_common_key(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {str(key): int(value) for key, value in ordered}


def _counter_to_ranked_skill_list(counter: Counter[str]) -> list[dict[str, Any]]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [{"skill": str(skill), "count": int(count)} for skill, count in ordered]


def _serialize_value(value: Any) -> str | None:
    if value is None:
        return None
    return yaml.safe_dump(value, sort_keys=False).strip()


def _deserialize_value(serialized: str) -> Any:
    return yaml.safe_load(serialized)


if __name__ == "__main__":
    main()
