"""Aggregate TaskIR skill steps into a reusable Skill Catalog."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_ir.batch import load_or_parse_task_irs


def extract_skill_context(task_ir: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-step context entries from one TaskIR object."""
    steps = task_ir.get("skill_steps", [])
    if not isinstance(steps, list):
        return []

    object_by_name: dict[str, dict[str, Any]] = {}
    for obj in task_ir.get("objects", []):
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name:
            object_by_name[str(name)] = obj

    embodiment_by_robot: dict[str, str] = {}
    for robot in task_ir.get("robots", []):
        if not isinstance(robot, dict):
            continue
        robot_name = robot.get("name")
        embodiment = robot.get("embodiment")
        if robot_name:
            embodiment_by_robot[str(robot_name)] = str(embodiment or "unknown")

    contexts: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        skill_name = str(step.get("skill_name") or "unknown")
        robot_name = str(step.get("robot_name") or "")
        embodiment = embodiment_by_robot.get(robot_name, "unknown")
        arm = str(step.get("arm") or "unknown")

        object_refs = step.get("object_refs", [])
        object_types: list[str] = []
        categories: list[str] = []
        for ref in object_refs if isinstance(object_refs, list) else []:
            obj = object_by_name.get(str(ref))
            if not obj:
                continue
            object_type = obj.get("object_type")
            category = obj.get("category")
            if object_type:
                object_types.append(str(object_type))
            if category:
                categories.append(str(category))

        predecessor = _safe_skill_name(steps[index - 1]) if index > 0 else None
        successor = _safe_skill_name(steps[index + 1]) if index + 1 < len(steps) else None

        contexts.append(
            {
                "skill_name": skill_name,
                "embodiment": embodiment,
                "arm": arm,
                "object_types": object_types,
                "categories": categories,
                "param_keys": _dict_keys(step.get("params")),
                "planner_hint_keys": _dict_keys(step.get("planner_hints")),
                "timing_keys": _dict_keys(step.get("timing")),
                "success_keys": _dict_keys(step.get("success_criteria")),
                "predecessor": predecessor,
                "successor": successor,
            }
        )
    return contexts


def aggregate_by_skill(all_contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-step contexts by ``skill_name``."""
    grouped: dict[str, dict[str, Any]] = {}

    for ctx in all_contexts:
        skill_name = str(ctx.get("skill_name") or "unknown")
        if skill_name not in grouped:
            grouped[skill_name] = {
                "total_count": 0,
                "embodiment_counter": Counter(),
                "arm_counter": Counter(),
                "object_type_counter": Counter(),
                "category_counter": Counter(),
                "param_key_counter": Counter(),
                "planner_key_counter": Counter(),
                "timing_key_counter": Counter(),
                "success_key_counter": Counter(),
                "predecessor_counter": Counter(),
                "successor_counter": Counter(),
            }

        stat = grouped[skill_name]
        stat["total_count"] += 1
        stat["embodiment_counter"].update([str(ctx.get("embodiment") or "unknown")])
        stat["arm_counter"].update([str(ctx.get("arm") or "unknown")])
        stat["object_type_counter"].update([str(item) for item in ctx.get("object_types", [])])
        stat["category_counter"].update([str(item) for item in ctx.get("categories", [])])
        stat["param_key_counter"].update(ctx.get("param_keys", []))
        stat["planner_key_counter"].update(ctx.get("planner_hint_keys", []))
        stat["timing_key_counter"].update(ctx.get("timing_keys", []))
        stat["success_key_counter"].update(ctx.get("success_keys", []))
        stat["predecessor_counter"].update([ctx.get("predecessor")])
        stat["successor_counter"].update([ctx.get("successor")])

    return grouped


def extract_typical_sequences(all_task_irs: list[dict[str, Any]], top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Extract top 3-step sequences for each starting skill name."""
    sequence_counter_by_skill: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)

    for task_ir in all_task_irs:
        steps = task_ir.get("skill_steps", [])
        if not isinstance(steps, list):
            continue
        names = [_safe_skill_name(step) for step in steps if isinstance(step, dict)]
        names = [name for name in names if name]
        if len(names) < 3:
            continue
        for idx in range(len(names) - 2):
            sequence = (names[idx], names[idx + 1], names[idx + 2])
            sequence_counter_by_skill[names[idx]][sequence] += 1

    result: dict[str, list[dict[str, Any]]] = {}
    for skill_name, counter in sequence_counter_by_skill.items():
        entries = []
        for sequence, count in counter.most_common(top_n):
            entries.append({"sequence": list(sequence), "count": int(count)})
        result[skill_name] = entries
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TaskIR skill usage into Skill Catalog.")
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
        default="output/knowledge/skill_catalog.yaml",
        help="Output path for aggregated Skill Catalog YAML.",
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

    all_contexts: list[dict[str, Any]] = []
    for task_ir in task_irs:
        all_contexts.extend(extract_skill_context(task_ir))
    aggregated = aggregate_by_skill(all_contexts)
    typical_sequences = extract_typical_sequences(task_irs, top_n=10)

    skill_items = []
    for skill_name, stat in sorted(
        aggregated.items(),
        key=lambda item: (-int(item[1]["total_count"]), item[0]),
    ):
        total = int(stat["total_count"])
        param_keys = _summarize_keys(stat["param_key_counter"], total)
        skill_items.append(
            {
                "name": skill_name,
                "total_count": total,
                "description": None,
                "embodiment_distribution": _counter_to_dict(stat["embodiment_counter"]),
                "arm_distribution": _counter_to_dict(stat["arm_counter"]),
                "common_object_types": _counter_to_dict(stat["object_type_counter"]),
                "common_categories": _counter_to_ranked_list(stat["category_counter"], top_n=20),
                "param_keys": param_keys,
                "planner_hint_keys": _counter_to_key_frequency_list(stat["planner_key_counter"], total),
                "timing_keys": _counter_to_key_frequency_list(stat["timing_key_counter"], total),
                "success_criteria_keys": _counter_to_key_frequency_list(stat["success_key_counter"], total),
                "common_predecessors": _counter_to_skill_ratio_list(stat["predecessor_counter"], total),
                "common_successors": _counter_to_skill_ratio_list(stat["successor_counter"], total),
                "typical_sequences": typical_sequences.get(skill_name, []),
            }
        )

    payload = {
        "summary": {
            "total_task_irs": len(task_irs),
            "total_skill_steps": len(all_contexts),
            "total_unique_skills": len(skill_items),
        },
        "skills": skill_items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"TaskIR entries: {len(task_irs)}")
    print(f"Skill steps: {len(all_contexts)}")
    print(f"Unique skills: {len(skill_items)}")
    print(f"Skill catalog written to: {output_path}")


def _dict_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [str(key) for key in value.keys()]


def _safe_skill_name(step: Any) -> str | None:
    if not isinstance(step, dict):
        return None
    name = step.get("skill_name")
    if name in (None, ""):
        return None
    return str(name)


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {str(key): int(value) for key, value in ordered}


def _counter_to_ranked_list(counter: Counter[str], top_n: int) -> list[dict[str, Any]]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    result = []
    for key, value in ordered[:top_n]:
        result.append({"name": str(key), "count": int(value)})
    return result


def _counter_to_key_frequency_list(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [
        {
            "key": str(key),
            "count": int(count),
            "frequency": round(float(count) / float(total), 4),
        }
        for key, count in ordered
    ]


def _summarize_keys(counter: Counter[str], total: int) -> dict[str, Any]:
    if total <= 0:
        return {"always_present": [], "frequent": [], "occasional": []}

    always_present: list[str] = []
    frequent: list[dict[str, Any]] = []
    occasional: list[dict[str, Any]] = []

    for key, count in sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        frequency = float(count) / float(total)
        if frequency >= 1.0:
            always_present.append(str(key))
        elif frequency > 0.5:
            frequent.append({"key": str(key), "frequency": round(frequency, 4)})
        elif frequency >= 0.1:
            occasional.append({"key": str(key), "frequency": round(frequency, 4)})
    return {
        "always_present": always_present,
        "frequent": frequent,
        "occasional": occasional,
    }


def _counter_to_skill_ratio_list(counter: Counter[Any], total: int, top_n: int = 10) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    result = []
    for key, count in ordered[:top_n]:
        skill = None if key is None else str(key)
        result.append({"skill": skill, "ratio": round(float(count) / float(total), 4)})
    return result


if __name__ == "__main__":
    main()
