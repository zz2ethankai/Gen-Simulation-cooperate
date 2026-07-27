"""Parse native task YAML into a compact, lossless Agent-facing representation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _resolved_tasks(path: Path) -> list[dict[str, Any]]:
    document = OmegaConf.load(str(path))
    if "tasks" not in document:
        raise ValueError(f"task YAML has no tasks key: {path}")
    value = OmegaConf.to_container(document["tasks"], resolve=True)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"tasks must resolve to a list of mappings: {path}")
    return value


def _task_family(path: Path) -> str:
    parts = path.parts
    if "tasks" in parts:
        index = parts.index("tasks")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "assets" in parts and "basic" in parts:
        return "basic"
    return "unknown"


def _flatten_steps(task: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    step_index = 0
    for outer_index, robot_entry in enumerate(task.get("skills", []) or []):
        if not isinstance(robot_entry, dict):
            continue
        for robot_name, stages in robot_entry.items():
            for stage_index, stage in enumerate(stages or []):
                if not isinstance(stage, dict):
                    continue
                for arm, skills in stage.items():
                    for skill_index, skill in enumerate(skills or []):
                        if not isinstance(skill, dict):
                            continue
                        step_index += 1
                        name = str(skill.get("name") or "unknown").lower()
                        objects = [str(item) for item in skill.get("objects", []) or []]
                        params = {
                            key: copy.deepcopy(value)
                            for key, value in skill.items()
                            if key not in {"name", "objects"}
                        }
                        result.append(
                            {
                                "step_id": f"step_{step_index:03d}",
                                "stage_index": outer_index,
                                "phase_index": stage_index,
                                "skill_index": skill_index,
                                "robot_name": str(robot_name),
                                "arm": str(arm),
                                "skill_name": name,
                                "object_refs": objects,
                                "params": params,
                                "planner_hints": {
                                    key: copy.deepcopy(value)
                                    for key, value in params.items()
                                    if key.startswith("filter_") or key in {"test_mode", "use_batch"}
                                },
                                "timing": {
                                    key: copy.deepcopy(value)
                                    for key, value in params.items()
                                    if key.endswith("steps") or key.startswith("hesitate")
                                },
                                "success_criteria": {
                                    key: copy.deepcopy(value)
                                    for key, value in params.items()
                                    if key.startswith("success") or key in {"lift_th", "t_eps", "o_eps"}
                                },
                            }
                        )
    return result


def _objects(task: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked = {
        ref
        for step in steps
        if step["skill_name"] in {"pick", "dexpick", "dynamicpick", "manualpick"}
        for ref in step["object_refs"][:1]
    }
    place_targets = {
        ref
        for step in steps
        if step["skill_name"] in {"place", "dexplace"}
        for ref in step["object_refs"][1:]
    }
    result = []
    for cfg in task.get("objects", []) or []:
        if not isinstance(cfg, dict):
            continue
        name = str(cfg.get("name") or "")
        target_class = str(cfg.get("target_class") or "unknown")
        result.append(
            {
                "name": name,
                "asset_path": cfg.get("path"),
                "target_class": target_class,
                "object_type": target_class.replace("Object", "").lower() or "unknown",
                "dataset": cfg.get("dataset") or cfg.get("source_dataset") or "unknown",
                "category": cfg.get("asset_category") or cfg.get("category") or "unknown",
                "scale": copy.deepcopy(cfg.get("scale")),
                "euler": copy.deepcopy(cfg.get("euler")),
                "capabilities": {
                    "is_pickable": name in picked,
                    "is_place_target": name in place_targets,
                    "is_container": bool(
                        isinstance(cfg.get("container_affordance"), dict)
                        and cfg["container_affordance"].get("can_receive_objects")
                    ),
                    "requires_grasp_annotation": name in picked,
                },
            }
        )
    return result


def _parse_task(path: Path, task: dict[str, Any], task_index: int) -> dict[str, Any]:
    steps = _flatten_steps(task)
    robots = [
        {
            "name": str(item.get("name") or ""),
            "embodiment": str(item.get("target_class") or item.get("name") or "unknown"),
            "config_path": item.get("robot_config_file"),
        }
        for item in task.get("robots", []) or []
        if isinstance(item, dict)
    ]
    return {
        "task_name": str(task.get("name") or "unknown"),
        "task_class": str(task.get("task") or "unknown"),
        "task_family": _task_family(path),
        "robots": robots,
        "objects": _objects(task, steps),
        "skill_steps": steps,
        "metadata": {
            "source_yaml": str(path.resolve()),
            "source_task_index": task_index,
        },
        "_source_path": str(path.resolve()),
        "_source_task_index": task_index,
        "_native_task": copy.deepcopy(task),
    }


def parse_tasks_yaml_to_ir(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    return [_parse_task(source, task, index) for index, task in enumerate(_resolved_tasks(source))]


def parse_task_yaml_to_ir(path: str | Path, task_index: int = 0) -> dict[str, Any]:
    values = parse_tasks_yaml_to_ir(path)
    if not 0 <= task_index < len(values):
        raise IndexError(f"task_index {task_index} is outside 0..{len(values) - 1}")
    return values[task_index]


def assemble_task_ir_to_task_dict(task_ir: dict[str, Any]) -> dict[str, Any]:
    native = task_ir.get("_native_task")
    if not isinstance(native, dict):
        raise ValueError("TaskIR is missing lossless _native_task payload")
    return copy.deepcopy(native)


def assemble_task_ir_to_document(task_ir: dict[str, Any]) -> dict[str, Any]:
    return {"tasks": [assemble_task_ir_to_task_dict(task_ir)]}


def validate_task_ir(
    task_ir: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    check_assets: bool = True,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    required = {"task_name", "task_class", "task_family", "robots", "objects", "skill_steps", "_native_task"}
    missing = sorted(required - set(task_ir))
    for key in missing:
        issues.append({"code": "SCHEMA_MISSING_FIELD", "message": key})
    object_names = {
        str(item.get("name")) for item in task_ir.get("objects", []) if isinstance(item, dict)
    }
    robot_names = {
        str(item.get("name")) for item in task_ir.get("robots", []) if isinstance(item, dict)
    }
    reference_issues = []
    compatibility_issues = []
    for step in task_ir.get("skill_steps", []):
        if not isinstance(step, dict):
            continue
        for ref in step.get("object_refs", []) or []:
            if str(ref) not in object_names:
                reference_issues.append(str(ref))
        if str(step.get("robot_name")) not in robot_names:
            compatibility_issues.append(str(step.get("robot_name")))
        if str(step.get("arm")) not in {"left", "right", "base"}:
            compatibility_issues.append(str(step.get("arm")))
    root = Path(repo_root).resolve() if repo_root else None
    asset_issues = []
    if check_assets and root is not None:
        for obj in task_ir.get("objects", []):
            asset_path = obj.get("asset_path") if isinstance(obj, dict) else None
            if asset_path and not Path(str(asset_path)).is_absolute() and not (root / str(asset_path).lstrip("/")).exists():
                asset_issues.append(str(asset_path))
    issues.extend({"code": "UNKNOWN_OBJECT_REF", "message": item} for item in sorted(set(reference_issues)))
    issues.extend({"code": "ROBOT_SKILL_INCOMPATIBLE", "message": item} for item in sorted(set(compatibility_issues)))
    issues.extend({"code": "ASSET_NOT_FOUND", "message": item} for item in sorted(set(asset_issues)))
    return {
        "schema_ok": not missing,
        "references_ok": not reference_issues,
        "compatibility_ok": not compatibility_issues,
        "assets_ok": None if not check_assets else not asset_issues,
        "issues": issues,
    }

