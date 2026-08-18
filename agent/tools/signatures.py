"""Canonical semantic signature for one compiled execution variant."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


def variant_signature(document: Mapping[str, Any]) -> str:
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise ValueError("variant signature requires exactly one compiled task")
    task = tasks[0]
    metadata = task.get("metadata")
    agent_plan = metadata.get("agent_plan") if isinstance(metadata, Mapping) else None
    if not isinstance(agent_plan, Mapping):
        raise ValueError("variant signature requires metadata.agent_plan")
    initial_pose = copy.deepcopy(
        ((metadata.get("robot_position_plan") or {}).get("initial") or {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if isinstance(initial_pose, dict):
        initial_pose.pop("candidate_id", None)
    payload = {
        "schema_version": 1,
        "selected_task_id": agent_plan.get("selected_task_id"),
        "profile_id": agent_plan.get("robot_profile_id"),
        "profile_hash": agent_plan.get("robot_profile_hash"),
        "placement_family": agent_plan.get("placement_family"),
        "scene_revision": agent_plan.get("scene_revision"),
        "arm_bindings": [
            {
                "subtask_id": item.get("subtask_id"),
                "arm": item.get("arm"),
                "relation": item.get("relation"),
            }
            for item in agent_plan.get("subtasks", [])
            if isinstance(item, Mapping)
        ],
        "robot_pose": initial_pose,
        "regions": _ordered_mappings(task.get("regions"), "object"),
        "container_regions": _ordered_mappings(
            task.get("container_regions"), "object"
        ),
        "skills": copy.deepcopy(task.get("skills") or []),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _ordered_mappings(value: Any, identity_key: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"compiled task {identity_key} collection must be mappings")
    rows = [_without_provenance(copy.deepcopy(dict(item))) for item in value]
    return sorted(
        rows,
        key=lambda item: str(
            item.get(identity_key)
            or item.get("name")
            or item.get("A")
            or item.get("target")
            or ""
        ),
    )


def _without_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_provenance(item)
            for key, item in value.items()
            if key not in {"candidate_id", "planned_by", "planner_version"}
        }
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value
