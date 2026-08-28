"""Strict task contract for the Physics Schema planning world."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

PHYSICS_SCHEMA_MODE = "physics_schema"
PASSTHROUGH_MODE = "passthrough"
DIRECT_EXECUTION_MODE = "direct_execution"
PHYSICS_SCHEMA_SKILLS = {"pick", "place"}
VALIDATION_ONLY_SKILLS = {"pick_plan_probe"}
NON_MANIPULATION_SKILLS = {"navigate", "wait", "observe_hold", "scan", "track"}
_PATTERN_CHARS = frozenset("*?[]{}^$()|+")


def _skills(task_cfg: Mapping[str, Any]):
    for robot_cfg in task_cfg.get("skills", ()) or ():
        if not isinstance(robot_cfg, Mapping):
            continue
        for robot, phases in robot_cfg.items():
            if not isinstance(phases, list):
                continue
            for phase in phases:
                if not isinstance(phase, Mapping):
                    continue
                for arm, entries in phase.items():
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, Mapping):
                                yield str(robot), str(arm), entry


def _skill_name(value: Any) -> str:
    return str(value.get("name", "")).strip().lower() if isinstance(value, Mapping) else str(value).strip().lower()


def is_passthrough_skill(skill_cfg_or_name: Any) -> bool:
    return _skill_name(skill_cfg_or_name) in NON_MANIPULATION_SKILLS


def _exact_entity_name(value: Any) -> bool:
    return (
        isinstance(value, str) and bool(value) and value == value.strip()
        and value not in {".", ".."} and not value.startswith("/")
        and "/" not in value and "\\" not in value
        and not any(char.isspace() or char in _PATTERN_CHARS for char in value)
    )


def validate_planning_exclusions(values: Any | None) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("planning_exclusions must be a YAML list of exact entity names")
    result = []
    for value in values:
        if not _exact_entity_name(value):
            raise ValueError(f"planning exclusion must be one exact task entity name: {value!r}")
        if value in result:
            raise ValueError(f"duplicate planning exclusion: {value}")
        result.append(value)
    return result


def _validate_world(planning: Mapping[str, Any]) -> None:
    world = planning.get("collision_world", {})
    if not isinstance(world, Mapping):
        raise ValueError("planning.collision_world must be a mapping")
    mode = world.get("mode", PHYSICS_SCHEMA_MODE)
    if str(mode).lower() != PHYSICS_SCHEMA_MODE:
        raise ValueError("Physics schema is the only supported planning world")
    for old in ("exact_exclusions", "ignore_substring", "reference_prim_path"):
        if old in world or old in planning:
            raise ValueError(f"unsupported historical planning field: {old}")


def canonicalize_planning_config(task_cfg: Mapping[str, Any], *, config_path: Any | None = None) -> dict[str, Any]:
    del config_path
    if not isinstance(task_cfg, Mapping):
        raise TypeError("task config must be a mapping")
    result = deepcopy(dict(task_cfg))
    planning_value = result.get("planning", {})
    if not isinstance(planning_value, Mapping):
        raise ValueError("planning must be a mapping")
    planning = deepcopy(dict(planning_value))
    _validate_world(planning)
    exclusions = validate_planning_exclusions(planning.get("planning_exclusions", []))
    world = deepcopy(dict(planning.get("collision_world", {})))
    world["mode"] = PHYSICS_SCHEMA_MODE
    planning["collision_world"] = world
    planning["planning_exclusions"] = exclusions
    result["planning"] = planning
    validate_planning_contract(result, PHYSICS_SCHEMA_MODE)
    return result


def validate_planning_contract(task_cfg: Mapping[str, Any], collision_world_mode: str | None = None, *, config_path: Any | None = None) -> None:
    del config_path
    if not isinstance(task_cfg, Mapping):
        raise TypeError("task config must be a mapping")
    if collision_world_mode is not None and str(collision_world_mode).lower() != PHYSICS_SCHEMA_MODE:
        raise ValueError("Physics schema is the only supported planning world")
    planning = task_cfg.get("planning", {})
    if not isinstance(planning, Mapping):
        raise ValueError("planning must be a mapping")
    _validate_world(planning)
    validate_planning_exclusions(planning.get("planning_exclusions", []))
    for _, _, skill in _skills(task_cfg):
        name = _skill_name(skill)
        if is_passthrough_skill(skill):
            continue
        if name == "pick_plan_probe":
            if not isinstance(task_cfg.get("metadata", {}).get("workspace_probe"), Mapping):
                raise ValueError("pick_plan_probe requires metadata.workspace_probe")
            if len(skill.get("objects", ()) or ()) != 1:
                raise ValueError("pick_plan_probe requires exactly one object")
        elif name in PHYSICS_SCHEMA_SKILLS:
            expected = 1 if name == "pick" else 2
            actual = len(skill.get("objects", ()) or ())
            if actual != expected:
                raise ValueError(f"physics_schema {name} requires {expected} object identities, got {actual}")


def resolve_skill_collision_world_mode(skill_name: str, requested_mode: str | None = None, *, config_path: Any | None = None) -> str:
    del config_path
    if is_passthrough_skill(skill_name):
        return PASSTHROUGH_MODE
    if requested_mode is not None and str(requested_mode).lower() != PHYSICS_SCHEMA_MODE:
        raise ValueError("Physics schema is the only supported planning world")
    return PHYSICS_SCHEMA_MODE


def resolve_collision_world_mode(task_cfg: Mapping[str, Any], requested_mode: str | None = None) -> tuple[str, str]:
    planning = task_cfg.get("planning", {})
    world = planning.get("collision_world", {}) if isinstance(planning, Mapping) else {}
    configured = world.get("mode") if isinstance(world, Mapping) else None
    mode = requested_mode or configured or PHYSICS_SCHEMA_MODE
    if str(mode).lower() != PHYSICS_SCHEMA_MODE:
        raise ValueError("Physics schema is the only supported planning world")
    validate_planning_contract(task_cfg, mode)
    return PHYSICS_SCHEMA_MODE, "Physics schema is the only planning world"


__all__ = [
    "DIRECT_EXECUTION_MODE", "NON_MANIPULATION_SKILLS", "PASSTHROUGH_MODE",
    "PHYSICS_SCHEMA_MODE", "PHYSICS_SCHEMA_SKILLS", "VALIDATION_ONLY_SKILLS",
    "canonicalize_planning_config", "is_passthrough_skill",
    "resolve_collision_world_mode", "resolve_skill_collision_world_mode",
    "validate_planning_contract", "validate_planning_exclusions",
]
