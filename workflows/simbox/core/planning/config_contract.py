"""Simulator-independent validation for Physics-schema manipulation configs."""

from __future__ import annotations

from typing import Any


PHYSICS_SCHEMA_SKILLS = {"pick", "place"}
VALIDATION_ONLY_SKILLS = {"pick_plan_probe"}
PHYSICS_SCHEMA_ONLY_SKILLS = {"pick", "pick_plan_probe"}
ATTACHED_PHYSICS_SCHEMA_SKILL_MODES = {
    ("heuristic__skill", "home"),
}
NON_MANIPULATION_SKILLS = {
    "navigate",
    "observe_hold",
}

PHYSICS_SCHEMA_MODE = "physics_schema"
LEGACY_STAGE_SCAN_MODE = "legacy_stage_scan"
HYBRID_MODE = "hybrid"
PASSTHROUGH_MODE = "passthrough"


def _arm_skill_names(task_cfg: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for cfg_skill_dict in task_cfg.get("skills", []):
        for robot_skill_list in cfg_skill_dict.values():
            for lr_skill_dict in robot_skill_list:
                for arm in ("left", "right"):
                    names.update(
                        str(skill_cfg.get("name", "")).lower()
                        for skill_cfg in lr_skill_dict.get(arm, [])
                    )
    return names


def resolve_skill_collision_world_mode(
    skill_name: str, requested_mode: str | None
) -> str:
    """Resolve one Skill without weakening explicit task-level requests."""

    name = str(skill_name).strip().lower()
    requested = "auto" if requested_mode is None else str(requested_mode).strip().lower()
    if name in NON_MANIPULATION_SKILLS:
        return PASSTHROUGH_MODE
    if requested == LEGACY_STAGE_SCAN_MODE:
        if name in PHYSICS_SCHEMA_ONLY_SKILLS:
            replacement = "legacy_pick" if name == "pick" else "physics_schema"
            raise ValueError(
                f"Skill {name!r} is Physics-schema-only and cannot run in "
                f"{LEGACY_STAGE_SCAN_MODE}; use {replacement!r}"
            )
        return LEGACY_STAGE_SCAN_MODE
    if name in PHYSICS_SCHEMA_SKILLS | VALIDATION_ONLY_SKILLS:
        return PHYSICS_SCHEMA_MODE
    return LEGACY_STAGE_SCAN_MODE


def resolve_runtime_skill_collision_world_mode(
    skill_cfg: Any,
    requested_mode: str | None,
    *,
    attached_object: bool,
) -> str:
    """Resolve conditional Physics adapters that depend on object ownership.

    Static task resolution keeps these Skills in the legacy fallback set.  At
    runtime, an attached object cannot cross into that world safely, so only
    explicitly adapted Skill modes stay in the Physics world until detach.
    """

    name = str(
        skill_cfg.get("name", "") if hasattr(skill_cfg, "get") else ""
    ).strip().lower()
    mode = str(
        skill_cfg.get("mode", "") if hasattr(skill_cfg, "get") else ""
    ).strip().lower()
    resolved = resolve_skill_collision_world_mode(name, requested_mode)
    requested = "auto" if requested_mode is None else str(requested_mode).strip().lower()
    if (
        attached_object
        and requested != LEGACY_STAGE_SCAN_MODE
        and (name, mode) in ATTACHED_PHYSICS_SCHEMA_SKILL_MODES
    ):
        return PHYSICS_SCHEMA_MODE
    return resolved


def task_uses_physics_schema(collision_world_mode: str) -> bool:
    return str(collision_world_mode) in {PHYSICS_SCHEMA_MODE, HYBRID_MODE}


def resolve_skill_test_mode(skill_cfg: Any, collision_world_mode: str) -> str:
    """Resolve legacy test_mode without weakening Physics-schema planning.

    ``test_mode: ik`` is retained only for explicit legacy-stage-scan Skills.
    Migrated Pick and Place Skills always require forward planning so that their
    transit paths are collision-validated.
    """

    if str(collision_world_mode) in {PHYSICS_SCHEMA_MODE, HYBRID_MODE}:
        return "forward"
    return str(skill_cfg.get("test_mode", "forward"))


def resolve_collision_world_mode(
    task_cfg: dict[str, Any], requested_mode: str | None
) -> tuple[str, str]:
    """Resolve auto mode without weakening explicit Physics-schema validation."""

    requested = "auto" if requested_mode is None else str(requested_mode).strip().lower()
    if requested == "auto":
        skill_names = _arm_skill_names(task_cfg)
        if not skill_names.intersection(PHYSICS_SCHEMA_SKILLS | VALIDATION_ONLY_SKILLS):
            return LEGACY_STAGE_SCAN_MODE, "task has no Physics-schema manipulation skills"
        legacy_skills = sorted(
            name
            for name in skill_names
            if resolve_skill_collision_world_mode(name, requested)
            == LEGACY_STAGE_SCAN_MODE
        )
        mode = HYBRID_MODE if legacy_skills else PHYSICS_SCHEMA_MODE
        validate_planning_contract(task_cfg, mode)
        if legacy_skills:
            return mode, (
                "Physics-schema Skills enabled with per-Skill legacy fallback: "
                + ", ".join(legacy_skills)
            )
        return mode, "all active manipulation skills support physics_schema"

    validate_planning_contract(task_cfg, requested)
    return requested, "explicit configuration"


def validate_planning_contract(task_cfg: dict[str, Any], collision_world_mode: str) -> None:
    """Reject configs that would silently bypass the stateful collision path."""

    if collision_world_mode == LEGACY_STAGE_SCAN_MODE:
        incompatible = sorted(
            _arm_skill_names(task_cfg).intersection(PHYSICS_SCHEMA_ONLY_SKILLS)
        )
        if incompatible:
            raise ValueError(
                "Physics-schema-only Skills cannot run in "
                f"{LEGACY_STAGE_SCAN_MODE}: {', '.join(incompatible)}; "
                "use name='legacy_pick' for the legacy Pick implementation"
            )
        return
    if collision_world_mode not in {PHYSICS_SCHEMA_MODE, HYBRID_MODE}:
        raise ValueError(
            f"unsupported planning.collision_world.mode: {collision_world_mode!r}"
        )

    for stage_index, cfg_skill_dict in enumerate(task_cfg.get("skills", [])):
        for robot_name, robot_skill_list in cfg_skill_dict.items():
            for phase_index, lr_skill_dict in enumerate(robot_skill_list):
                active_arms = []
                for arm in ("left", "right"):
                    arm_skills = lr_skill_dict.get(arm, [])
                    if any(
                        str(skill_cfg.get("name", "")).lower()
                        not in NON_MANIPULATION_SKILLS
                        for skill_cfg in arm_skills
                    ):
                        active_arms.append(arm)
                if len(active_arms) > 1:
                    raise ValueError(
                        "UNSUPPORTED_CONCURRENT_MANIPULATION: "
                        f"stage={stage_index} robot={robot_name} "
                        f"phase={phase_index} arms={active_arms}"
                    )
                for arm in ("left", "right"):
                    for skill_cfg in lr_skill_dict.get(arm, []):
                        skill_name = str(skill_cfg.get("name", "")).lower()
                        if skill_name in NON_MANIPULATION_SKILLS:
                            continue
                        if skill_name in VALIDATION_ONLY_SKILLS:
                            if not isinstance(task_cfg.get("metadata", {}).get("workspace_probe"), dict):
                                raise ValueError(
                                    f"validation-only Skill {skill_name!r} requires metadata.workspace_probe"
                                )
                            if len(skill_cfg.get("objects", [])) != 1:
                                raise ValueError(
                                    f"validation-only Skill {skill_name!r} requires exactly 1 object identity"
                                )
                            continue
                        if skill_name not in PHYSICS_SCHEMA_SKILLS:
                            if collision_world_mode == HYBRID_MODE:
                                continue
                            raise ValueError(
                                f"Skill {skill_name!r} is not migrated to physics_schema; "
                                "set planning.collision_world.mode=legacy_stage_scan explicitly"
                            )
                        object_count = len(skill_cfg.get("objects", []))
                        expected_count = 1 if skill_name == "pick" else 2
                        if object_count != expected_count:
                            raise ValueError(
                                f"physics_schema {skill_name} requires exactly "
                                f"{expected_count} object identities, got {object_count}"
                            )
