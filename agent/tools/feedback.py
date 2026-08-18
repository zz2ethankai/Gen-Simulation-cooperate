"""Deterministic failure routing for the Agent feedback loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any


class RepairAction(str, Enum):
    KEEP = "keep"
    DIAGNOSE = "diagnose"
    MUTATE_LAYOUT = "mutate_layout"
    MUTATE_SKILL = "mutate_skill"
    NEXT_CANDIDATE = "next_candidate"
    BLOCK = "block"


@dataclass(frozen=True)
class RepairDecision:
    action: RepairAction
    failure_code: str
    deterministic: bool
    allowed_scene_mutations: tuple[str, ...] = ()
    reason: str = ""
    failing_subtask_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        return value


def failure_code_from_text(value: object, fallback: str) -> str:
    match = re.search(r"\b[A-Z][A-Z0-9_]{2,}\b", str(value))
    return match.group(0) if match else fallback


_BLOCKING = {
    "INVALID_TASK_CONFIG",
    "RELATION_NOT_ADMITTED",
    "RELATION_INSERT_NOT_ADMITTED",
    "CONTAINER_REGION_WORLD_FRAME_RANDOMIZED",
    "LAYOUT_SUBTASK_UNKNOWN",
    "UNSUPPORTED_SKILL",
    "UNSUPPORTED_CONCURRENT_MANIPULATION",
    "PHYSICS_CUROBO_WORLD_MISMATCH",
    "ATTACH_COLLISION_CONFIG_MISSING",
    "ATTACH_COLLISION_CONFIG_INVALID",
    "ATTACH_COLLISION_CONFIG_CONFLICT",
    "ATTACH_COLLISION_PRIM_NOT_FOUND",
    "ATTACH_COLLISION_PRIM_NOT_COLLIDABLE",
    "ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT",
    "ATTACH_COLLISION_PRIM_AMBIGUOUS",
    "EVENT_MISSING",
    "DATA_INTEGRITY_FAILED",
    "FAILURE_SUBTASK_UNATTRIBUTED",
}
_NEXT_CANDIDATE = {
    "NO_CUROBO_CANDIDATE",
    "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE",
    "NO_JOINT_GRASP_PLAN",
    "NO_COLLISION_FREE_PREPLACE_PLAN",
    "NO_COLLISION_FREE_PLACE_DESCENT_PLAN",
}
_LAYOUT = {
    "NO_GEOMETRY_CANDIDATE",
    "NO_COMMON_WORKSPACE_CANDIDATE",
    "PROBE_SPAWN_UNSTABLE",
    "SPAWN_COLLISION",
    "SUPPORT_ALIGNMENT_FAILED",
    "TARGET_OCCLUDED",
}
_SKILL = {
    "GRASP_CONTACT_MISSING",
    "TERMINAL_DISTANCE_EXCEEDED",
    "attached_object_dropped",
    "attached_object_translation_slip",
    "attached_object_rotation_slip",
    "PLACE_PREDICATE_FAILED",
}


def classify_failure(
    failure_code: str,
    category: str,
    *,
    failing_subtask_id: str | None = None,
) -> RepairDecision:
    code = failure_code or "UNKNOWN_FAILURE"
    if code == "NONE" or category == "success":
        return RepairDecision(
            RepairAction.KEEP,
            code,
            True,
            reason="the candidate passed strict evaluation and is kept",
            failing_subtask_id=failing_subtask_id,
        )
    if code in _BLOCKING or category in {"configuration", "asset_contract", "data_integrity"}:
        return RepairDecision(
            RepairAction.BLOCK,
            code,
            True,
            reason="configuration, asset, collision-world, and data-integrity faults require a deterministic repair",
            failing_subtask_id=failing_subtask_id,
        )
    if code in _NEXT_CANDIDATE:
        return RepairDecision(
            RepairAction.NEXT_CANDIDATE,
            code,
            True,
            reason="the current candidate failed a planning feasibility gate",
            failing_subtask_id=failing_subtask_id,
        )
    if code in _LAYOUT:
        return RepairDecision(
            RepairAction.MUTATE_LAYOUT,
            code,
            True,
            (
                "move_entity_on_support",
                "rotate_entity_on_support",
                "set_support_height",
                "set_robot_placement",
            ),
            "the failure needs a measured typed SceneMutation before a new scene revision can be compiled",
            failing_subtask_id,
        )
    if code in _SKILL:
        return RepairDecision(
            RepairAction.MUTATE_SKILL,
            code,
            True,
            reason="the scene passed placement/planning gates and one Agent-owned Skill parameter may be revised",
            failing_subtask_id=failing_subtask_id,
        )
    return RepairDecision(
        RepairAction.DIAGNOSE,
        code,
        False,
        reason="structured evidence does not yet identify one safe causal edit",
        failing_subtask_id=failing_subtask_id,
    )
