"""Simulator-independent validation for Physics-schema manipulation configs."""

from __future__ import annotations

from typing import Any


PHYSICS_SCHEMA_SKILLS = {"pick", "place"}
VALIDATION_ONLY_SKILL_OBJECT_COUNTS = {
    "pick_plan_probe": 1,
    "place_plan_probe": 2,
}
NON_MANIPULATION_SKILLS = {"observe_hold"}
SPAWN_SETTLE_FIELDS = {
    "max_object_linear_speed_m_s",
    "max_object_angular_speed_rad_s",
    "max_robot_joint_speed_rad_s",
    "max_unexpected_contact_n",
    "target_support",
}


def validate_planning_contract(task_cfg: dict[str, Any], collision_world_mode: str) -> None:
    """Reject configs that would silently bypass the stateful collision path."""

    if collision_world_mode == "legacy_stage_scan":
        return
    if collision_world_mode != "physics_schema":
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
                        if skill_name in VALIDATION_ONLY_SKILL_OBJECT_COUNTS:
                            if not isinstance(task_cfg.get("metadata", {}).get("workspace_probe"), dict):
                                raise ValueError(
                                    f"validation-only Skill {skill_name!r} requires metadata.workspace_probe"
                                )
                            expected_count = VALIDATION_ONLY_SKILL_OBJECT_COUNTS[skill_name]
                            if len(skill_cfg.get("objects", [])) != expected_count:
                                raise ValueError(
                                    f"validation-only Skill {skill_name!r} requires exactly "
                                    f"{expected_count} object identities"
                                )
                            if (
                                skill_name == "place_plan_probe"
                                and str(skill_cfg.get("test_mode", "forward")) != "forward"
                            ):
                                raise ValueError(
                                    "place_plan_probe requires test_mode=forward because "
                                    "IK-only checks do not validate a collision-free path"
                                )
                            if skill_name == "pick_plan_probe":
                                expectation = skill_cfg.get("spawn_expectation")
                                missing = (
                                    sorted(SPAWN_SETTLE_FIELDS - expectation.keys())
                                    if isinstance(expectation, dict)
                                    else sorted(SPAWN_SETTLE_FIELDS)
                                )
                                if missing:
                                    raise ValueError(
                                        "pick_plan_probe spawn_expectation is missing: "
                                        + ", ".join(missing)
                                    )
                            continue
                        if skill_name not in PHYSICS_SCHEMA_SKILLS:
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
                        if str(skill_cfg.get("test_mode", "forward")) != "forward":
                            raise ValueError(
                                f"physics_schema {skill_name} requires test_mode=forward "
                                "because IK-only checks do not validate a collision-free path"
                            )
