"""Compile candidate-specific Probe and Pick task YAMLs without loading Isaac."""

from __future__ import annotations
import os
from pathlib import Path
import copy
from typing import Any, Mapping, Sequence, cast

import yaml

from ..utils.camera_template import room_bounds_xy_from_arena
from .planner import apply_candidate_to_document, dump_yaml, load_yaml


REPO_ROOT = Path(__file__).resolve().parents[4]


def task_room_bounds_xy(task: Mapping[str, Any], task_path: Path) -> list[float] | None:
    arena_ref = task.get("arena_file")
    if not isinstance(arena_ref, str) or not arena_ref.strip():
        return None
    arena_path = Path(arena_ref).expanduser()
    if not arena_path.is_absolute():
        candidates = (REPO_ROOT / arena_path, task_path.parent / arena_path)
        arena_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not arena_path.is_file():
        return None
    arena = yaml.safe_load(arena_path.read_text(encoding="utf-8")) or {}
    return room_bounds_xy_from_arena(arena) if isinstance(arena, Mapping) else None


def _skill_stage(robot_name: str, left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{robot_name: [{"base": [], "left": left, "right": right}]}]


def _apply_candidate(document: Mapping[str, Any], input_path: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return apply_candidate_to_document(document, input_path, candidate)

def _sequential_arm_stages(
    robot_name: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run independent arm probes as two phases in one Isaac process."""

    return [
        {
            robot_name: [
                {"base": [], "left": left, "right": []},
                {"base": [], "left": [], "right": right},
            ]
        }
    ]


def _base_skill(name: str, target: str) -> dict[str, Any]:
    return {
        "name": name,
        "objects": [target],
        "filter_y_dir": ["forward", 90],
        "filter_z_dir": ["downward", 140],
    }


def _apply_probe_contract(
    task: dict[str, Any],
    target: str,
    planning: Mapping[str, Any] | None,
    attach_prim_path_children: Sequence[str] | None,
) -> list[str]:
    """Make a planning Probe use the same world and attach proxy as execution."""

    if planning is not None:
        task["planning"] = copy.deepcopy(dict(planning))
    paths = [str(value).strip() for value in (attach_prim_path_children or [])]
    if any(not value for value in paths) or len(paths) != len(set(paths)):
        raise ValueError("attach_prim_path_children must be unique non-empty paths")
    if paths:
        target_object = next(
            (value for value in task.get("objects", []) if value.get("name") == target),
            None,
        )
        if target_object is None:
            raise ValueError(f"probe target object is missing: {target}")
        target_object["attach_prim_path_children"] = paths
        target_object.pop("attach_prim_path_child", None)
    return paths


def _configure_probe_rendering(
    task: dict[str, Any], diagnostic_capture: Mapping[str, Any] | None = None
) -> None:
    capture = dict(diagnostic_capture or {})
    if capture:
        task["cameras"] = []
        task["render"] = bool(capture.get("overview"))
        task["debug_topdown_check"] = False
        if capture.get("trajectory"):
            trajectory = task.setdefault("visualization", {}).setdefault(
                "curobo_trajectory", {}
            )
            trajectory["enabled"] = True
            trajectory["export_usd"] = True
            trajectory.setdefault("show_ee_path", True)
            trajectory.setdefault("show_robot_spheres", False)
        return
    if os.environ.get("INTERNDATA_DEBUG_TOPDOWN") == "1":
        source_cameras = task.get("cameras", [])
        task["cameras"] = source_cameras[:1] if source_cameras else []
        task["render"] = True
    else:
        task["cameras"] = []
        task["render"] = False
    task["debug_topdown_check"] = True


def _source_pick_place_pair(
    task: Mapping[str, Any],
    robot_name: str,
    arm: str,
    target: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for skill_stage in task.get("skills", []):
        robot_phases = skill_stage.get(robot_name, [])
        for phase in robot_phases:
            arm_skills = phase.get(arm, [])
            for index in range(len(arm_skills) - 1):
                pick_skill = arm_skills[index]
                place_skill = arm_skills[index + 1]
                if (
                    str(pick_skill.get("name", "")).lower() == "pick"
                    and list(pick_skill.get("objects", [])) == [target]
                    and str(place_skill.get("name", "")).lower() == "place"
                    and len(place_skill.get("objects", [])) == 2
                    and place_skill["objects"][0] == target
                ):
                    matches.append((copy.deepcopy(pick_skill), copy.deepcopy(place_skill)))
    if len(matches) != 1:
        raise ValueError(
            "Pick+Place planning probe requires exactly one adjacent source pair "
            f"for robot={robot_name!r} arm={arm!r} target={target!r}; found {len(matches)}"
        )
    return matches[0]


def compile_probe_task(
    source_task: Path,
    candidate: Mapping[str, Any],
    target: str,
    output_path: Path,
    result_dir: Path,
    *,
    arm: str | None = None,
    planning: Mapping[str, Any] | None = None,
    attach_prim_path_children: Sequence[str] | None = None,
    expected_target_world_xyz: Sequence[float] | None = None,
    spawn_settle: Mapping[str, Any] | None = None,
    diagnostic_disable_curobo_obstacle_paths: Sequence[str] | None = None,
    diagnostic_disable_physics_and_curobo_obstacle_paths: Sequence[str] | None = None,
    diagnostic_disable_collision_entities: Sequence[str] | None = None,
    diagnostic_collision_world: str = "full",
    diagnostic_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if arm is not None and arm not in {"left", "right"}:
        raise ValueError(f"unsupported Probe arm: {arm}")
    required_spawn_settle = {
        "max_object_linear_speed_m_s",
        "max_object_angular_speed_rad_s",
        "max_robot_joint_speed_rad_s",
        "max_unexpected_contact_n",
        "target_support",
    }
    provided_spawn_settle = dict(spawn_settle or {})
    missing_spawn_settle = sorted(required_spawn_settle - provided_spawn_settle.keys())
    if missing_spawn_settle:
        raise ValueError(
            "spawn_settle is missing required measurements: "
            + ", ".join(missing_spawn_settle)
        )
    if not isinstance(provided_spawn_settle["target_support"], str) or not str(
        provided_spawn_settle["target_support"]
    ).strip():
        raise ValueError("spawn_settle.target_support must be a non-empty fixture name")
    for key in required_spawn_settle - {"target_support"}:
        value = float(provided_spawn_settle[key])
        if value < 0.0:
            raise ValueError(f"spawn_settle.{key} must be non-negative")
    source_doc = load_yaml(source_task)
    document = _apply_candidate(source_doc, source_task, candidate)
    task = document["tasks"][0]
    robot_name = str(task["robots"][0]["name"])
    candidate_id = str(candidate["candidate_id"])
    attach_paths = _apply_probe_contract(
        task,
        target,
        planning,
        attach_prim_path_children,
    )
    raw_diagnostic_paths = diagnostic_disable_curobo_obstacle_paths or []
    if isinstance(raw_diagnostic_paths, (str, bytes)):
        raise ValueError(
            "diagnostic CuRobo obstacle paths must be a list of exact Prim paths"
        )
    diagnostic_paths = [str(value).strip() for value in raw_diagnostic_paths]
    if any(not value or not value.startswith("/") for value in diagnostic_paths) or len(
        diagnostic_paths
    ) != len(set(diagnostic_paths)):
        raise ValueError(
            "diagnostic CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    raw_dual_paths = diagnostic_disable_physics_and_curobo_obstacle_paths or []
    if isinstance(raw_dual_paths, (str, bytes)):
        raise ValueError("diagnostic Physics+CuRobo obstacle paths must be a list")
    dual_paths = [str(value).strip() for value in raw_dual_paths]
    if any(not value or not value.startswith("/") for value in dual_paths) or len(
        dual_paths
    ) != len(set(dual_paths)):
        raise ValueError(
            "diagnostic Physics+CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    raw_entities = diagnostic_disable_collision_entities or []
    if isinstance(raw_entities, (str, bytes)):
        raise ValueError("diagnostic collision entities must be a list")
    collision_entities = [str(value).strip() for value in raw_entities]
    if any(not value for value in collision_entities) or len(collision_entities) != len(
        set(collision_entities)
    ):
        raise ValueError(
            "diagnostic collision entity names must be unique and non-empty"
        )
    if diagnostic_collision_world not in {"full", "target-only", "empty"}:
        raise ValueError(
            "diagnostic_collision_world must be one of: full, target-only, empty"
        )
    if (
        diagnostic_paths or dual_paths or collision_entities
    ) and diagnostic_collision_world != "full":
        raise ValueError(
            "exact/entity diagnostic collision isolation cannot be combined with "
            "target-only or empty-world diagnostics"
        )
    if diagnostic_paths and (dual_paths or collision_entities):
        raise ValueError(
            "CuRobo-only and Physics+CuRobo diagnostic isolation modes are mutually exclusive"
        )
    capture = copy.deepcopy(dict(diagnostic_capture or {}))
    if capture:
        if arm is None:
            raise ValueError("diagnostic capture requires one explicit probe arm")
        if not capture.get("overview") and not capture.get("trajectory"):
            raise ValueError("diagnostic capture must enable overview or trajectory")
        output_dir = str(capture.get("output_dir") or "").strip()
        if not output_dir:
            raise ValueError("diagnostic capture requires output_dir")
        capture["output_dir"] = str(Path(output_dir).expanduser().resolve())
        camera = capture.get("camera") or {}
        if camera.get("template"):
            camera.setdefault("room_bounds_xy", task_room_bounds_xy(task, source_task))
            capture["camera"] = camera
    arm_skills: dict[str, list[dict[str, Any]]] = {}
    selected_arms = (arm,) if arm is not None else ("left", "right")
    for selected_arm in selected_arms:
        skill = _base_skill("pick_plan_probe", target)
        skill.update(
            {
                "candidate_id": candidate_id,
                "result_path": str(
                    (result_dir / f"{candidate_id}.{selected_arm}.json").resolve()
                ),
                "debug": True,
                "spawn_expectation": {
                    "robot_world_xy": [float(value) for value in candidate["world_xy"]],
                    "robot_yaw_deg": float(candidate["yaw_deg"]),
                    "target_world_xyz": (
                        [float(value) for value in expected_target_world_xyz]
                        if expected_target_world_xyz is not None
                        else None
                    ),
                    "robot_xy_tolerance_m": 0.05,
                    "robot_yaw_tolerance_deg": 5.0,
                    "target_xy_tolerance_m": 0.25,
                    **provided_spawn_settle,
                },
            }
        )
        if diagnostic_paths:
            skill["diagnostic_disable_curobo_obstacle_paths"] = diagnostic_paths
        if dual_paths:
            skill["diagnostic_disable_physics_and_curobo_obstacle_paths"] = dual_paths
        if collision_entities:
            skill["diagnostic_disable_collision_entities"] = collision_entities
        if diagnostic_collision_world == "target-only":
            skill["diagnostic_target_only_world"] = True
        elif diagnostic_collision_world == "empty":
            skill["diagnostic_empty_world"] = True
        if capture:
            skill["diagnostic_capture"] = capture
        arm_skills[selected_arm] = [skill]
    if arm is None:
        task["skills"] = _sequential_arm_stages(
            robot_name,
            arm_skills["left"],
            arm_skills["right"],
        )
        execution_mode = "dual_arm_sequential_probe"
    else:
        task["skills"] = _skill_stage(
            robot_name,
            arm_skills.get("left", []),
            arm_skills.get("right", []),
        )
        execution_mode = "required_arm_probe"
    _configure_probe_rendering(task, capture)
    task.setdefault("metadata", {})["workspace_probe"] = {
        "candidate_id": candidate_id,
        "target": target,
        "result_dir": str(result_dir.resolve()),
        "execution_mode": execution_mode,
        "required_arm": arm,
        "attach_prim_path_children": attach_paths,
        "collision_world_mode": str(
            ((task.get("planning") or {}).get("collision_world") or {}).get(
                "mode", ""
            )
        ),
        "spawn_stability_gate": True,
        "diagnostic_disable_curobo_obstacle_paths": diagnostic_paths,
        "diagnostic_disable_physics_and_curobo_obstacle_paths": dual_paths,
        "diagnostic_disable_collision_entities": collision_entities,
        "diagnostic_resolved_collision_entities": "runtime_collision_scene_manager",
        "diagnostic_collision_world": diagnostic_collision_world,
        "diagnostic_capture": capture,
    }
    dump_yaml(document, output_path)
    return document


def compile_pick_place_probe_task(
    source_task: Path,
    candidate: Mapping[str, Any],
    target: str,
    arm: str,
    output_path: Path,
    result_path: Path,
    *,
    planning: Mapping[str, Any] | None = None,
    attach_prim_path_children: Sequence[str] | None = None,
) -> dict[str, Any]:
    if arm not in {"left", "right"}:
        raise ValueError(f"unsupported Pick+Place probe arm: {arm}")
    source_doc = load_yaml(source_task)
    document = _apply_candidate(source_doc, source_task, candidate)
    task = document["tasks"][0]
    robot_name = str(task["robots"][0]["name"])
    pick_skill, place_skill = _source_pick_place_pair(task, robot_name, arm, target)
    attach_paths = _apply_probe_contract(
        task,
        target,
        planning,
        attach_prim_path_children,
    )
    target_object = next(
        (value for value in task.get("objects", []) if value.get("name") == target),
        None,
    )
    if target_object is None:
        raise ValueError(f"probe target object is missing: {target}")
    if not attach_paths:
        configured_paths = target_object.get("attach_prim_path_children")
        if configured_paths is None and target_object.get("attach_prim_path_child"):
            configured_paths = [target_object["attach_prim_path_child"]]
        attach_paths = [str(value).strip() for value in configured_paths or []]
    if (
        not attach_paths
        or any(not value for value in attach_paths)
        or len(attach_paths) != len(set(attach_paths))
    ):
        raise ValueError("Pick+Place planning probe requires exact attach Prim paths")
    collision_world_mode = str(
        ((task.get("planning") or {}).get("collision_world") or {}).get("mode", "")
    )
    if collision_world_mode != "physics_schema":
        raise ValueError(
            "Pick+Place planning probe requires planning.collision_world.mode=physics_schema"
        )
    candidate_id = str(candidate["candidate_id"])
    place_skill["name"] = "place_plan_probe"
    place_skill["candidate_id"] = candidate_id
    place_skill["result_path"] = str(result_path.resolve())
    task["skills"] = _skill_stage(
        robot_name,
        [pick_skill, place_skill] if arm == "left" else [],
        [pick_skill, place_skill] if arm == "right" else [],
    )
    _configure_probe_rendering(task)
    task.setdefault("metadata", {})["workspace_probe"] = {
        "candidate_id": candidate_id,
        "target": target,
        "support": str(place_skill["objects"][1]),
        "result_path": str(result_path.resolve()),
        "execution_mode": "pick_execution_place_planning",
        "required_arm": arm,
        "attach_prim_path_children": attach_paths,
        "collision_world_mode": collision_world_mode,
        "requires_verified_pick_attachment": True,
    }
    dump_yaml(document, output_path)
    return document


def compile_pick_task(
    source_task: Path,
    candidate: Mapping[str, Any],
    target: str,
    arm: str,
    output_path: Path,
) -> dict[str, Any]:
    if arm not in {"left", "right"}:
        raise ValueError(f"unsupported Pick arm: {arm}")
    source_doc = load_yaml(source_task)
    document = _apply_candidate(source_doc, source_task, candidate)
    task = document["tasks"][0]
    robot_name = str(task["robots"][0]["name"])
    skill = _base_skill("pick", target)
    task["skills"] = _skill_stage(robot_name, [skill] if arm == "left" else [], [skill] if arm == "right" else [])
    task.setdefault("metadata", {})["workspace_pick_validation"] = {
        "candidate_id": str(candidate["candidate_id"]),
        "target": target,
        "arm": arm,
    }
    dump_yaml(document, output_path)
    return document


def compile_existing_pose_probe_task(
    source_task: Path,
    output_path: Path,
    result_dir: Path,
    candidate_id: str = "existing_pose",
) -> dict[str, Any]:
    """Convert existing Pick entries to Probe while preserving the delivered pose."""
    document = copy.deepcopy(load_yaml(source_task))
    task = document["tasks"][0]
    converted = 0
    for robot_entry in task.get("skills", []):
        for stages in robot_entry.values():
            for stage in stages:
                for arm, skills in stage.items():
                    if arm not in {"left", "right"}:
                        continue
                    for skill in skills:
                        if str(skill.get("name", "")).lower() != "pick":
                            continue
                        skill["name"] = "pick_plan_probe"
                        skill["candidate_id"] = candidate_id
                        skill["result_path"] = str((result_dir / f"{candidate_id}.{arm}.json").resolve())
                        skill["debug"] = True
                        converted += 1
    if converted == 0:
        raise ValueError(f"task has no Pick skill to probe: {source_task}")
    task.setdefault("metadata", {})["workspace_probe"] = {
        "candidate_id": candidate_id,
        "result_dir": str(result_dir.resolve()),
        "source": "existing_pose_control",
    }
    dump_yaml(document, output_path)
    return cast(dict[str, Any], document)
