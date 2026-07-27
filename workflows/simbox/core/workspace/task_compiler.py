"""Compile candidate-specific Probe and Pick task YAMLs without loading Isaac."""

from __future__ import annotations
import os
from pathlib import Path
import copy
from typing import Any, Mapping, Sequence, cast

from .planner import apply_candidate_to_document, apply_tabletop_candidate_to_document, dump_yaml, load_yaml


def _skill_stage(robot_name: str, left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{robot_name: [{"base": [], "left": left, "right": right}]}]


def _resolve_robot_mounting(document: Mapping[str, Any]) -> str:
    tasks = document.get("tasks") or []
    task = tasks[0] if tasks else {}
    workspace = task.get("manipulation_workspace") or {}
    workspace_robot = workspace.get("robot") or {}
    return str(workspace_robot.get("mounting", "floor"))


def _apply_candidate(document: Mapping[str, Any], input_path: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if _resolve_robot_mounting(document) == "tabletop":
        return apply_tabletop_candidate_to_document(document, input_path, candidate)
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
) -> dict[str, Any]:
    if arm is not None and arm not in {"left", "right"}:
        raise ValueError(f"unsupported Probe arm: {arm}")
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
                },
            }
        )
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
    # Probe runs are planning-only.  Camera RenderProducts and data logging are
    # deliberately removed so candidate screening does not pay five-camera
    # memory and startup costs before a pose is accepted.
    # When INTERNDATA_DEBUG_TOPDOWN is set, keep one camera + render so the
    # top-down check screenshot can be captured.
    if os.environ.get("INTERNDATA_DEBUG_TOPDOWN") == "1":
        source_cameras = task.get("cameras", [])
        task["cameras"] = source_cameras[:1] if source_cameras else []
        task["render"] = True
    else:
        task["cameras"] = []
        task["render"] = False
    task["debug_topdown_check"] = True
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
