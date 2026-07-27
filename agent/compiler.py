"""Deterministically compile TaskPlan objects into SimBox task YAML."""

from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import PlaceRelation, SceneCapabilityManifest, StageExecutionMode, TaskPlan
from .settings import (
    DEFAULT_CONFIG_PATH,
    load_agent_settings,
    merge_mappings,
    resolve_data_generation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CompileError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list) or not value["tasks"]:
        raise CompileError(f"invalid SimBox task document: {path}")
    return value


def _container_region(manifest: SceneCapabilityManifest, target: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in manifest.container_regions
            if str(item.get("object") or item.get("target")) == target
            and bool(item.get("can_receive_objects", True))
        ),
        None,
    )


def _skill_defaults(settings: Mapping[str, Any], name: str) -> dict[str, Any]:
    values = settings.get("skill_defaults", {})
    result = values.get(name) if isinstance(values, Mapping) else None
    if not isinstance(result, Mapping):
        raise CompileError(f"agent config has no skill_defaults.{name} mapping")
    return copy.deepcopy(dict(result))


def _compile_skill(
    skill,
    relation: str,
    manifest: SceneCapabilityManifest,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    name = skill.name.lower()
    if name == "pick":
        value = _skill_defaults(settings, "pick")
    elif name == "place":
        value = _skill_defaults(settings, "place")
        relation_defaults = settings.get("skill_defaults", {}).get("relation", {})
        if not isinstance(relation_defaults, Mapping):
            raise CompileError("agent config skill_defaults.relation must be a mapping")
        value.update(copy.deepcopy(dict(relation_defaults.get(relation, {}))))
        if relation == PlaceRelation.INSIDE.value:
            target = skill.objects[1]
            region = _container_region(manifest, target)
            if region is None:
                raise CompileError(f"inside placement requires a declared container region: {target}")
    else:
        raise CompileError(f"unsupported physics-schema Skill: {name}")
    value.update(copy.deepcopy(skill.params))
    value["name"] = name
    value["objects"] = list(skill.objects)
    value["test_mode"] = "forward"
    if "ignore_substring" in value:
        raise CompileError("Agent-generated physics-schema Skills may not use ignore_substring")
    return value


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _resolve_arm(skill_arm: str, subtask_arm: str) -> str:
    arm = skill_arm if skill_arm != "auto" else subtask_arm
    if arm not in {"left", "right"}:
        raise CompileError("workspace planning must select left or right before final config compilation")
    return arm


def _compile_stage_phases(
    stage,
    subtask_arm: str,
    relation: str,
    manifest: SceneCapabilityManifest,
    settings: Mapping[str, Any],
) -> list[dict[str, list[dict[str, Any]]]]:
    """Translate the explicit planning mode to SimBox's phase/arm nesting."""

    mode = _enum_text(stage.execution_mode)
    ordered: list[tuple[str, dict[str, Any]]] = [
        (
            _resolve_arm(skill.arm, subtask_arm),
            _compile_skill(skill, relation, manifest, settings),
        )
        for skill in stage.skills
    ]

    def phase(left=None, right=None):
        return {"base": [], "left": left or [], "right": right or []}

    if mode in {
        StageExecutionMode.SINGLE_ARM_SINGLE_SKILL.value,
        StageExecutionMode.SINGLE_ARM_SEQUENTIAL.value,
    }:
        arms = {arm for arm, _ in ordered}
        if len(arms) != 1:
            raise CompileError(f"stage {stage.stage_id} {mode} must use exactly one arm")
        arm = next(iter(arms))
        skills = [skill for _, skill in ordered]
        return [phase(left=skills) if arm == "left" else phase(right=skills)]

    if mode == StageExecutionMode.DUAL_ARM_SEQUENTIAL.value:
        arms = [arm for arm, _ in ordered]
        if set(arms) != {"left", "right"}:
            raise CompileError(
                f"stage {stage.stage_id} dual_arm_sequential requires both arms"
            )
        groups: list[tuple[str, list[dict[str, Any]]]] = []
        for arm, skill in ordered:
            if not groups or groups[-1][0] != arm:
                groups.append((arm, []))
            groups[-1][1].append(skill)
        if len(groups) != 2:
            raise CompileError(
                f"stage {stage.stage_id} dual_arm_sequential must contain two arm blocks"
            )
        return [
            phase(left=skills) if arm == "left" else phase(right=skills)
            for arm, skills in groups
        ]

    if mode == StageExecutionMode.DUAL_ARM_SIMULTANEOUS.value:
        # SimBox's YAML can represent this mode, but the stateful
        # physics_schema collision manager has not yet been validated for two
        # concurrent manipulation owners.  Keep an explicit capability slot
        # instead of silently compiling an unsafe task.
        raise CompileError(
            "dual_arm_simultaneous is represented by TaskPlan but is not enabled "
            "for physics_schema execution"
        )
    raise CompileError(f"unsupported stage execution_mode: {mode}")


def apply_workspace_candidate(document: dict[str, Any], source_path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        from workflows.simbox.core.workspace.planner import apply_candidate_to_document
    except ImportError as exc:
        raise CompileError(f"workspace compiler is unavailable: {exc}") from exc
    return apply_candidate_to_document(document, source_path, candidate)

def apply_tabletop_workspace_candidate(document: dict[str, Any], source_path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        from workflows.simbox.core.workspace.planner import apply_tabletop_candidate_to_document
    except ImportError as exc:
        raise CompileError(f"tabletop workspace compiler is unavailable: {exc}") from exc
    return apply_tabletop_candidate_to_document(document, source_path, candidate)

def compile_task_config(
    plan: TaskPlan,
    manifest: SceneCapabilityManifest,
    output_path: Path,
    workspace_candidate: dict[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Path:
    effective_settings = dict(settings or load_agent_settings())
    source_path = Path(plan.source_task).resolve()
    document = copy.deepcopy(load_yaml(source_path))
    if workspace_candidate is not None:
        if manifest.robot_mounting == "tabletop":
            document = apply_tabletop_workspace_candidate(document, source_path, workspace_candidate)
        else:
            document = apply_workspace_candidate(document, source_path, workspace_candidate)
    task = document["tasks"][0]
    robot_name = str(task.get("robots", [{}])[0].get("name", ""))
    if robot_name != plan.robot.robot_type:
        raise CompileError(f"plan robot {plan.robot.robot_type} does not match source task robot {robot_name}")

    stages: list[dict[str, Any]] = []
    for subtask in plan.subtasks:
        for stage in subtask.stages:
            phases = _compile_stage_phases(
                stage,
                subtask.arm,
                _enum_text(subtask.relation),
                manifest,
                effective_settings,
            )
            stages.append({robot_name: phases})
    task["skills"] = stages
    configured_planning = effective_settings.get("planning", {})
    if not isinstance(configured_planning, Mapping):
        raise CompileError("agent config planning must be a mapping")
    source_planning = task.get("planning", {})
    if source_planning is not None and not isinstance(source_planning, Mapping):
        raise CompileError("source task planning must be a mapping")
    planning = merge_mappings(configured_planning, source_planning)
    # The collision-world contract is compiler-owned and cannot be weakened by
    # a source task or an LLM-produced plan.
    planning["collision_world"] = copy.deepcopy(dict(configured_planning.get("collision_world", {})))
    task["planning"] = planning


    metadata = task.setdefault("metadata", {})
    workspace_id = str(workspace_candidate.get("candidate_id")) if workspace_candidate else None
    metadata["agent_plan"] = {
        "selected_task_id": plan.selected_task_id,
        "prompt": plan.prompt,
        "workspace_candidate_id": workspace_id,
        "data_generation": resolve_data_generation(
            plan.task_request.data_generation,
            effective_settings,
        ),
        "subtasks": [
            {"subtask_id": item.subtask_id, "center_object": item.manipulated_object, "arm": item.arm}
            for item in plan.subtasks
        ],
        "execution_modes": [
            _enum_text(stage.execution_mode)
            for subtask in plan.subtasks
            for stage in subtask.stages
        ],
    }
    if workspace_candidate is not None:
        metadata["robot_position_plan"] = {
            "selection_mode": "single" if len(plan.subtasks) == 1 else "common",
            "initial": {
                "targets": [item.manipulated_object for item in plan.subtasks],
                "world_xy": list(workspace_candidate["world_xy"]),
                "yaw_deg": float(workspace_candidate["yaw_deg"]),
                "candidate_id": workspace_id,
            },
            "subtasks": [
                {
                    "subtask_id": item.subtask_id,
                    "target": item.manipulated_object,
                    "arm": item.arm,
                }
                for item in plan.subtasks
            ],
        }

    from workflows.simbox.core.planning.config_contract import validate_planning_contract

    collision_world_mode = str((planning.get("collision_world") or {}).get("mode", ""))
    validate_planning_contract(task, collision_world_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def generate_workspace_manifest(
    source_task: Path,
    target: str,
    arm: str,
    output_dir: Path,
    robot_mounting: str = "floor",
) -> Path:
    if arm not in {"left", "right"}:
        raise CompileError(f"workspace generation requires a fixed left/right arm, got {arm!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if robot_mounting == "tabletop":
        script = "scripts/simbox/plan_tabletop_workspace.py"
    else:
        script = "scripts/simbox/plan_workspace_layout.py"

    command = [
        "python",
        script,
        "--task",
        str(source_task),
        "--target",
        target,
        "--arm",
        arm,
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (output_dir / "workspace_plan_stderr.log").write_text(completed.stderr or "", encoding="utf-8"
)
    (output_dir / "workspace_plan_stdout.log").write_text(completed.stdout or "", encoding="utf-8"
)
    manifest_path = output_dir / "candidates.json"
    if completed.returncode != 0 or not manifest_path.is_file():
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            feasible = sum(c.get("geometry_feasible") for c in manifest.get("geometry_candidates", []))
            if feasible == 0:
                _write_workspace_diagnostics(manifest, target, arm, output_dir)
        detail = (completed.stderr or "").strip().split("\n")[-1] if completed.stderr else ""
        raise CompileError(
            f"workspace geometry planning failed (exit {completed.returncode}): {detail}\n"
            f"expected: {manifest_path}\n"
            f"stderr log: {output_dir / 'workspace_plan_stderr.log'}"
        )
    
    return manifest_path

def _write_workspace_diagnostics(
    manifest: dict[str, Any],
    target: str,
    arm: str,
    output_dir: Path,
) -> None:
    from collections import Counter

    candidates = manifest.get("geometry_candidates", [])
    rejection_codes = Counter(
        c.get("rejection_code", "UNKNOWN") for c in candidates
    )
    collision_fixtures = Counter(
        c.get("obstacle") for c in candidates if c.get("obstacle")
    )
    target_info = manifest.get("target", {})
    sampling = manifest.get("sampling", {})

    sample_candidates = []
    for c in candidates[:3]:
        sample_candidates.append({
            "candidate_id": c.get("candidate_id"),
            "world_xy": c.get("world_xy"),
            "radius_m": c.get("radius_m"),
            "yaw_deg": c.get("yaw_deg"),
            "rejection_code": c.get("rejection_code"),
            "obstacle": c.get("obstacle"),
        })

    diagnostics = {
        "target": {"name": target, "world_xyz": target_info.get("world_xyz")},
        "arm": arm,
        "sampling": sampling,
        "total_candidates": len(candidates),
        "feasible": 0,
        "rejection_code_counts": dict(rejection_codes.most_common()),
        "collision_fixture_counts": dict(collision_fixtures.most_common()),
        "sample_candidates": sample_candidates,
    }
    (output_dir / "workspace_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_workspace_manifest(
    manifest_path: Path,
    gpu: int,
    conda_env: str = "interndata",
    candidate_id: str | None = None,
    stop_after_feasible: bool = True,
    *,
    arm: str | None = None,
    attach_prim_path_children: list[str] | None = None,
    planning_config: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_arm = arm or manifest_before.get("required_arm")
    if required_arm not in {"left", "right"}:
        raise CompileError("workspace validation requires a preselected left/right arm")
    manifest_arm = manifest_before.get("required_arm")
    if manifest_arm is not None and manifest_arm != required_arm:
        raise CompileError(
            f"workspace manifest arm {manifest_arm!r} does not match requested arm {required_arm!r}"
        )
    attach_paths = [str(value) for value in (attach_prim_path_children or []) if str(value)]
    if not attach_paths:
        raise CompileError("required-arm workspace validation needs attach_prim_path_children")
    command = [
        "python",
        "scripts/simbox/validate_workspace_candidates.py",
        "--manifest",
        str(manifest_path),
        "--gpus",
        str(gpu),
        "--conda-env",
        conda_env,
        "--arm",
        required_arm,
        "--planning-config",
        str(planning_config),
        "--planning-only",
    ]
    for path in attach_paths:
        command.extend(["--attach-prim-path-child", path])
    if stop_after_feasible:
        command.append("--stop-after-feasible")
    if candidate_id:
        command.extend(["--candidate-id", candidate_id])
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 or manifest.get("status") != "planning_success":
        code = manifest.get("failure_code") or "WORKSPACE_VALIDATION_FAILED"
        raise CompileError(f"workspace planning-only validation failed: {code}")
    selected = manifest.get("selected_candidate")
    if not isinstance(selected, dict) or selected.get("arm") != required_arm:
        raise CompileError("workspace validator did not select a candidate on the required arm")
    return selected


def validate_next_workspace_candidate(
    manifest_path: Path,
    current_candidate_id: str,
    gpu: int,
    conda_env: str = "interndata",
    *,
    arm: str | None = None,
    attach_prim_path_children: list[str] | None = None,
    planning_config: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [
        str(item["candidate_id"])
        for item in manifest.get("geometry_candidates", [])
        if item.get("geometry_feasible") and str(item.get("candidate_id")) != current_candidate_id
    ]
    errors: list[str] = []
    for candidate_id in candidates:
        try:
            return validate_workspace_manifest(
                manifest_path,
                gpu,
                conda_env,
                candidate_id=candidate_id,
                stop_after_feasible=False,
                arm=arm,
                attach_prim_path_children=attach_prim_path_children,
                planning_config=planning_config,
            )
        except CompileError as exc:
            errors.append(f"{candidate_id}: {exc}")
    raise CompileError("no alternative CuRobo-feasible workspace candidate: " + "; ".join(errors[-3:]))


def _angle_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _yaw_to_target_deg(base_xy: list[float], target_xy: list[float]) -> float:
    dx = float(target_xy[0]) - float(base_xy[0])
    dy = float(target_xy[1]) - float(base_xy[1])
    if math.hypot(dx, dy) < 1e-9:
        raise CompileError("workspace candidate overlaps a center object")
    return math.degrees(math.atan2(dy, dx))


def rank_common_workspace_candidates(
    workspace_paths: Mapping[str, str | Path],
    max_heading_error_deg: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank shared base poses before running each subtask's required-arm Probe."""

    if len(workspace_paths) < 2:
        raise CompileError("common workspace ranking requires at least two subtasks")
    if max_heading_error_deg <= 0.0 or max_heading_error_deg > 180.0:
        raise CompileError("workspace.common_pose_max_heading_error_deg must be in (0, 180]")
    if limit <= 0:
        raise CompileError("workspace.max_common_candidates must be positive")

    manifests = {
        subtask_id: json.loads(Path(path).read_text(encoding="utf-8"))
        for subtask_id, path in workspace_paths.items()
    }
    pool: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[float, float, float]] = set()
    for source_subtask, workspace_manifest in manifests.items():
        for candidate in workspace_manifest.get("geometry_candidates", []):
            if not candidate.get("geometry_feasible"):
                continue
            key = (
                round(float(candidate["world_xy"][0]), 6),
                round(float(candidate["world_xy"][1]), 6),
                round(float(candidate["yaw_deg"]), 6),
            )
            if key in seen:
                continue
            seen.add(key)
            pool.append((source_subtask, candidate))

    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    subtask_order = list(workspace_paths)
    for source_subtask, source_candidate in pool:
        score = 0.0
        metrics: dict[str, Any] = {}
        valid = True
        for subtask_id, workspace_manifest in manifests.items():
            target_xy = [float(value) for value in workspace_manifest["target"]["world_xyz"][:2]]
            world_xy = [float(value) for value in source_candidate["world_xy"][:2]]
            distance = math.dist(world_xy, target_xy)
            expected_yaw = _yaw_to_target_deg(world_xy, target_xy)
            heading_error = _angle_error_deg(float(source_candidate["yaw_deg"]), expected_yaw)
            sampling = workspace_manifest["sampling"]
            minimum = float(sampling["min_radius_m"])
            maximum = float(sampling["max_radius_m"])
            preferred = float(sampling["preferred_radius_m"])
            if not minimum <= distance <= maximum or heading_error > max_heading_error_deg:
                valid = False
                break
            radial_error = abs(distance - preferred)
            target_score = radial_error / max(maximum - minimum, 1e-6)
            target_score += heading_error / max(max_heading_error_deg, 1e-6)
            score = max(score, target_score)
            metrics[subtask_id] = {
                "target": workspace_manifest["target"]["name"],
                "required_arm": workspace_manifest.get("required_arm"),
                "distance_m": round(distance, 6),
                "heading_error_deg": round(heading_error, 6),
            }
        if valid:
            candidate = copy.deepcopy(source_candidate)
            candidate["source_candidate_id"] = str(source_candidate["candidate_id"])
            candidate["source_subtask_id"] = source_subtask
            candidate["common_metrics"] = metrics
            ranked.append(
                (
                    score,
                    subtask_order.index(source_subtask),
                    str(source_candidate["candidate_id"]),
                    candidate,
                )
            )
    ranked.sort(key=lambda value: value[:3])
    result: list[dict[str, Any]] = []
    for index, (_, _, _, candidate) in enumerate(ranked[:limit]):
        candidate["candidate_id"] = f"common_{index:03d}"
        result.append(candidate)
    return result


def _write_common_candidate_manifest(
    source_path: Path,
    output_path: Path,
    candidate: Mapping[str, Any],
) -> Path:
    workspace_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    target_xy = [float(value) for value in workspace_manifest["target"]["world_xyz"][:2]]
    world_xy = [float(value) for value in candidate["world_xy"][:2]]
    local_candidate = {
        "candidate_id": str(candidate["candidate_id"]),
        "world_xy": world_xy,
        "yaw_deg": float(candidate["yaw_deg"]),
        "radius_m": math.dist(world_xy, target_xy),
        "angle_deg": math.degrees(
            math.atan2(world_xy[1] - target_xy[1], world_xy[0] - target_xy[0])
        ),
        "collision_free": True,
        "inside_floor": True,
        "geometry_feasible": True,
        "obstacle": None,
        "rejection_code": None,
    }
    workspace_manifest.update(
        {
            "geometry_candidates": [local_candidate],
            "curobo_results": [],
            "pick_attempts": [],
            "selected_candidate": None,
            "status": "geometry_ready",
            "failure_code": None,
            "common_candidate": {
                "source_subtask_id": candidate.get("source_subtask_id"),
                "source_candidate_id": candidate.get("source_candidate_id"),
                "metrics": candidate.get("common_metrics", {}),
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(workspace_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _attach_paths(manifest: SceneCapabilityManifest, target: str) -> list[str]:
    asset = next((item for item in manifest.objects if item.name == target), None)
    if asset is None or not asset.attach_prim_path_children:
        raise CompileError(f"center object has no explicit attach proxy: {target}")
    return list(asset.attach_prim_path_children)


def select_task_workspace_candidate(
    plan: TaskPlan,
    workspace_paths: Mapping[str, str | Path],
    manifest: SceneCapabilityManifest,
    output_dir: Path,
    gpu: int,
    conda_env: str = "interndata",
    settings: Mapping[str, Any] | None = None,
    excluded_candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Select one base pose that passes every subtask's preselected-arm Probe."""

    effective_settings = dict(settings or load_agent_settings())
    excluded = set(excluded_candidate_ids or set())
    subtask_by_id = {item.subtask_id: item for item in plan.subtasks}
    if set(workspace_paths) != set(subtask_by_id):
        raise CompileError("workspace manifests do not match TaskPlan subtasks")

    if len(plan.subtasks) == 1:
        subtask = plan.subtasks[0]
        workspace_manifest_path = Path(workspace_paths[subtask.subtask_id])
        attach_paths = _attach_paths(manifest, subtask.manipulated_object)
        if not excluded:
            selected = validate_workspace_manifest(
                workspace_manifest_path,
                gpu,
                conda_env,
                arm=subtask.arm,
                attach_prim_path_children=attach_paths,
            )
        else:
            workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8"))
            candidate_ids = [
                str(item["candidate_id"])
                for item in workspace_manifest.get("geometry_candidates", [])
                if item.get("geometry_feasible") and str(item["candidate_id"]) not in excluded
            ]
            errors: list[str] = []
            selected = None
            for candidate_id in candidate_ids:
                try:
                    selected = validate_workspace_manifest(
                        workspace_manifest_path,
                        gpu,
                        conda_env,
                        candidate_id=candidate_id,
                        stop_after_feasible=False,
                        arm=subtask.arm,
                        attach_prim_path_children=attach_paths,
                    )
                    break
                except CompileError as exc:
                    errors.append(f"{candidate_id}: {exc}")
            if selected is None:
                raise CompileError(
                    "no alternative required-arm workspace candidate: " + "; ".join(errors[-3:])
                )
        selection = {
            "mode": "single",
            "candidate": selected,
            "subtasks": [
                {
                    "subtask_id": subtask.subtask_id,
                    "target": subtask.manipulated_object,
                    "arm": subtask.arm,
                }
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return selected

    workspace_settings = effective_settings.get("workspace", {})
    candidates = rank_common_workspace_candidates(
        workspace_paths,
        float(workspace_settings.get("common_pose_max_heading_error_deg", 45.0)),
        int(workspace_settings.get("max_common_candidates", 8)),
    )
    candidates = [item for item in candidates if str(item["candidate_id"]) not in excluded]
    if not candidates:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(
                {
                    "mode": "blocked",
                    "failure_code": "NO_COMMON_WORKSPACE_CANDIDATE",
                    "reason": "no shared geometry pose for all center objects",
                    "subtasks": [
                        {
                            "subtask_id": item.subtask_id,
                            "target": item.manipulated_object,
                            "arm": item.arm,
                        }
                        for item in plan.subtasks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise CompileError("NO_COMMON_WORKSPACE_CANDIDATE: no shared geometry pose for all center objects")

    failures: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = output_dir / str(candidate["candidate_id"])
        passed: list[dict[str, Any]] = []
        failed = False
        for subtask in plan.subtasks:
            probe_manifest = _write_common_candidate_manifest(
                Path(workspace_paths[subtask.subtask_id]),
                candidate_dir / subtask.subtask_id / "candidates.json",
                candidate,
            )
            try:
                selected = validate_workspace_manifest(
                    probe_manifest,
                    gpu,
                    conda_env,
                    candidate_id=str(candidate["candidate_id"]),
                    stop_after_feasible=False,
                    arm=subtask.arm,
                    attach_prim_path_children=_attach_paths(manifest, subtask.manipulated_object),
                )
                passed.append(
                    {
                        "subtask_id": subtask.subtask_id,
                        "target": subtask.manipulated_object,
                        "arm": subtask.arm,
                        "selected": selected,
                    }
                )
            except CompileError as exc:
                failures.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "subtask_id": subtask.subtask_id,
                        "error": str(exc),
                    }
                )
                failed = True
                break
        if not failed:
            selected_candidate = copy.deepcopy(candidate)
            selected_candidate["validated_subtasks"] = passed
            selection = {
                "mode": "common",
                "candidate": selected_candidate,
                "failures_before_selection": failures,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "position_selection.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return selected_candidate

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "position_selection.json").write_text(
        json.dumps(
            {
                "mode": "blocked",
                "failure_code": "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE",
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise CompileError(
        "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE: no shared pose passed every required-arm Probe"
    )
