"""Deterministically compile TaskPlan objects into SimBox task YAML."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from workflows.simbox.core.robots.profile import (
    PlacementFamily,
    RobotModelProfile,
    load_robot_profile,
)
from workflows.simbox.core.workspace.task_compiler import task_room_bounds_xy

from .contracts import (
    ExecutionVariant,
    PlaceRelation,
    SceneCapabilityManifest,
    StageExecutionMode,
    TaskPlan,
)
from .robot_skills import (
    EXECUTABLE_ADMISSION_STATES,
    RobotSkillContractError,
    validate_profile_skill_admission,
)
from .robot_skills.relation_admission import (
    RelationAdmissionError,
    validate_relation_admission,
)
from .settings import (
    DEFAULT_CONFIG_PATH,
    load_agent_settings,
    merge_mappings,
    resolve_data_generation,
    resolve_debug_camera,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CompileError(ValueError):
    def __init__(self, message: str, *, failing_subtask_id: str | None = None):
        super().__init__(message)
        self.failing_subtask_id = failing_subtask_id


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list) or not value["tasks"]:
        raise CompileError(f"invalid SimBox task document: {path}")
    return value


def _source_document_paths(source_path: Path, document: Mapping[str, Any]) -> set[Path]:
    paths = {source_path.resolve()}
    tasks = document.get("tasks")
    task = tasks[0] if isinstance(tasks, list) and tasks else {}
    arena_value = task.get("arena_file") if isinstance(task, Mapping) else None
    if isinstance(arena_value, str) and arena_value.strip():
        arena_path = Path(arena_value).expanduser()
        if arena_path.is_absolute():
            paths.add(arena_path.resolve())
        else:
            paths.add((REPO_ROOT / arena_path).resolve())
            paths.add((source_path.parent / arena_path).resolve())
    return paths


def _is_global_camera(camera: Mapping[str, Any]) -> bool:
    name = str(camera.get("name", "")).lower()
    role = str(camera.get("role", "")).lower()
    return role in {"global", "overview", "navigation"} or any(
        token in name for token in ("global", "overview", "navigate")
    )


def _global_save_name(camera: Mapping[str, Any]) -> str:
    if camera.get("save_name"):
        return str(camera["save_name"])
    name = str(camera.get("name", "global"))
    return "global" if name == "navigate_global" or name.endswith("_global") else name


def _compile_profile_cameras(
    task: Mapping[str, Any],
    instance_name: str,
    profile: RobotModelProfile,
) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for source_camera in task.get("cameras") or []:
        if not isinstance(source_camera, Mapping) or not _is_global_camera(source_camera):
            continue
        camera = copy.deepcopy(dict(source_camera))
        camera["record_to"] = instance_name
        camera["save_name"] = _global_save_name(camera)
        camera.setdefault("record_mode", "lmdb_and_video")
        cameras.append(camera)
    cameras.extend(camera.to_task_camera(instance_name) for camera in profile.camera_rig)
    names = [str(camera["name"]) for camera in cameras]
    save_names = [str(camera["save_name"]) for camera in cameras]
    if len(names) != len(set(names)):
        raise CompileError(f"compiled camera names are not unique: {names}")
    if len(save_names) != len(set(save_names)):
        raise CompileError(f"compiled camera save names are not unique: {save_names}")
    return cameras


def _execution_profile(variant: ExecutionVariant) -> RobotModelProfile:
    profile_path = Path(variant.robot_config_file).expanduser()
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    try:
        profile = load_robot_profile(profile_path)
    except ValueError as exc:
        raise CompileError(str(exc)) from exc
    if profile.profile_id != variant.profile_id:
        raise CompileError(
            f"execution variant profile {variant.profile_id!r} does not match "
            f"canonical profile {profile.profile_id!r}"
        )
    if profile.profile_hash != variant.profile_hash:
        raise CompileError(f"execution variant profile hash is stale: {profile.profile_id}")
    family = str(getattr(profile.placement.family, "value", profile.placement.family))
    if family != variant.placement_family:
        raise CompileError(
            f"execution variant placement family {variant.placement_family!r} does not "
            f"match canonical profile {family!r}"
        )
    return profile


def _replace_robot_instance(
    task: dict[str, Any],
    variant: ExecutionVariant,
) -> None:
    source_robots = task.get("robots") or []
    if len(source_robots) != 1 or not isinstance(source_robots[0], Mapping):
        raise CompileError("Agent compilation requires exactly one source robot instance")
    source_robot = source_robots[0]
    source_name = str(source_robot.get("name", "")).strip()
    if not source_name:
        raise CompileError("source robot instance must have a name")
    robot = {
        "name": variant.instance_name,
        "robot_config_file": variant.robot_config_file,
        "euler": copy.deepcopy(source_robot.get("euler", [0.0, 0.0, 0.0])),
        "use_batch": bool(source_robot.get("use_batch", True)),
        "collision_activation_distance": float(
            source_robot.get("collision_activation_distance", 0.05)
        ),
    }
    task["robots"] = [robot]
    for region in task.get("regions", []) or []:
        if isinstance(region, dict) and str(region.get("object", "")) == source_name:
            region["object"] = variant.instance_name
    for region in task.get("source_regions", []) or []:
        if not isinstance(region, dict):
            continue
        for key in ("A", "object", "robot"):
            if str(region.get(key, "")) == source_name:
                region[key] = variant.instance_name


def _container_region(
    container_regions: list[dict[str, Any]],
    target: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in container_regions
            if str(item.get("object") or item.get("target")) == target
            and bool(item.get("can_receive_objects", True))
        ),
        None,
    )


def _relation_predicate(
    relation: str,
    target: str,
    container_regions: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    if relation not in {
        PlaceRelation.ON.value,
        PlaceRelation.INSIDE.value,
        PlaceRelation.INSERT.value,
    }:
        return None
    planning = settings.get("planning", {})
    values = planning.get("relation_predicate") if isinstance(planning, Mapping) else None
    if not isinstance(values, Mapping):
        raise CompileError("agent config planning.relation_predicate must be a mapping")
    predicate = {"relation": relation}
    for key in (
        "geometry_tolerance_m",
        "support_gap_tolerance_m",
        "minimum_support_contact_n",
        "max_unexpected_contact_n",
    ):
        raw_value = values.get(key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise CompileError(f"planning.relation_predicate.{key} must be a number") from exc
        if not math.isfinite(value) or value < 0.0:
            raise CompileError(
                f"planning.relation_predicate.{key} must be finite and non-negative"
            )
        predicate[key] = value
    if relation in {PlaceRelation.INSIDE.value, PlaceRelation.INSERT.value}:
        region = _container_region(container_regions, target)
        if region is None:
            raise CompileError(
                f"{relation} placement requires a declared container region: {target}"
            )
        center = region.get("center")
        inner_size = region.get("inner_size")
        support_z = region.get("interior_support_z")
        if not (
            isinstance(center, (list, tuple))
            and len(center) == 2
            and isinstance(inner_size, (list, tuple))
            and len(inner_size) == 2
        ):
            raise CompileError(
                f"container region for {target!r} requires center and inner_size XY pairs"
            )
        try:
            center_values = [float(item) for item in center]
            size_values = [float(item) for item in inner_size]
            support_value = float(support_z)
        except (TypeError, ValueError) as exc:
            raise CompileError(f"container region for {target!r} has non-numeric geometry") from exc
        geometry = [*center_values, *size_values, support_value]
        if not all(math.isfinite(item) for item in geometry) or any(
            item <= 0.0 for item in size_values
        ):
            raise CompileError(f"container region for {target!r} has invalid geometry")
        predicate["container_region"] = {
            "name": str(region.get("name") or region.get("region") or ""),
            "center": center_values,
            "inner_size": size_values,
            "interior_support_z": support_value,
        }
    return predicate


def _skill_defaults(settings: Mapping[str, Any], name: str) -> dict[str, Any]:
    values = settings.get("skill_defaults", {})
    result = values.get(name) if isinstance(values, Mapping) else None
    if not isinstance(result, Mapping):
        raise CompileError(f"agent config has no skill_defaults.{name} mapping")
    return copy.deepcopy(dict(result))


def _compile_skill(
    skill,
    subtask_id: str,
    relation: str,
    container_regions: list[dict[str, Any]],
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
    else:
        raise CompileError(f"unsupported physics-schema Skill: {name}")
    value.update(copy.deepcopy(skill.params))
    value["name"] = name
    value["objects"] = list(skill.objects)
    value["agent_subtask_id"] = subtask_id
    value["test_mode"] = "forward"
    if name == "place":
        value["semantic_relation"] = relation
        predicate = _relation_predicate(
            relation,
            skill.objects[1],
            container_regions,
            settings,
        )
        if predicate is not None:
            value["success_mode"] = "relation_predicate"
            value["relation_predicate"] = predicate
    if "ignore_substring" in value:
        raise CompileError("Agent-generated physics-schema Skills may not use ignore_substring")
    return value


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _resolve_arm(skill_arm: str, subtask_arm: str) -> str:
    arm = skill_arm if skill_arm not in {"auto", "any_single_arm"} else subtask_arm
    if arm not in {"left", "right"}:
        raise CompileError("workspace planning must select left or right before final config compilation")
    return arm


def _compile_stage_phases(
    stage,
    subtask_id: str,
    subtask_arm: str,
    relation: str,
    container_regions: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> list[dict[str, list[dict[str, Any]]]]:
    """Translate the explicit planning mode to SimBox's phase/arm nesting."""

    mode = _enum_text(stage.execution_mode)
    ordered: list[tuple[str, dict[str, Any]]] = [
        (
            _resolve_arm(skill.arm, subtask_arm),
            _compile_skill(
                skill,
                subtask_id,
                relation,
                container_regions,
                settings,
            ),
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

def compile_task_config(
    plan: TaskPlan,
    execution_variant: ExecutionVariant,
    manifest: SceneCapabilityManifest,
    output_path: Path,
    workspace_candidate: dict[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    admission_states: frozenset[str] = EXECUTABLE_ADMISSION_STATES,
    scene_task_path: Path | None = None,
) -> Path:
    effective_settings = dict(settings or load_agent_settings())
    plan_source_path = Path(plan.source_task).resolve()
    if (
        plan.selected_task_id != manifest.task_id
        or plan_source_path != Path(manifest.source_task).resolve()
    ):
        raise CompileError("TaskPlan source does not match the selected inventory manifest")
    source_path = (scene_task_path or plan_source_path).resolve()
    document = copy.deepcopy(load_yaml(source_path))
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        raise CompileError("compiled task output must be a YAML file")
    protected_sources = _source_document_paths(source_path, document)
    if source_path != plan_source_path:
        protected_sources.update(
            _source_document_paths(plan_source_path, load_yaml(plan_source_path))
        )
    if output_path in protected_sources:
        raise CompileError(
            f"compiled output must not overwrite a source task or arena: {output_path}"
        )
    task = document["tasks"][0]
    scene_layout = (task.get("metadata") or {}).get("agent_scene_layout")
    if source_path != plan_source_path:
        if not isinstance(scene_layout, Mapping):
            raise CompileError("scene_task_path must be a typed SceneLayout revision")
        declared_source = Path(str(scene_layout.get("source_task") or "")).resolve()
        if declared_source != plan_source_path:
            raise CompileError(
                "SceneLayout revision source does not match TaskPlan.source_task"
            )
        declared_hash = str(scene_layout.get("source_task_hash") or "")
        current_hash = hashlib.sha256(plan_source_path.read_bytes()).hexdigest()
        if declared_hash != current_hash:
            raise CompileError("SceneLayout revision source hash no longer matches")
    robot_profile = _execution_profile(execution_variant)
    _replace_robot_instance(task, execution_variant)
    robot_name = execution_variant.instance_name
    if set(execution_variant.arm_binding) != {
        subtask.subtask_id for subtask in plan.subtasks
    }:
        raise CompileError("execution variant arm_binding must cover every TaskPlan subtask")
    for subtask in plan.subtasks:
        bound_arm = execution_variant.arm_binding[subtask.subtask_id]
        if bound_arm not in robot_profile.arms:
            raise CompileError(
                f"execution variant binds {subtask.subtask_id!r} to unavailable arm {bound_arm!r}"
            )
        if subtask.arm in {"left", "right"} and subtask.arm != bound_arm:
            raise CompileError(
                f"execution variant arm {bound_arm!r} violates subtask constraint {subtask.arm!r}"
            )
        if subtask.arm != "both":
            explicit_arms = {
                skill.arm
                for stage in subtask.stages
                for skill in stage.skills
                if skill.arm in {"left", "right"}
            }
            if explicit_arms and explicit_arms != {bound_arm}:
                raise CompileError(
                    f"execution variant arm {bound_arm!r} violates explicit Skill arms "
                    f"{sorted(explicit_arms)}"
                )
    if workspace_candidate is not None:
        document = apply_workspace_candidate(document, source_path, workspace_candidate)
        task = document["tasks"][0]
    task["cameras"] = _compile_profile_cameras(task, robot_name, robot_profile)
    debug_settings = effective_settings.get("debug", {})
    if not isinstance(debug_settings, Mapping):
        raise CompileError("Agent config debug must be a mapping")
    task["debug_topdown_check"] = bool(debug_settings.get("topdown_check", False))
    if task["debug_topdown_check"]:
        task["debug_topdown_camera"] = resolve_debug_camera(effective_settings)
        task["debug_topdown_camera"]["target_object"] = plan.subtasks[0].manipulated_object
        task["debug_topdown_camera"]["room_bounds_xy"] = task_room_bounds_xy(
            task, source_path
        )
    else:
        task.pop("debug_topdown_camera", None)

    stages: list[dict[str, Any]] = []
    for subtask in plan.subtasks:
        relation = _enum_text(subtask.relation)
        try:
            validate_relation_admission(
                relation,
                str(subtask.target_object or ""),
                (
                    item
                    for item in task.get("container_regions", []) or []
                    if isinstance(item, Mapping)
                ),
                (
                    item
                    for item in task.get("regions", []) or []
                    if isinstance(item, Mapping)
                ),
            )
        except RelationAdmissionError as exc:
            raise CompileError(
                str(exc), failing_subtask_id=subtask.subtask_id
            ) from exc
        for stage in subtask.stages:
            phases = _compile_stage_phases(
                stage,
                subtask.subtask_id,
                execution_variant.arm_binding[subtask.subtask_id],
                relation,
                [
                    dict(item)
                    for item in task.get("container_regions", []) or []
                    if isinstance(item, Mapping)
                ],
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

    collision_world_mode = str((planning.get("collision_world") or {}).get("mode", ""))
    if execution_variant.collision_world_mode != collision_world_mode:
        raise CompileError(
            f"execution variant collision mode {execution_variant.collision_world_mode!r} "
            f"does not match compiler mode {collision_world_mode!r}"
        )
    skill_names = [
        skill.name.lower()
        for subtask in plan.subtasks
        for stage in subtask.stages
        for skill in stage.skills
    ]
    try:
        required_capabilities = validate_profile_skill_admission(
            robot_profile,
            skill_names,
            collision_world_mode,
            allowed_states=admission_states,
        )
    except RobotSkillContractError as exc:
        raise CompileError(str(exc)) from exc
    if frozenset(plan.robot_requirement.required_capabilities) != required_capabilities:
        raise CompileError(
            "TaskPlan robot_requirement.required_capabilities does not match its Skill requirements"
        )


    metadata = task.setdefault("metadata", {})
    workspace_id = str(workspace_candidate.get("candidate_id")) if workspace_candidate else None
    metadata["agent_plan"] = {
        "selected_task_id": plan.selected_task_id,
        "prompt": plan.prompt,
        "execution_variant_id": execution_variant.variant_id,
        "robot_instance": robot_name,
        "robot_profile_id": robot_profile.profile_id,
        "robot_profile_hash": robot_profile.profile_hash,
        "placement_family": str(
            getattr(robot_profile.placement.family, "value", robot_profile.placement.family)
        ),
        "required_capabilities": sorted(required_capabilities),
        "workspace_candidate_id": workspace_id,
        "scene_revision": (
            str(scene_layout.get("scene_revision"))
            if isinstance(scene_layout, Mapping)
            else "source"
        ),
        "data_generation": resolve_data_generation(
            plan.task_request.data_generation,
            effective_settings,
        ),
        "subtasks": [
            {
                "subtask_id": item.subtask_id,
                "center_object": item.manipulated_object,
                "target_object": item.target_object,
                "relation": _enum_text(item.relation),
                "arm_constraint": item.arm,
                "arm": execution_variant.arm_binding[item.subtask_id],
            }
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
                    "arm": execution_variant.arm_binding[item.subtask_id],
                }
                for item in plan.subtasks
            ],
        }

    from workflows.simbox.core.planning.config_contract import validate_planning_contract

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
    placement_family: str,
) -> Path:
    if arm not in {"left", "right"}:
        raise CompileError(f"workspace generation requires a fixed left/right arm, got {arm!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        family = PlacementFamily(placement_family)
    except ValueError as exc:
        raise CompileError(f"invalid robot placement family: {placement_family!r}") from exc
    if family == PlacementFamily.SUPPORT_MOUNTED:
        script = "scripts/simbox/plan_support_mounted_workspace.py"
    elif family == PlacementFamily.FLOOR_STANDING:
        script = "scripts/simbox/plan_workspace_layout.py"
    else:
        raise CompileError(f"unsupported robot placement family: {family}")

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
        failure_code = "WORKSPACE_GEOMETRY_PLANNING_FAILED"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            feasible = sum(c.get("geometry_feasible") for c in manifest.get("geometry_candidates", []))
            if feasible == 0:
                _write_workspace_diagnostics(manifest, target, arm, output_dir)
            declared_code = manifest.get("failure_code")
            if isinstance(declared_code, str) and declared_code.strip():
                failure_code = declared_code.strip()
        detail = (completed.stderr or "").strip().split("\n")[-1] if completed.stderr else ""
        raise CompileError(
            f"{failure_code}: workspace geometry planning failed "
            f"(exit {completed.returncode}): {detail}\n"
            f"expected: {manifest_path}\n"
            f"stderr log: {output_dir / 'workspace_plan_stderr.log'}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_path = output_dir / "static_validation.json"
    validation = _workspace_static_validation(
        manifest,
        source_task=source_task,
        target=target,
        arm=arm,
        placement_family=family.value,
    )
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not validation["hard_ok"]:
        raise CompileError(
            f"workspace static validation failed: {validation['failure_code']}; "
            f"artifact: {validation_path}"
        )
    return manifest_path


def _workspace_static_validation(
    manifest: Mapping[str, Any],
    *,
    source_task: Path,
    target: str,
    arm: str,
    placement_family: str,
) -> dict[str, Any]:
    source_document = load_yaml(source_task)
    source_metadata = source_document["tasks"][0].get("metadata") or {}
    scene_layout = source_metadata.get("agent_scene_layout")
    scene_revision = (
        str(scene_layout.get("scene_revision"))
        if isinstance(scene_layout, Mapping)
        else "source"
    )
    robot = manifest.get("robot")
    target_value = manifest.get("target")
    support = manifest.get("support")
    candidates = manifest.get("geometry_candidates")
    asset_rows = manifest.get("asset_audit")
    checks = {
        "schema_version": manifest.get("version") == 4,
        "source_task": Path(str(manifest.get("source_task") or "")).resolve()
        == source_task.resolve(),
        "placement_family": isinstance(robot, Mapping)
        and robot.get("placement_family") == placement_family,
        "required_arm": manifest.get("required_arm") == arm,
        "target": isinstance(target_value, Mapping)
        and target_value.get("name") == target
        and isinstance(target_value.get("world_xyz"), list)
        and len(target_value["world_xyz"]) == 3,
        "support": isinstance(support, Mapping)
        and bool(support.get("name")),
        "geometry": isinstance(candidates, list)
        and any(
            isinstance(candidate, Mapping) and candidate.get("geometry_feasible") is True
            for candidate in candidates
        ),
        "assets": _workspace_assets_valid(asset_rows, target),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "stage": "workspace_static_validation",
        "scene_revision": scene_revision,
        "source_task": str(source_task.resolve()),
        "target": target,
        "arm": arm,
        "placement_family": placement_family,
        "checks": checks,
        "hard_ok": not failed,
        "failure_code": None if not failed else "WORKSPACE_STATIC_VALIDATION_FAILED",
        "failed_checks": failed,
    }


def _workspace_assets_valid(value: Any, target: str) -> bool:
    if not isinstance(value, list) or not value:
        return False
    target_seen = False
    for row in value:
        if not isinstance(row, Mapping):
            return False
        if row.get("usd_exists") is not True or row.get("scale_valid") is not True:
            return False
        if row.get("name") == target:
            target_seen = True
            if not (
                row.get("grasp_exists") is True
                and row.get("grasp_shape_valid") is True
                and row.get("grasp_finite") is True
            ):
                return False
    return target_seen

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
    diagnostic_disable_curobo_obstacle_paths: list[str] | None = None,
    diagnostic_disable_physics_and_curobo_obstacle_paths: list[str] | None = None,
    diagnostic_disable_collision_entities: list[str] | None = None,
    planning_gate: str = "pick-place",
    seed: int = 0,
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
    if planning_gate not in {"pick", "pick-place"}:
        raise CompileError("planning_gate must be pick or pick-place")
    if seed < 0:
        raise CompileError("workspace validation seed must be non-negative")
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
        "--planning-gate",
        planning_gate,
        "--seed",
        str(seed),
    ]
    for path in attach_paths:
        command.extend(["--attach-prim-path-child", path])
    diagnostic_paths = [
        str(value).strip() for value in (diagnostic_disable_curobo_obstacle_paths or [])
    ]
    if any(not value or not value.startswith("/") for value in diagnostic_paths) or len(
        diagnostic_paths
    ) != len(set(diagnostic_paths)):
        raise CompileError(
            "diagnostic CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    for path in diagnostic_paths:
        command.extend(["--diagnostic-disable-curobo-obstacle-path", path])
    dual_paths = [
        str(value).strip()
        for value in (diagnostic_disable_physics_and_curobo_obstacle_paths or [])
    ]
    collision_entities = [
        str(value).strip() for value in (diagnostic_disable_collision_entities or [])
    ]
    if any(not value or not value.startswith("/") for value in dual_paths) or len(
        dual_paths
    ) != len(set(dual_paths)):
        raise CompileError(
            "diagnostic Physics+CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    if any(not value for value in collision_entities) or len(collision_entities) != len(
        set(collision_entities)
    ):
        raise CompileError(
            "diagnostic collision entity names must be unique and non-empty"
        )
    if diagnostic_paths and (dual_paths or collision_entities):
        raise CompileError(
            "CuRobo-only and Physics+CuRobo diagnostic isolation modes are mutually exclusive"
        )
    if planning_gate == "pick-place" and (
        diagnostic_paths or dual_paths or collision_entities
    ):
        raise CompileError(
            "Pick+Place planning validation requires the complete Physics/CuRobo world"
        )
    for path in dual_paths:
        command.extend(
            ["--diagnostic-disable-physics-and-curobo-obstacle-path", str(path)]
        )
    for entity in collision_entities:
        command.extend(["--diagnostic-disable-collision-entity", str(entity)])
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
    diagnostic_disable_curobo_obstacle_paths: list[str] | None = None,
    diagnostic_disable_physics_and_curobo_obstacle_paths: list[str] | None = None,
    diagnostic_disable_collision_entities: list[str] | None = None,
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
                diagnostic_disable_curobo_obstacle_paths=(
                    diagnostic_disable_curobo_obstacle_paths
                ),
                diagnostic_disable_physics_and_curobo_obstacle_paths=(
                    diagnostic_disable_physics_and_curobo_obstacle_paths
                ),
                diagnostic_disable_collision_entities=(
                    diagnostic_disable_collision_entities
                ),
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
        "mount_support": candidate.get("mount_support"),
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
    execution_variant: ExecutionVariant,
    workspace_paths: Mapping[str, str | Path],
    manifest: SceneCapabilityManifest,
    output_dir: Path,
    gpu: int,
    conda_env: str = "interndata",
    settings: Mapping[str, Any] | None = None,
    excluded_candidate_ids: set[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Select one base pose that passes every subtask's preselected-arm Probe."""

    effective_settings = dict(settings or load_agent_settings())
    debug_settings = effective_settings.get("debug", {})
    if not isinstance(debug_settings, Mapping):
        raise CompileError("agent debug config must be a mapping")
    workspace_probe_settings = debug_settings.get("workspace_probe", {})
    if not isinstance(workspace_probe_settings, Mapping):
        raise CompileError("agent debug.workspace_probe config must be a mapping")
    raw_diagnostic_paths = workspace_probe_settings.get(
        "disable_curobo_obstacle_paths", []
    )
    if not isinstance(raw_diagnostic_paths, (list, tuple)):
        raise CompileError(
            "debug.workspace_probe.disable_curobo_obstacle_paths must be a list"
        )
    diagnostic_paths = [str(value).strip() for value in raw_diagnostic_paths]
    if any(not value or not value.startswith("/") for value in diagnostic_paths) or len(
        diagnostic_paths
    ) != len(set(diagnostic_paths)):
        raise CompileError(
            "debug.workspace_probe.disable_curobo_obstacle_paths must contain "
            "unique, non-empty absolute Prim paths"
        )
    raw_dual_paths = workspace_probe_settings.get(
        "disable_physics_and_curobo_obstacle_paths", []
    )
    raw_collision_entities = workspace_probe_settings.get(
        "disable_collision_entities", []
    )
    if not isinstance(raw_dual_paths, (list, tuple)) or not isinstance(
        raw_collision_entities, (list, tuple)
    ):
        raise CompileError("debug workspace probe diagnostic selectors must be lists")
    dual_paths = [str(value).strip() for value in raw_dual_paths]
    collision_entities = [str(value).strip() for value in raw_collision_entities]
    if diagnostic_paths and (dual_paths or collision_entities):
        raise CompileError(
            "CuRobo-only and Physics+CuRobo diagnostic isolation modes are mutually exclusive"
        )
    excluded = set(excluded_candidate_ids or set())
    if seed < 0:
        raise CompileError("workspace selection seed must be non-negative")
    subtask_by_id = {item.subtask_id: item for item in plan.subtasks}
    if set(workspace_paths) != set(subtask_by_id):
        raise CompileError("workspace manifests do not match TaskPlan subtasks")
    if set(execution_variant.arm_binding) != set(subtask_by_id):
        raise CompileError("execution variant arm_binding does not match TaskPlan subtasks")

    if len(plan.subtasks) == 1:
        subtask = plan.subtasks[0]
        workspace_manifest_path = Path(workspace_paths[subtask.subtask_id])
        attach_paths = _attach_paths(manifest, subtask.manipulated_object)
        if not excluded:
            try:
                selected = validate_workspace_manifest(
                    workspace_manifest_path,
                    gpu,
                    conda_env,
                    arm=execution_variant.arm_binding[subtask.subtask_id],
                    attach_prim_path_children=attach_paths,
                    diagnostic_disable_curobo_obstacle_paths=diagnostic_paths,
                    diagnostic_disable_physics_and_curobo_obstacle_paths=dual_paths,
                    diagnostic_disable_collision_entities=collision_entities,
                    seed=seed,
                )
            except CompileError as exc:
                raise CompileError(
                    str(exc), failing_subtask_id=subtask.subtask_id
                ) from exc
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
                        arm=execution_variant.arm_binding[subtask.subtask_id],
                        attach_prim_path_children=attach_paths,
                        diagnostic_disable_curobo_obstacle_paths=diagnostic_paths,
                        diagnostic_disable_physics_and_curobo_obstacle_paths=dual_paths,
                        diagnostic_disable_collision_entities=collision_entities,
                        seed=seed,
                    )
                    break
                except CompileError as exc:
                    errors.append(f"{candidate_id}: {exc}")
            if selected is None:
                raise CompileError(
                    "no alternative required-arm workspace candidate: " + "; ".join(errors[-3:]),
                    failing_subtask_id=subtask.subtask_id,
                )
        selection = {
            "mode": "single",
            "seed": seed,
            "candidate": selected,
            "subtasks": [
                {
                    "subtask_id": subtask.subtask_id,
                    "target": subtask.manipulated_object,
                    "arm": execution_variant.arm_binding[subtask.subtask_id],
                    "workspace_manifest_path": str(workspace_manifest_path.resolve()),
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
                    "failing_subtask_id": None,
                    "reason": "no shared geometry pose for all center objects",
                    "subtasks": [
                        {
                            "subtask_id": item.subtask_id,
                            "target": item.manipulated_object,
                            "arm": execution_variant.arm_binding[item.subtask_id],
                            "workspace_manifest_path": str(
                                Path(workspace_paths[item.subtask_id]).resolve()
                            ),
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
                    arm=execution_variant.arm_binding[subtask.subtask_id],
                    attach_prim_path_children=_attach_paths(manifest, subtask.manipulated_object),
                    diagnostic_disable_curobo_obstacle_paths=diagnostic_paths,
                    diagnostic_disable_physics_and_curobo_obstacle_paths=dual_paths,
                    diagnostic_disable_collision_entities=collision_entities,
                    seed=seed,
                )
                passed.append(
                    {
                        "subtask_id": subtask.subtask_id,
                        "target": subtask.manipulated_object,
                        "arm": execution_variant.arm_binding[subtask.subtask_id],
                        "workspace_manifest_path": str(probe_manifest.resolve()),
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
                "seed": seed,
                "candidate": selected_candidate,
                "failures_before_selection": failures,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "position_selection.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return selected_candidate

    failed_subtasks = {str(item["subtask_id"]) for item in failures}
    failing_subtask_id = (
        next(iter(failed_subtasks)) if len(failed_subtasks) == 1 else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "position_selection.json").write_text(
        json.dumps(
            {
                "mode": "blocked",
                "failure_code": "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE",
                "failing_subtask_id": failing_subtask_id,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise CompileError(
        "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE: no shared pose passed every required-arm Probe",
        failing_subtask_id=failing_subtask_id,
    )
