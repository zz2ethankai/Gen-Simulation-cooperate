"""YAML-facing orchestration for the offline target-annulus planner."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import yaml

from .geometry import (
    colliding_fixture,
    colliding_fixture_layer,
    inside_floor,
    sample_target_annulus,
    yaw_to_align_arm_base_deg,
    inside_rect,
    sample_table_edge,
    table_edge_centers,
    _edge_endpoints,
)
from .models import (
    DEFAULT_ROBOT_PROFILES,
    GeometryCandidate,
    SamplingConfig,
    WorkspaceManifest,
    WorkspacePlanningError,
)


def load_yaml(path: Path) -> MutableMapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, MutableMapping):
        raise WorkspacePlanningError("INVALID_TASK_YAML", f"YAML root must be a mapping: {path}")
    return value


def dump_yaml(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def dump_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _one_task(document: Mapping[str, Any]) -> MutableMapping[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], MutableMapping):
        raise WorkspacePlanningError("INVALID_TASK_COUNT", "workspace planner requires exactly one task")
    return tasks[0]

def _remove_stale_robot_regions(task: dict[str, Any], current_robot: str) -> None:
    object_names = {str(o.get("name", "")) for o in task.get("objects") or []}
    task["regions"] = [
        r
        for r in task.get("regions", [])
        if str(r.get("object", "")) == current_robot or str(r.get("object", "")) in object_names
    ]

def _resolve_config_path(path_value: str, input_path: Path) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, input_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WorkspacePlanningError(
        "ASSET_NOT_FOUND",
        f"referenced config does not exist: {path_value}",
        {"candidates": [str(candidate) for candidate in candidates]},
    )


def _resolve_asset_root(task: Mapping[str, Any], obj: Mapping[str, Any]) -> Path:
    root_value = obj.get("asset_root", task.get("asset_root"))
    if not root_value:
        raise WorkspacePlanningError("ASSET_NOT_FOUND", f"object {obj.get('name')!r} has no asset_root")
    return Path(os.path.abspath(os.path.expanduser(str(root_value))))


def _resolve_object_asset_path(task: Mapping[str, Any], obj: Mapping[str, Any]) -> Path:
    path_value = obj.get("path")
    if not path_value:
        raise WorkspacePlanningError("ASSET_NOT_FOUND", f"object {obj.get('name')!r} has no path")
    path = Path(os.path.expanduser(str(path_value)))
    return path.resolve() if path.is_absolute() else (_resolve_asset_root(task, obj) / path).resolve()


def _valid_size_xy(values: Any) -> list[float] | None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) < 2:
        return None
    try:
        size_xy = [float(values[0]), float(values[1])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in size_xy):
        return None
    return size_xy


def _valid_size_xyz(values: Any) -> list[float] | None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) < 3:
        return None
    try:
        size_xyz = [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in size_xyz):
        return None
    return size_xyz


def _resolve_fixture_asset_path(
    task: Mapping[str, Any], fixture: Mapping[str, Any], value: Any
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(os.path.expanduser(value))
    if path.is_absolute():
        return path.resolve()
    root_value = fixture.get("asset_root", task.get("asset_root"))
    if not root_value:
        return None
    return (Path(os.path.abspath(os.path.expanduser(str(root_value)))) / path).resolve()


def _metadata_size_xyz(metadata_path: Path) -> tuple[list[float], str] | None:
    """Return an extent only from fields whose coordinate convention is known."""

    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, Mapping):
        return None

    # These two fields are already expressed in the delivered Z-up USD frame.
    for section, field in (
        ("geometry_alignment", "usd_size_xyz_m"),
        ("missing_base_repair", "target_usd_extents"),
    ):
        section_value = metadata.get(section)
        values = section_value.get(field) if isinstance(section_value, Mapping) else None
        size_xyz = _valid_size_xyz(values)
        if size_xyz is not None:
            return size_xyz, f"{section}.{field}"

    # Legacy layout metadata is [x, up_y, z], so X/Z are horizontal.
    layout = metadata.get("layout_pose")
    values = layout.get("size_xyz_m") if isinstance(layout, Mapping) else None
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) >= 3:
        size_xyz = _valid_size_xyz([values[0], values[2], values[1]])
        if size_xyz is not None:
            return size_xyz, "layout_pose.size_xyz_m[x,z,y]"
    return None


def _metadata_size_xy(metadata_path: Path) -> tuple[list[float], str] | None:
    result = _metadata_size_xyz(metadata_path)
    if result is None:
        return None
    size_xyz, source = result
    return size_xyz[:2], source


def _usd_bbox_size_xyz(asset_path: Path) -> list[float] | None:
    """Compute the delivered asset's horizontal extent without launching Isaac Sim."""

    if not asset_path.is_file():
        return None
    try:
        # pxr is deliberately imported lazily: normal inline/metadata paths do
        # not need to open a USD stage, and this package remains Isaac-free.
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(asset_path))
        if stage is None:
            return None
        prim = stage.GetDefaultPrim()
        if not prim or not prim.IsValid():
            prim = stage.GetPseudoRoot()
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        aligned_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes).ComputeWorldBound(
            prim
        ).ComputeAlignedRange()
        size = aligned_range.GetSize()
        return _valid_size_xyz([size[0], size[1], size[2]])
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


def _usd_bbox_size_xy(asset_path: Path) -> list[float] | None:
    size_xyz = _usd_bbox_size_xyz(asset_path)
    return size_xyz[:2] if size_xyz is not None else None


def _fixture_asset_size_xyz(
    task: Mapping[str, Any], fixture: Mapping[str, Any]
) -> tuple[list[float] | None, str | None]:
    explicit_metadata = _resolve_fixture_asset_path(task, fixture, fixture.get("source_metadata"))
    if explicit_metadata is not None:
        result = _metadata_size_xyz(explicit_metadata)
        if result is not None:
            size_xyz, field = result
            return size_xyz, f"source_metadata:{explicit_metadata}#{field}"

    asset_path = _resolve_fixture_asset_path(task, fixture, fixture.get("path"))
    if asset_path is not None:
        adjacent_metadata = asset_path.with_name("metadata.json")
        if explicit_metadata is None or adjacent_metadata != explicit_metadata:
            result = _metadata_size_xyz(adjacent_metadata)
            if result is not None:
                size_xyz, field = result
                return size_xyz, f"adjacent_metadata:{adjacent_metadata}#{field}"
        size_xyz = _usd_bbox_size_xyz(asset_path)
        if size_xyz is not None:
            return size_xyz, f"usd_bbox:{asset_path}"
    return None, None


def _fixture_size_xy(
    task: Mapping[str, Any], fixture: Mapping[str, Any]
) -> tuple[list[float] | None, str | None]:
    """Resolve a fixture footprint in a deterministic, auditable order.

    Priority: arena inline size -> explicit source_metadata -> adjacent
    metadata.json -> actual USD bbox -> strict caller failure.
    """

    inline_size = _valid_size_xy(fixture.get("size"))
    if inline_size is not None:
        return inline_size, "arena_inline_size"

    size_xyz, source = _fixture_asset_size_xyz(task, fixture)
    return (size_xyz[:2], source) if size_xyz is not None else (None, None)


def _fixtures_with_asset_extents(
    task: Mapping[str, Any], fixtures: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Copy fixtures and materialize missing collision footprint dimensions."""

    resolved: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for fixture in fixtures:
        value = copy.deepcopy(dict(fixture))
        if value.get("collision_enabled", value.get("collision", True)):
            size_xy, size_source = _fixture_size_xy(task, value)
            if size_xy is not None:
                # Inline arena dimensions are already world-space dimensions.
                # Asset-derived extents still need the fixture instance scale.
                if size_source != "arena_inline_size":
                    scale = value.get("scale", [1.0, 1.0, 1.0])
                    if isinstance(scale, Sequence) and len(scale) >= 2:
                        size_xy = [size_xy[0] * float(scale[0]), size_xy[1] * float(scale[1])]
                value["size"] = size_xy
                value["size_source"] = size_source
                if str(value.get("target_class", "")) == "GeometryObject":
                    inline_xyz = _valid_size_xyz(fixture.get("size"))
                    size_xyz, xyz_source = (
                        (inline_xyz, "arena_inline_size")
                        if inline_xyz is not None and size_source == "arena_inline_size"
                        else _fixture_asset_size_xyz(task, value)
                    )
                    if size_xyz is not None:
                        scale = value.get("scale", [1.0, 1.0, 1.0])
                        if xyz_source != "arena_inline_size" and isinstance(scale, Sequence) and len(scale) >= 3:
                            size_xyz = [
                                size_xyz[index] * float(scale[index]) for index in range(3)
                            ]
                        value["size_xyz"] = size_xyz
                        value["size_xyz_source"] = xyz_source
            elif str(value.get("target_class", "")) == "GeometryObject":
                asset_path = _resolve_fixture_asset_path(task, value, value.get("path"))
                metadata_path = _resolve_fixture_asset_path(
                    task, value, value.get("source_metadata")
                )
                missing.append(
                    {
                        "name": str(value.get("name", "")),
                        "asset_path": str(asset_path) if asset_path else None,
                        "source_metadata": str(metadata_path) if metadata_path else None,
                    }
                )
        resolved.append(value)
    if missing:
        raise WorkspacePlanningError(
            "FIXTURE_EXTENT_MISSING",
            "collision-enabled GeometryObject fixture has no auditable horizontal extent",
            {"fixtures": missing},
        )
    return resolved


def audit_assets(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in task.get("objects", []):
        row: dict[str, Any] = {
            "name": str(obj.get("name", "")),
            "role": str(obj.get("role", "")),
            "target_class": str(obj.get("target_class", "")),
        }
        try:
            usd_path = _resolve_object_asset_path(task, obj)
            grasp_path = usd_path.with_name("Aligned_grasp_sparse.npy")
            row.update(
                {
                    "usd_path": str(usd_path),
                    "usd_exists": usd_path.is_file(),
                    "grasp_path": str(grasp_path),
                    "grasp_exists": grasp_path.is_file(),
                }
            )
            if grasp_path.is_file():
                try:
                    grasps = np.load(grasp_path, mmap_mode="r")
                    row["grasp_shape"] = list(grasps.shape)
                    row["grasp_shape_valid"] = (
                        grasps.ndim == 2 and grasps.shape[0] > 0 and grasps.shape[1] in {16, 17}
                    )
                    row["grasp_finite"] = bool(np.isfinite(grasps).all())
                except (OSError, TypeError, ValueError) as exc:
                    row.update(
                        {
                            "grasp_error": str(exc),
                            "grasp_shape_valid": False,
                            "grasp_finite": False,
                        }
                    )
        except (OSError, TypeError, ValueError, WorkspacePlanningError) as exc:
            row.update({"error": str(exc), "usd_exists": False, "grasp_exists": False})
        try:
            scale = [float(value) for value in obj.get("scale", [1.0, 1.0, 1.0])]
            row["scale"] = scale
            row["scale_valid"] = len(scale) == 3 and all(
                math.isfinite(value) and value > 0 for value in scale
            )
        except (TypeError, ValueError):
            row.update({"scale": obj.get("scale"), "scale_valid": False})
        rows.append(row)
    return rows


def _pick_target(task: Mapping[str, Any]) -> str | None:
    for robot_entry in task.get("skills", []):
        if not isinstance(robot_entry, Mapping):
            continue
        for stages in robot_entry.values():
            if not isinstance(stages, Sequence):
                continue
            for stage in stages:
                if not isinstance(stage, Mapping):
                    continue
                for skills in stage.values():
                    if not isinstance(skills, Sequence):
                        continue
                    for skill in skills:
                        if isinstance(skill, Mapping) and str(skill.get("name", "")).lower() == "pick":
                            objects = skill.get("objects") or []
                            if objects:
                                return str(objects[0])
    return None


def _resolve_target(task: Mapping[str, Any], target_name: str | None) -> str:
    target = target_name or _pick_target(task)
    if not target:
        active = list(task.get("delivery_active_objects") or [])
        target = str(active[0]) if active else None
    object_names = {str(item.get("name", "")) for item in task.get("objects", [])}
    if not target or target not in object_names:
        raise WorkspacePlanningError("TARGET_NOT_FOUND", f"target {target!r} is absent from task objects")
    return target


def _target_reference(
    task: Mapping[str, Any], target_name: str, fixture_by_name: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    region = next((item for item in task.get("regions", []) if str(item.get("object", "")) == target_name), None)
    if region is None:
        raise WorkspacePlanningError("TARGET_REGION_MISSING", f"target {target_name!r} has no runtime region")
    center = region.get("center")
    if not isinstance(center, Sequence) or len(center) < 2:
        raise WorkspacePlanningError("TARGET_REGION_MISSING", f"target {target_name!r} region has no center metadata")
    support_name = str(region.get("target", region.get("B", "")))
    if not support_name:
        raise WorkspacePlanningError("WORKSPACE_NOT_FOUND", f"target {target_name!r} region has no support")
    support = fixture_by_name.get(support_name)
    if support is None:
        raise WorkspacePlanningError("WORKSPACE_NOT_FOUND", f"support {support_name!r} is absent from arena")
    translation = support.get("translation", [0.0, 0.0, 0.0])
    z = float(region.get("support_surface_z", support.get("support_surface_z", translation[2])))
    target = {
        "name": target_name,
        "world_xyz": [float(center[0]), float(center[1]), z],
        "region": str(region.get("name", "")),
        "support": support_name,
    }
    support_ref = {
        "name": support_name,
        "parent_fixture": region.get("parent_fixture", region.get("support_target_fixture")),
    }
    return target, support_ref


def _sampling_config(task: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> SamplingConfig:
    workspace = task.get("manipulation_workspace") or {}
    values = dict(workspace.get("sampling") or {})
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    planner = str(values.get("planner", workspace.get("planner", "target_annulus_v1")))
    radial_count = values.get("radial_count")
    angular_count = values.get("angular_count")
    yaw_offsets = tuple(float(value) for value in values.get("yaw_offsets_deg", [0.0]))
    if planner == "target_annulus_v2" and "candidate_count" not in values:
        candidate_count = (
            int(radial_count or 16) * int(angular_count or 72) * len(yaw_offsets)
        )
    else:
        candidate_count = int(values.get("candidate_count", 96))
    config = SamplingConfig(
        planner=planner,
        min_radius_m=float(values.get("min_radius_m", 0.45)),
        max_radius_m=float(values.get("max_radius_m", 1.05)),
        candidate_count=candidate_count,
        preferred_radius_m=float(values.get("preferred_radius_m", 0.75)),
        sequence=str(
            values.get(
                "sequence", "polar_grid" if planner == "target_annulus_v2" else "golden_angle"
            )
        ),
        radial_count=int(radial_count) if radial_count is not None else None,
        angular_count=int(angular_count) if angular_count is not None else None,
        yaw_policy=str(values.get("yaw_policy", "face_target")),
        yaw_offsets_deg=yaw_offsets,
    )
    config.validate()
    return config


def _robot_config(task: Mapping[str, Any]):
    workspace = task.get("manipulation_workspace") or {}
    spec = workspace.get("robot") or {}
    profile_name = str(spec.get("profile", "split_aloha_tabletop_v1"))
    profile = DEFAULT_ROBOT_PROFILES.get(profile_name)
    if profile is None:
        raise WorkspacePlanningError("ROBOT_PROFILE_NOT_FOUND", f"unknown robot profile: {profile_name}")
    configured = tuple(float(value) for value in spec.get("footprint_m", profile.footprint_m))
    if len(configured) != 2 or any(not math.isfinite(value) or value <= 0 for value in configured):
        raise WorkspacePlanningError("INVALID_ROBOT_FOOTPRINT", f"invalid footprint: {configured}")
    # Arena navigation footprints are allowed to be larger, but never shrink
    # the measured manipulation-spawn envelope from the robot profile.
    footprint = tuple(max(configured[index], profile.footprint_m[index]) for index in range(2))
    return profile_name, footprint, profile


def build_manifest(
    document: Mapping[str, Any],
    input_path: Path,
    target_name: str | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
    required_arm: str | None = None,
) -> WorkspaceManifest:
    if required_arm not in {None, "left", "right"}:
        raise WorkspacePlanningError("INVALID_ARM", f"unsupported workspace arm: {required_arm}")
    task = _one_task(document)
    task_name = str(task.get("name", input_path.parent.name))
    target = _resolve_target(task, target_name)
    asset_rows = audit_assets(task)
    missing_assets = [row["name"] for row in asset_rows if not row.get("usd_exists")]
    invalid_scales = [row["name"] for row in asset_rows if not row.get("scale_valid")]
    if missing_assets:
        raise WorkspacePlanningError(
            "ASSET_NOT_FOUND", "one or more object assets are missing", {"objects": missing_assets}
        )
    if invalid_scales:
        raise WorkspacePlanningError(
            "INVALID_ASSET_SCALE", "one or more object scales are invalid", {"objects": invalid_scales}
        )
    target_asset = next(row for row in asset_rows if row["name"] == target)
    if not target_asset.get("grasp_exists"):
        raise WorkspacePlanningError(
            "TARGET_GRASP_ANNOTATION_MISSING",
            f"target {target!r} has no Aligned_grasp_sparse.npy",
        )
    if not target_asset.get("grasp_shape_valid") or not target_asset.get("grasp_finite"):
        raise WorkspacePlanningError(
            "TARGET_GRASP_ANNOTATION_INVALID",
            f"target {target!r} has an invalid grasp annotation",
            {"grasp_path": target_asset.get("grasp_path")},
        )

    arena_path = _resolve_config_path(str(task.get("arena_file", "")), input_path)
    arena = load_yaml(arena_path)
    fixtures = _fixtures_with_asset_extents(task, list(arena.get("fixtures") or []))
    fixture_audit = [
        {
            "name": str(fixture.get("name", "")),
            "target_class": str(fixture.get("target_class", "")),
            "collision_enabled": bool(
                fixture.get("collision_enabled", fixture.get("collision", True))
            ),
            "size_xy": list(fixture.get("size") or []),
            "size_source": fixture.get("size_source"),
            "size_xyz": list(fixture.get("size_xyz") or []),
            "size_xyz_source": fixture.get("size_xyz_source"),
            "path": fixture.get("path"),
        }
        for fixture in fixtures
    ]
    fixture_by_name = {str(fixture.get("name", "")): fixture for fixture in fixtures}
    floor = fixture_by_name.get("floor")
    if floor is None:
        raise WorkspacePlanningError("WORKSPACE_NOT_FOUND", "arena has no floor fixture")
    target_ref, support_ref = _target_reference(task, target, fixture_by_name)
    sampling = _sampling_config(task, sampling_overrides)
    profile_name, footprint, robot_profile = _robot_config(task)
    collision_layers = robot_profile.collision_layers

    candidates = sample_target_annulus(target_ref["world_xyz"][:2], sampling)
    if sampling.yaw_policy == "align_required_arm":
        if required_arm is None:
            raise WorkspacePlanningError(
                "ARM_REQUIRED_FOR_YAW_POLICY",
                "align_required_arm requires a preselected left or right arm",
            )
        arm_base_xy = (
            robot_profile.left_arm_base_xy_m
            if required_arm == "left"
            else robot_profile.right_arm_base_xy_m
        )
        if arm_base_xy is None:
            raise WorkspacePlanningError(
                "ARM_BASE_OFFSET_MISSING",
                f"robot profile {profile_name} has no {required_arm} arm-base offset",
            )
        for candidate in candidates:
            candidate.yaw_deg = yaw_to_align_arm_base_deg(
                candidate.world_xy,
                target_ref["world_xyz"][:2],
                arm_base_xy,
            ) + candidate.yaw_offset_deg
    for candidate in candidates:
        obstacle = colliding_fixture(candidate, footprint, fixtures)
        home_layer_obstacle = None
        if obstacle is None:
            for layer in collision_layers:
                home_layer_obstacle = colliding_fixture_layer(candidate, layer, fixtures)
                if home_layer_obstacle is not None:
                    break
        obstacle = obstacle or home_layer_obstacle
        candidate.obstacle = obstacle
        candidate.collision_free = obstacle is None
        candidate.inside_floor = inside_floor(candidate, footprint, floor)
        candidate.geometry_feasible = candidate.collision_free and candidate.inside_floor
        if obstacle:
            candidate.rejection_code = (
                "HOME_POSE_COLLISION" if home_layer_obstacle is not None else "BASE_COLLISION"
            )
        elif not candidate.inside_floor:
            candidate.rejection_code = "BASE_OUTSIDE_FLOOR"
    candidates.sort(
        key=lambda item: (
            not item.geometry_feasible,
            abs(item.radius_m - sampling.preferred_radius_m),
            item.candidate_id,
        )
    )
    feasible_count = sum(candidate.geometry_feasible for candidate in candidates)
    return WorkspaceManifest(
        source_task=str(input_path.resolve()),
        task_name=task_name,
        target=target_ref,
        support=support_ref,
        sampling=sampling.to_dict(),
        robot={
            "profile": profile_name,
            "footprint_m": list(footprint),
            "collision_layers": [
                {
                    "name": layer.name,
                    "center_xy_m": list(layer.center_xy_m),
                    "size_xy_m": list(layer.size_xy_m),
                    "min_z_m": layer.min_z_m,
                    "max_z_m": layer.max_z_m,
                }
                for layer in collision_layers
            ],
            "left_arm_base_xy_m": (
                list(robot_profile.left_arm_base_xy_m)
                if robot_profile.left_arm_base_xy_m is not None
                else None
            ),
            "right_arm_base_xy_m": (
                list(robot_profile.right_arm_base_xy_m)
                if robot_profile.right_arm_base_xy_m is not None
                else None
            ),
        },
        geometry_candidates=[candidate.to_dict() for candidate in candidates],
        required_arm=required_arm,
        status="geometry_ready" if feasible_count else "no_geometry_candidate",
        failure_code=None if feasible_count else "NO_GEOMETRY_CANDIDATE",
        asset_audit=asset_rows,
        fixture_audit=fixture_audit,
    )


def apply_candidate_to_document(
    document: Mapping[str, Any], input_path: Path, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(document)
    task = _one_task(output)
    arena_path = _resolve_config_path(str(task.get("arena_file", "")), input_path)
    arena = load_yaml(arena_path)
    floor = next((fixture for fixture in arena.get("fixtures", []) if fixture.get("name") == "floor"), None)
    if floor is None:
        raise WorkspacePlanningError("WORKSPACE_NOT_FOUND", "arena has no floor fixture")
    world_xy = candidate.get("world_xy")
    if not isinstance(world_xy, Sequence) or len(world_xy) < 2:
        raise WorkspacePlanningError("INVALID_CANDIDATE", "candidate has no world_xy")
    yaw_deg = float(candidate["yaw_deg"])
    candidate_id = str(candidate["candidate_id"])

    robots = task.get("robots") or []
    if not robots:
        raise WorkspacePlanningError("ROBOT_NOT_FOUND", "task has no robot")
    robot_name = str(robots[0]["name"])
    robots[0].setdefault("euler", [0.0, 0.0, 0.0])
    while len(robots[0]["euler"]) < 3:
        robots[0]["euler"].append(0.0)
    robots[0]["euler"][2] = round(yaw_deg, 6)

    source_region = next(
        (
            region
            for region in task.get("source_regions", [])
            if region.get("name") == "robot_initial_region" or region.get("A") in {"robot", robot_name}
        ),
        None,
    )
    if source_region is None:
        source_region = {"name": "robot_initial_region", "type": "A_on_B_region_sampler", "A": "robot", "B": "floor"}
        task.setdefault("source_regions", []).append(source_region)
    source_region.update(
        {
            "center": [round(float(world_xy[0]), 6), round(float(world_xy[1]), 6)],
            "center_xyz": [round(float(world_xy[0]), 6), 0.0, round(float(world_xy[1]), 6)],
            "yaw_range": [round(yaw_deg, 6), round(yaw_deg, 6)],
            "planned_by": "target_annulus_v1",
            "candidate_id": candidate_id,
        }
    )
    for obsolete in (
        "pose_source",
        "annotation_name",
        "recommended_arm",
        "interaction_edge",
        "edge_clearance_m",
        "lateral_offset_m",
    ):
        source_region.pop(obsolete, None)

    robot_region = next((region for region in task.get("regions", []) if region.get("object") == robot_name), None)
    if robot_region is None:
        robot_region = {"object": robot_name, "target": "floor", "random_type": "A_on_B_region_sampler"}
        task.setdefault("regions", []).append(robot_region)
    floor_translation = floor.get("translation", [0.0, 0.0, 0.0])
    shift = [
        round(float(world_xy[0]) - float(floor_translation[0]), 6),
        round(float(world_xy[1]) - float(floor_translation[1]), 6),
        0.0,
    ]
    robot_region.update(
        {
            "random_config": {"pos_range": [shift, shift], "yaw_rotation": [0.0, 0.0]},
            "placement_mode": "planned_workspace_pose",
            "candidate_id": candidate_id,
        }
    )
    for obsolete in ("pose_source", "annotation_name"):
        robot_region.pop(obsolete, None)
    _remove_stale_robot_regions(task, robot_name)
    task.setdefault("metadata", {})["workspace_candidate"] = {
        "planner": "target_annulus_v1",
        "candidate_id": candidate_id,
        "world_xy": [float(world_xy[0]), float(world_xy[1])],
        "yaw_deg": yaw_deg,
    }
    return output


def generate_manifest_file(
    input_path: Path,
    output_dir: Path,
    target_name: str | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
    required_arm: str | None = None,
) -> WorkspaceManifest:
    manifest_path = output_dir / "candidates.json"
    try:
        manifest = build_manifest(
            load_yaml(input_path),
            input_path,
            target_name,
            sampling_overrides,
            required_arm,
        )
    except WorkspacePlanningError as exc:
        dump_json(
            {
                "version": 3,
                "source_task": str(input_path.resolve()),
                "task_name": input_path.parent.name,
                "target": {},
                "support": {},
                "sampling": {},
                "robot": {},
                "geometry_candidates": [],
                "required_arm": required_arm,
                "curobo_results": [],
                "pick_attempts": [],
                "selected_candidate": None,
                "status": "blocked",
                "failure_code": exc.code,
                "failure_message": str(exc),
                "failure_details": exc.details,
                "asset_audit": [],
                "fixture_audit": [],
            },
            manifest_path,
        )
        raise
    dump_json(manifest.to_dict(), manifest_path)
    return manifest

def _resolve_table_name(task: Mapping[str, Any]) -> str:
    active = list(task.get("delivery_active_objects") or [])
    target = str(active[0]) if active else None
    if target is None:
        raise WorkspacePlanningError("TABLE_NOT_FOUND", "task has no delivery_active_objects")
    region = next((r for r in task.get("regions", []) if str(r.get("object", "")) == target), None)
    if region is None:
        raise WorkspacePlanningError("TABLE_NOT_FOUND", f"no region for target {target!r}")
    table_name = str(region.get("target") or region.get("B") or "")
    if not table_name:
        raise WorkspacePlanningError("TABLE_NOT_FOUND", f"region for {target!r} has no support target")
    return table_name

def build_tabletop_manifest(
    document: Mapping[str, Any],
    input_path: Path,
    target_name: str | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
) -> WorkspaceManifest:
    task = _one_task(document)
    task_name = str(task.get("name", input_path.parent.name))
    target = _resolve_target(task, target_name)

    asset_rows = audit_assets(task)
    missing_assets = [row["name"] for row in asset_rows if not row.get("usd_exists")]
    invalid_scales = [row["name"] for row in asset_rows if not row.get("scale_valid")]
    if missing_assets:
        raise WorkspacePlanningError(
            "ASSET_NOT_FOUND", "one or more object assets are missing", {"objects": missing_assets}
        )
    if invalid_scales:
        raise WorkspacePlanningError(
            "INVALID_ASSET_SCALE", "one or more object scales are invalid", {"objects": invalid_scales}
        )

    target_asset = next(row for row in asset_rows if row["name"] == target)
    if not target_asset.get("grasp_exists"):
        raise WorkspacePlanningError(
            "TARGET_GRASP_ANNOTATION_MISSING",
            f"target {target!r} has no Aligned_grasp_sparse.npy",
        )
    if not target_asset.get("grasp_shape_valid") or not target_asset.get("grasp_finite"):
        raise WorkspacePlanningError(
            "TARGET_GRASP_ANNOTATION_INVALID",
            f"target {target!r} has an invalid grasp annotation",
            {"grasp_path": target_asset.get("grasp_path")},
        )

    arena_path = _resolve_config_path(str(task.get("arena_file", "")), input_path)
    arena = load_yaml(arena_path)
    fixtures = list(arena.get("fixtures") or [])
    fixture_by_name = {str(fixture.get("name", "")): fixture for fixture in fixtures}

    table_name = _resolve_table_name(task)
    table = fixture_by_name.get(table_name)
    if table is None:
        raise WorkspacePlanningError(
            "WORKSPACE_NOT_FOUND", f"table {table_name!r} is absent from arena fixtures"
        )

    table_translation = table.get("translation", [0.0, 0.0, 0.0])
    table_size = table.get("size", [1.0, 1.0])
    if (
        not isinstance(table_translation, Sequence)
        or len(table_translation) < 2
        or not isinstance(table_size, Sequence)
        or len(table_size) < 2
    ):
        raise WorkspacePlanningError(
            "INVALID_TABLE", f"table {table_name!r} missing translation or size"
        )

    table_euler = table.get("euler", table.get("rotation", [0.0, 0.0, 0.0]))
    table_yaw = float(table_euler[2]) if isinstance(table_euler, Sequence) and len(table_euler) >= 3 else 0.0

    target_ref, support_ref = _target_reference(task, target, fixture_by_name)
    sampling = _sampling_config(task, sampling_overrides)
    profile_name, footprint, _robot_profile = _robot_config(task)

    table_center = (float(table_translation[0]), float(table_translation[1]))
    table_size_tuple = (float(table_size[0]), float(table_size[1]))

    edges = table_edge_centers(table_center, table_size_tuple, table_yaw)
    endpoints = _edge_endpoints(table_center, table_size_tuple, table_yaw)

    candidates_per_edge = max(1, sampling.candidate_count // 4)
    all_candidates: list[GeometryCandidate] = []
    for edge_name, _edge_center, inward_yaw in edges:
        start, end = endpoints[edge_name]
        all_candidates.extend(
            sample_table_edge(edge_name, start, end, inward_yaw, footprint, candidates_per_edge)
        )

    non_table_fixtures = [f for f in fixtures if str(f.get("name", "")) != table_name]

    for candidate in all_candidates:
        candidate.inside_floor = inside_rect(
            candidate, footprint, table_center, table_size_tuple, table_yaw
        )
        obstacle = colliding_fixture(candidate, footprint, non_table_fixtures)
        candidate.obstacle = obstacle
        candidate.collision_free = obstacle is None
        candidate.geometry_feasible = candidate.collision_free and candidate.inside_floor
        if not candidate.inside_floor:
            candidate.rejection_code = "BASE_OUTSIDE_TABLE"
        elif obstacle:
            candidate.rejection_code = "BASE_COLLISION"

    all_candidates.sort(
        key=lambda item: (
            not item.geometry_feasible,
            item.candidate_id,
        )
    )

    feasible_count = sum(candidate.geometry_feasible for candidate in all_candidates)
    return WorkspaceManifest(
        source_task=str(input_path.resolve()),
        task_name=task_name,
        target=target_ref,
        support=support_ref,
        sampling=sampling.to_dict(),
        robot={"profile": profile_name, "footprint_m": list(footprint)},
        geometry_candidates=[candidate.to_dict() for candidate in all_candidates],
        status="geometry_ready" if feasible_count else "no_geometry_candidate",
        failure_code=None if feasible_count else "NO_GEOMETRY_CANDIDATE",
        asset_audit=asset_rows,
    )

def generate_tabletop_manifest_file(
    input_path: Path,
    output_dir: Path,
    target_name: str | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
    *,
    required_arm: str | None = None,
) -> WorkspaceManifest:
    manifest_path = output_dir / "candidates.json"
    try:
        manifest = build_tabletop_manifest(
            load_yaml(input_path), input_path, target_name, sampling_overrides
        )
        if required_arm:
            manifest.required_arm = required_arm
    except WorkspacePlanningError as exc:
        dump_json(
            {
                "version": 3,
                "source_task": str(input_path.resolve()),
                "task_name": input_path.parent.name,
                "target": {},
                "support": {},
                "required_arm": required_arm,
                "sampling": {},
                "robot": {},
                "geometry_candidates": [],
                "curobo_results": [],
                "pick_attempts": [],
                "selected_candidate": None,
                "status": "blocked",
                "failure_code": exc.code,
                "failure_message": str(exc),
                "failure_details": exc.details,
                "asset_audit": [],
            },
            manifest_path,
        )
        raise
    dump_json(manifest.to_dict(), manifest_path)
    return manifest

def apply_tabletop_candidate_to_document(
    document: Mapping[str, Any], input_path: Path, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(document)
    task = _one_task(output)

    world_xy = candidate.get("world_xy")
    if not isinstance(world_xy, Sequence) or len(world_xy) < 2:
        raise WorkspacePlanningError("INVALID_CANDIDATE", "candidate has no world_xy")
    yaw_deg = float(candidate["yaw_deg"])
    candidate_id = str(candidate["candidate_id"])

    robots = task.get("robots") or []
    if not robots:
        raise WorkspacePlanningError("ROBOT_NOT_FOUND", "task has no robot")
    robot_name = str(robots[0]["name"])
    robots[0].setdefault("euler", [0.0, 0.0, 0.0])
    while len(robots[0]["euler"]) < 3:
        robots[0]["euler"].append(0.0)
    robots[0]["euler"][2] = round(yaw_deg, 6)

    table_name = _resolve_table_name(task)

    arena_path = _resolve_config_path(str(task.get("arena_file", "")), input_path)
    arena = load_yaml(arena_path)
    table_fixture = next((f for f in arena.get("fixtures", []) if f.get("name") == table_name), None)
    if table_fixture is None and "__" in table_name:
        fixture_name = table_name.split("__")[0]
        table_fixture = next((f for f in arena.get("fixtures", []) if f.get("name") == fixture_name), None)
    table_translation = table_fixture.get("translation", [0.0, 0.0, 0.0]) if table_fixture else [0.0, 0.0, 0.0]
    shift = [
        round(float(world_xy[0]) - float(table_translation[0]), 6),
        round(float(world_xy[1]) - float(table_translation[1]), 6),
        0.0,
    ]

    source_region = {
        "name": "robot_initial_region",
        "type": "A_on_B_region_sampler",
        "A": "robot",
        "B": table_name,
        "center": [round(float(world_xy[0]), 6), round(float(world_xy[1]), 6)],
        "center_xyz": [round(float(world_xy[0]), 6), 0.0, round(float(world_xy[1]), 6)],
        "yaw_range": [round(yaw_deg, 6), round(yaw_deg, 6)],
        "planned_by": "tabletop_edge_v1",
        "candidate_id": candidate_id,
    }
    task["source_regions"] = [
        r
        for r in task.get("source_regions", [])
        if r.get("name") != "robot_initial_region"
    ]
    task.setdefault("source_regions", []).append(source_region)

    robot_region = next(
        (r for r in task.get("regions", []) if r.get("object") == robot_name),
        None,
    )
    if robot_region is None:
        robot_region = {"object": robot_name, "target": table_name, "random_type": "A_on_B_region_sampler"}
        task.setdefault("regions", []).append(robot_region)
    robot_region.update(
        {
            "target": table_name,
            "random_config": {"pos_range": [shift, shift], "yaw_rotation": [0.0, 0.0]},
            "placement_mode": "planned_tabletop_pose",
            "candidate_id": candidate_id,
        }
    )
    for obsolete in ("pose_source", "annotation_name"):
        robot_region.pop(obsolete, None)
    _remove_stale_robot_regions(task, robot_name)
    task.setdefault("metadata", {})["workspace_candidate"] = {
        "planner": "tabletop_edge_v1",
        "candidate_id": candidate_id,
        "world_xy": [float(world_xy[0]), float(world_xy[1])],
        "yaw_deg": yaw_deg,
    }
    return output
