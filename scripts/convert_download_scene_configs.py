#!/usr/bin/env python3
"""Convert download/<scene>/arena.yaml + task.yaml into SimBox YAMLs.

The converter has two intentionally separate stages:

1. Mechanical conversion from the downloaded schema to the current SimBox
   schema. This is deterministic and runs offline.
2. Optional skill generation through the API settings used by Codex. That stage
   asks a model to translate natural-language task descriptions into SimBox
   `skills` and `positions`, then validates object and goal references before
   merging.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_ROOT = Path("download")
DEFAULT_ARENA_OUT_NAME = "simbox_arena.yaml"
DEFAULT_TASK_OUT_NAME = "simbox_task.yaml"
DEFAULT_ROBOT_NAME = "split_aloha"
DEFAULT_ROBOT_CONFIG_FILE = "workflows/simbox/core/configs/robots/split_aloha.yaml"
DEFAULT_ROBOT_ASSET = Path("InternDataAssets/robots/split_aloha_mid_360/robot.usd")
DEFAULT_ENVMAP_LIB = "../../InternDataAssets/assets/envmap_lib"
DEFAULT_NAV_WAYPOINT_CLEARANCE_M = 0.50
DEFAULT_FLOOR_EDGE_MARGIN_M = 0.35
INTERDATA_SCENE_DIR_NAME = "interndata_scene"

SIMBOX_SUPPORTED_CLASSES = {
    "ArticulatedObject",
    "ConveyorObject",
    "GeometryObject",
    "PlaneObject",
    "RigidObject",
    "ShapeObject",
    "XFormObject",
}

SKILL_SUMMARY = """\
Current SimBox skill config facts from source:

- skills is a list of phases. Each phase maps robot name to a list of
  controller queues, for example:
    - <robot_name_from_context>:
      - base: [...]
        left: [...]
        right: [...]
- DAG mode is enabled when any skill has id or depends_on. In DAG mode every
  skill must have a unique id and depends_on must be a list.
- navigate:
  Required: name: navigate plus either goal or goal_x/goal_y/goal_yaw.
  If goal is used, task.positions[goal] must contain x, y, yaw.
  It belongs on the base queue for mobile-base navigation.
- pick:
  Required: name: pick, objects: [task_object_name].
  The object must be in task.objects and target_class RigidObject.
  Useful optional fields: filter_x_dir/filter_y_dir/filter_z_dir,
  pre_grasp_offset, gripper_change_steps, t_eps, o_eps, process_valid,
  lift_th, post_grasp_offset_min/max, test_mode.
- place:
  Required: name: place, objects: [held_object_name, target_object_or_fixture].
  The second object can be a task object or arena fixture because Place reads
  task._task_objects. Useful fields: place_direction, x_ratio_range,
  y_ratio_range, pre_place_z_offset, place_z_offset, position_constraint,
  success_mode, gripper_change_steps, t_eps, o_eps.
- heuristic__skill:
  Useful for arm home after manipulation. Typical fields: mode: home,
  gripper_state: 1.0. It belongs on left or right queue.

Generate only YAML content with keys `positions`, `skills`, and optionally
`regions`. Do not include Markdown fences or commentary.
"""

SKILL_FORMAT_TEMPLATE = """\
Required output shape must match existing SimBox task YAML exactly:

positions:
  optional_goal_name:
    x: 0.0
    y: 0.0
    yaw: 0.0
skills:
  - <robot_name_from_context>:
      - base:
          - name: navigate
            id: nav_to_pick
            depends_on: []
            goal: existing_or_generated_goal_name
          - name: navigate
            id: nav_to_place
            depends_on: [pick_source_object]
            goal: existing_or_generated_goal_name
        left:
          - name: pick
            id: pick_source_object
            depends_on: [nav_to_pick]
            objects: [source_object_name]
            filter_x_dir: ["backward", 135]
            filter_y_dir: ["downward", 120]
            filter_z_dir: ["forward", 135, 45]
            pre_grasp_offset: 0.05
            gripper_change_steps: 20
            t_eps: 0.025
            o_eps: 1
            process_valid: True
            lift_th: 0.02
            post_grasp_offset_min: 0.05
            post_grasp_offset_max: 0.10
          - name: place
            id: place_source_object
            depends_on: [nav_to_place]
            objects: [source_object_name, destination_object_or_fixture_name]
            position_constraint: object
            filter_x_dir: ["backward", 110]
            filter_y_dir: ["downward", 120]
            filter_z_dir: ["forward", 70]
            x_ratio_range: [0.35, 0.65]
            y_ratio_range: [0.35, 0.65]
            success_mode: xybbox
            pre_place_z_offset: 0.10
            place_z_offset: 0.10
            gripper_change_steps: 20
          - name: heuristic__skill
            id: home_left
            depends_on: [place_source_object]
            mode: home
            gripper_state: 1.0
        right: []

Rules:
- Return a single phase list under `skills`.
- Inside the phase use exactly the robot name from JSON context as the mapping key.
- Inside that robot use one queue mapping containing `base`, `left`, and `right`.
- Put navigate skills only in `base`.
- For mobile manipulation tasks, use DAG ids and depends_on like the repository mobile manipulation example:
  first navigate id `nav_to_pick` with depends_on: [], pick depends_on: [nav_to_pick],
  optional second navigate id `nav_to_place` depends_on: [pick_*], place depends_on: [nav_to_place],
  home depends_on: [place_*].
- Put pick/place/home skills in exactly one arm queue, usually `left`, unless the source task says otherwise.
- Use object and fixture names exactly as listed in JSON context; never invent names.
- Pick is allowed even if has_grasp_annotation is false, because grasp annotation will be added later.
- Pick only movable task objects, not fixtures. Place target may be a fixture or a task object such as a tray, box, basket, holder, coaster, board, shelf, cabinet, desk, nightstand, organizer, or storage object.
- Match the chosen source task semantics: source object, destination object/fixture, and navigation waypoint should correspond to the natural-language steps.
"""


class SimBoxYamlDumper(yaml.SafeDumper):
    pass


def represent_bool(dumper: yaml.Dumper, value: bool) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:bool", "True" if value else "False")


SimBoxYamlDumper.add_representer(bool, represent_bool)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def posix_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is not under asset_root {root}") from exc


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict[str, Any], *, overwrite: bool, dry_run: bool) -> None:
    text = yaml.dump(
        payload,
        Dumper=SimBoxYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if dry_run:
        print(f"[DRY-RUN] Would write {path}")
        print(text)
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[WROTE] {display_path(path)}")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sanitize_name(value: str) -> str:
    value = str(value).replace("__", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "asset"
    if name[0].isdigit():
        name = f"asset_{name}"
    return name


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def xyz_from_download(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    # Download YAML stores layout-like coordinates as [x, y/up, z]. SimBox uses
    # [x, y, z] with z as up, so map to [x, z, y].
    return [float(value[0]), float(value[2]), float(value[1])]


def xyz_from_simbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return [float(value[0]), float(value[1]), float(value[2])]


def is_interdata_scene_source(source_dir: Path) -> bool:
    return source_dir.name == INTERDATA_SCENE_DIR_NAME


def convert_translation(value: Any, source_dir: Path) -> list[float] | None:
    if is_interdata_scene_source(source_dir):
        return xyz_from_simbox(value)
    return xyz_from_download(value)


def vector3(value: Any, default: list[float] | None = None) -> list[float]:
    if value is None:
        if default is None:
            raise ValueError("missing vector")
        return default
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return [float(value[0]), float(value[1]), float(value[2])]


def is_vector3(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) for item in value)


def plane_size(value: Any) -> list[float]:
    if value is None:
        return [1.0, 1.0]
    if not isinstance(value, list) or len(value) not in {2, 3}:
        raise ValueError(f"Expected 2-vector or 3-vector plane size, got {value!r}")
    return [float(value[0]), float(value[1])]


def convert_texture(texture: Any, scene_dir: Path, asset_root: Path) -> Any:
    if not isinstance(texture, dict):
        return texture
    out = {
        key: value
        for key, value in texture.items()
        if key in {"texture_lib", "apply_randomization", "texture_id", "texture_scale", "target_prim_path"}
    }
    texture_lib = out.get("texture_lib")
    if isinstance(texture_lib, str):
        candidate = scene_dir / "texture_libs" / texture_lib
        if candidate.exists():
            out["texture_lib"] = posix_rel(candidate.resolve(), asset_root)
            if not out.get("apply_randomization", False):
                out["texture_id"] = 0
    return out


def copy_metadata(src: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: src[key] for key in keys if key in src and src[key] is not None}


def resolve_asset_path(source_dir: Path, usd_path: str) -> Path:
    candidates = [(source_dir / usd_path).resolve()]
    if source_dir.name == INTERDATA_SCENE_DIR_NAME:
        scene_dir = source_dir.parent
        path_obj = Path(usd_path)
        parts = list(path_obj.parts)
        if ".." in parts:
            rel_parts = [part for part in parts if part != ".."]
            candidates.append((scene_dir / Path(*rel_parts)).resolve())
        if "assets" in parts:
            idx = parts.index("assets")
            candidates.append((scene_dir / Path(*parts[idx:])).resolve())

            if "fixtures" in parts:
                fixture_idx = parts.index("fixtures")
                category = parts[fixture_idx + 1] if fixture_idx + 1 < len(parts) else None
                stem = path_obj.parent.name
                fixture_root = scene_dir / "assets" / "basic" / scene_dir.name / "fixtures"
                aliases = {
                    "closed_door": "door",
                    "fixed_window": "window",
                    "outlet_panel": "kitchen_outlet_panel",
                    "pantry_cabinet": "pantry_tall_cabinet",
                    "wall_cabinet": "wall_cabinet_display",
                }
                search_categories = []
                if category:
                    search_categories.extend([category, aliases.get(category, category)])
                if stem:
                    normalized_stem = re.sub(r"_id\d+$", "", stem)
                    normalized_stem = re.sub(r"_0$", "", normalized_stem)
                    search_categories.append(normalized_stem)
                for search_category in dict.fromkeys(item for item in search_categories if item):
                    candidates.extend(sorted(fixture_root.glob(f"{search_category}/*/Aligned_obj.usd")))
                    candidates.extend(sorted(fixture_root.glob(f"{search_category}*/**/Aligned_obj.usd")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"USD not found for {usd_path!r}; tried: {', '.join(display_path(p) for p in candidates)}")


def is_optional_missing_fixture(src: dict[str, Any]) -> bool:
    if bool(src.get("support_surface", False)):
        return False
    return src.get("asset_category") in {"fixed_window", "closed_door"}


def detect_rigid_prim_path_child(abs_usd: Path, default: str) -> str:
    """Return the child path that carries RigidBodyAPI inside an imported USD."""
    try:
        from pxr import Usd, UsdPhysics  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        print(f"[WARN] pxr unavailable; keep prim_path_child={default!r} for {display_path(abs_usd)}: {exc}")
        return default

    try:
        stage = Usd.Stage.Open(str(abs_usd))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        print(f"[WARN] failed to open USD; keep prim_path_child={default!r} for {display_path(abs_usd)}: {exc}")
        return default
    if stage is None:
        print(f"[WARN] failed to open USD; keep prim_path_child={default!r} for {display_path(abs_usd)}")
        return default

    default_prim = stage.GetDefaultPrim()
    root_path = str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else ""
    candidates: list[str] = []
    for prim in stage.Traverse():
        if not bool(UsdPhysics.RigidBodyAPI(prim)):
            continue
        prim_path = str(prim.GetPath())
        if root_path and prim_path.startswith(f"{root_path}/"):
            prim_path = prim_path[len(root_path) + 1 :]
        elif prim_path.startswith("/"):
            prim_path = prim_path[1:]
        if prim_path:
            candidates.append(prim_path)

    if len(candidates) == 1:
        return candidates[0]

    base_link_candidates = [path for path in candidates if path.split("/")[-1] == "base_link"]
    if len(base_link_candidates) == 1:
        return base_link_candidates[0]

    if not candidates:
        print(f"[WARN] no RigidBodyAPI found; keep prim_path_child={default!r} for {display_path(abs_usd)}")
    else:
        print(
            f"[WARN] ambiguous RigidBodyAPI paths {candidates}; "
            f"keep prim_path_child={default!r} for {display_path(abs_usd)}"
        )
    return default


def convert_fixture(src: dict[str, Any], scene_dir: Path, asset_root: Path) -> dict[str, Any]:
    source_class = src.get("target_class")
    if source_class == "PlaneObject":
        translation = convert_translation(src.get("translation"), scene_dir) or [0.0, 0.0, 0.0]
        size = plane_size(src.get("size"))
        if is_interdata_scene_source(scene_dir) and src.get("name") == "floor" and translation[:2] == [0.0, 0.0]:
            translation = [0.5 * size[0], 0.5 * size[1], translation[2]]
        out: dict[str, Any] = {
            "name": sanitize_name(src["name"]),
            "target_class": "PlaneObject",
            "size": size,
            "translation": translation,
        }
        out["collision_enabled"] = True
        out["collision_thickness"] = 0.02
        euler = src.get("euler") or src.get("rotation")
        if euler is not None:
            out["euler"] = vector3(euler)
        if src.get("texture"):
            out["texture"] = convert_texture(src["texture"], scene_dir, asset_root)
        out.update(copy_metadata(src, ["role", "support_surface", "static"]))
        return out

    if source_class not in {"FixtureObject", "GeometryObject", "TaskObject", "RigidObject", "XFormObject"}:
        raise ValueError(f"Unsupported fixture target_class {source_class!r} for {src.get('name')}")

    usd_path = src.get("path") or src.get("usd_path")
    if not isinstance(usd_path, str):
        raise ValueError(f"Fixture {src.get('name')} missing usd_path/path")
    abs_usd = resolve_asset_path(scene_dir, usd_path)

    out = {
        "name": sanitize_name(src["name"]),
        "path": posix_rel(abs_usd, asset_root),
        "target_class": "GeometryObject",
        "translation": convert_translation(src.get("translation"), scene_dir) or [0.0, 0.0, 0.0],
        "euler": vector3(src.get("euler") or src.get("rotation"), [0.0, 0.0, 0.0]),
        "scale": vector3(src.get("scale"), [1.0, 1.0, 1.0]),
        "collision_enabled": True,
        "collision_approximation": "bbox",
        "collision_visible": False,
    }
    out.update(copy_metadata(src, ["category", "asset_category", "asset_source_mode", "support_surface", "static", "metadata"]))
    return out


def convert_task_object(
    src: dict[str, Any],
    scene_dir: Path,
    asset_root: Path,
    *,
    object_mode: str,
    default_prim_path_child: str,
) -> dict[str, Any]:
    usd_path = src.get("path") or src.get("usd_path")
    if not isinstance(usd_path, str):
        raise ValueError(f"Task object {src.get('name')} missing usd_path/path")
    abs_usd = resolve_asset_path(scene_dir, usd_path)

    target_class = "GeometryObject" if object_mode == "geometry" else "RigidObject"
    out = {
        "name": sanitize_name(src["name"]),
        "path": posix_rel(abs_usd, asset_root),
        "target_class": target_class,
        "translation": convert_translation(src.get("translation"), scene_dir) or [0.0, 0.0, 0.0],
        "euler": vector3(src.get("euler") or src.get("rotation"), [0.0, 0.0, 0.0]),
        "scale": vector3(src.get("scale"), [1.0, 1.0, 1.0]),
        "apply_randomization": False,
    }
    if target_class == "RigidObject":
        out["prim_path_child"] = detect_rigid_prim_path_child(
            abs_usd,
            str(src.get("prim_path_child") or default_prim_path_child),
        )
        if "mass_kg" in src:
            out["mass"] = float(src["mass_kg"])
    out.update(
        copy_metadata(src, ["category", "asset_category", "asset_source_mode", "metadata", "parent_fixture", "spawn_region"])
    )
    return out


def convert_region(region: dict[str, Any]) -> dict[str, Any]:
    out = {
        "name": sanitize_name(region.get("name", "region")),
        "source_type": region.get("type"),
        "center": xyz_from_download([region.get("center", [0.0, 0.0])[0], region.get("height", 0.0), region.get("center", [0.0, 0.0])[1]])
        if isinstance(region.get("center"), list) and len(region["center"]) == 2
        else region.get("center"),
        "size": region.get("size"),
        "height": region.get("height"),
        "sampling": region.get("sampling"),
    }
    return {key: value for key, value in out.items() if value is not None}


def task_with_robot_pose_fallback(scene_dir: Path, task_cfg: dict[str, Any]) -> dict[str, Any]:
    robot_cfg = task_cfg.get("robot") if isinstance(task_cfg.get("robot"), dict) else {}
    if any(key in robot_cfg for key in ("translation", "euler", "quaternion")):
        return task_cfg

    inter_task = scene_dir / "interndata_scene" / "task.yaml"
    if not inter_task.exists():
        return task_cfg

    try:
        fallback_task = load_yaml(inter_task)
    except Exception as exc:  # pragma: no cover - defensive for hand-edited YAML
        print(f"[WARN] failed to read fallback robot pose from {display_path(inter_task)}: {exc}")
        return task_cfg

    fallback_robot = fallback_task.get("robot") if isinstance(fallback_task, dict) else None
    if not isinstance(fallback_robot, dict):
        return task_cfg

    pose_keys = [key for key in ("translation", "euler", "quaternion") if key in fallback_robot]
    if not pose_keys:
        return task_cfg

    merged_task = deepcopy(task_cfg)
    merged_robot = dict(merged_task.get("robot") if isinstance(merged_task.get("robot"), dict) else {})
    for key in pose_keys:
        merged_robot[key] = deepcopy(fallback_robot[key])
    merged_task["robot"] = merged_robot
    print(f"[INFO] Using fallback robot pose from {display_path(inter_task)}")
    return merged_task


def robot_base_euler(robot_cfg: dict[str, Any], robot_name: str) -> list[float]:
    if robot_name == DEFAULT_ROBOT_NAME:
        return [0.0, 0.0, 90.0]
    return vector3(robot_cfg.get("euler"), [0.0, 0.0, 0.0])


def build_robot(
    task_cfg: dict[str, Any],
    robot_name: str,
    include_robot: bool,
    *,
    asset_root_abs: Path,
) -> list[dict[str, Any]]:
    if not include_robot:
        return []
    robot_cfg = task_cfg.get("robot") if isinstance(task_cfg.get("robot"), dict) else {}
    robot_path = robot_cfg.get("path") if robot_name != DEFAULT_ROBOT_NAME else None
    if not robot_path:
        robot_path = os.path.relpath(
            repo_path(DEFAULT_ROBOT_ASSET),
            asset_root_abs,
        )
    robot_config_file = (
        robot_cfg.get("robot_config_file")
        if robot_name != DEFAULT_ROBOT_NAME
        else DEFAULT_ROBOT_CONFIG_FILE
    )
    return [
        {
            "name": robot_name,
            "robot_config_file": robot_config_file or DEFAULT_ROBOT_CONFIG_FILE,
            "path": Path(robot_path).as_posix(),
            "euler": robot_base_euler(robot_cfg, robot_name),
            "ignore_substring": ["material", "table"],
            "use_batch": True,
            "collision_activation_distance": 0.05,
        }
    ]


def build_positions(task_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    robot_cfg = task_cfg.get("robot") if isinstance(task_cfg.get("robot"), dict) else {}
    positions: dict[str, dict[str, float]] = {}
    for waypoint in robot_cfg.get("waypoints", []) or []:
        if not isinstance(waypoint, dict):
            continue
        pose = waypoint.get("pose_xy_yaw")
        if not isinstance(pose, list) or len(pose) != 3:
            continue
        name = sanitize_name(str(waypoint.get("name", "waypoint")).lower())
        positions[name] = {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "yaw": wrap_to_pi(math.radians(float(pose[2]))),
        }
    return positions


def usd_bbox_xy(abs_usd: Path) -> tuple[float, float, float, float] | None:
    try:
        from pxr import Gf, Usd, UsdGeom  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        print(f"[WARN] pxr unavailable; cannot derive fixture bbox for {display_path(abs_usd)}: {exc}")
        return None

    try:
        stage = Usd.Stage.Open(str(abs_usd))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        print(f"[WARN] failed to open fixture USD for bbox {display_path(abs_usd)}: {exc}")
        return None
    if stage is None:
        print(f"[WARN] failed to open fixture USD for bbox {display_path(abs_usd)}")
        return None

    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        root_prim = stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = bbox_cache.ComputeWorldBound(root_prim)
    aligned = Gf.BBox3d(bounds.ComputeAlignedRange()).GetBox()
    min_point = aligned.GetMin()
    max_point = aligned.GetMax()
    values = (float(min_point[0]), float(min_point[1]), float(max_point[0]), float(max_point[1]))
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def fixture_nav_bbox_xy(fixture: dict[str, Any], asset_root: Path) -> tuple[float, float, float, float] | None:
    if fixture.get("target_class") != "GeometryObject":
        return None
    if not bool(fixture.get("collision_enabled", False)):
        return None
    rel_path = fixture.get("path")
    if not isinstance(rel_path, str):
        return None
    translation = fixture.get("translation")
    if not is_vector3(translation):
        return None

    local_bbox = usd_bbox_xy(asset_root / rel_path)
    if local_bbox is None:
        return None

    scale = vector3(fixture.get("scale"), [1.0, 1.0, 1.0])
    min_x, min_y, max_x, max_y = local_bbox
    corners = [
        (min_x * scale[0], min_y * scale[1]),
        (min_x * scale[0], max_y * scale[1]),
        (max_x * scale[0], min_y * scale[1]),
        (max_x * scale[0], max_y * scale[1]),
    ]
    euler = vector3(fixture.get("euler"), [0.0, 0.0, 0.0])
    yaw = math.radians(float(euler[2]))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_corners = [
        (
            float(translation[0]) + x * cos_yaw - y * sin_yaw,
            float(translation[1]) + x * sin_yaw + y * cos_yaw,
        )
        for x, y in corners
    ]
    xs = [point[0] for point in world_corners]
    ys = [point[1] for point in world_corners]
    return (min(xs), min(ys), max(xs), max(ys))


def floor_bounds_xy(arena_payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for fixture in arena_payload.get("fixtures", []):
        if fixture.get("name") != "floor" or fixture.get("target_class") != "PlaneObject":
            continue
        translation = fixture.get("translation")
        size = fixture.get("size")
        if not is_vector3(translation) or not isinstance(size, list) or len(size) < 2:
            return None
        half_x = 0.5 * float(size[0])
        half_y = 0.5 * float(size[1])
        return (
            float(translation[0]) - half_x,
            float(translation[1]) - half_y,
            float(translation[0]) + half_x,
            float(translation[1]) + half_y,
        )
    return None


def point_in_bbox(x: float, y: float, bbox: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    return bbox[0] - margin <= x <= bbox[2] + margin and bbox[1] - margin <= y <= bbox[3] + margin


def distance_to_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    dx = max(float(bbox[0]) - x, 0.0, x - float(bbox[2]))
    dy = max(float(bbox[1]) - y, 0.0, y - float(bbox[3]))
    return math.hypot(dx, dy)


def candidate_position_valid(
    x: float,
    y: float,
    *,
    floor_bounds: tuple[float, float, float, float] | None,
    fixture_bboxes: list[dict[str, Any]],
) -> bool:
    if floor_bounds is not None and not (
        floor_bounds[0] + DEFAULT_FLOOR_EDGE_MARGIN_M <= x <= floor_bounds[2] - DEFAULT_FLOOR_EDGE_MARGIN_M
        and floor_bounds[1] + DEFAULT_FLOOR_EDGE_MARGIN_M <= y <= floor_bounds[3] - DEFAULT_FLOOR_EDGE_MARGIN_M
    ):
        return False
    return not any(point_in_bbox(x, y, fixture["bbox"]) for fixture in fixture_bboxes)


def adjust_position_outside_bbox(
    pose: dict[str, Any],
    bbox: tuple[float, float, float, float],
    *,
    floor_bounds: tuple[float, float, float, float] | None,
    fixture_bboxes: list[dict[str, Any]],
) -> dict[str, float] | None:
    x = float(pose["x"])
    y = float(pose["y"])
    clearance = DEFAULT_NAV_WAYPOINT_CLEARANCE_M
    y_clamped = min(max(y, bbox[1]), bbox[3])
    x_clamped = min(max(x, bbox[0]), bbox[2])
    candidates = [
        (bbox[0] - clearance, y_clamped, 0.0),
        (bbox[2] + clearance, y_clamped, wrap_to_pi(math.pi)),
        (x_clamped, bbox[1] - clearance, wrap_to_pi(0.5 * math.pi)),
        (x_clamped, bbox[3] + clearance, wrap_to_pi(-0.5 * math.pi)),
    ]
    valid_candidates = [
        (cand_x, cand_y, cand_yaw)
        for cand_x, cand_y, cand_yaw in candidates
        if candidate_position_valid(
            cand_x,
            cand_y,
            floor_bounds=floor_bounds,
            fixture_bboxes=fixture_bboxes,
        )
    ]
    if not valid_candidates:
        return None

    def score(candidate: tuple[float, float, float]) -> tuple[float, float]:
        cand_x, cand_y, _cand_yaw = candidate
        nearest_fixture = min(
            (distance_to_bbox(cand_x, cand_y, fixture["bbox"]) for fixture in fixture_bboxes),
            default=0.0,
        )
        distance_from_source = -math.hypot(cand_x - x, cand_y - y)
        return (nearest_fixture, distance_from_source)

    cand_x, cand_y, cand_yaw = max(valid_candidates, key=score)
    return {"x": float(cand_x), "y": float(cand_y), "yaw": float(cand_yaw)}


def adjust_positions_outside_fixture_bboxes(
    task_payload: dict[str, Any],
    arena_payload: dict[str, Any],
    asset_root: Path,
) -> None:
    tasks = task_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        return
    positions = tasks[0].get("positions")
    if not isinstance(positions, dict):
        return

    fixture_bboxes: list[dict[str, Any]] = []
    for fixture in arena_payload.get("fixtures", []):
        if not isinstance(fixture, dict):
            continue
        bbox = fixture_nav_bbox_xy(fixture, asset_root)
        if bbox is not None:
            fixture_bboxes.append({"name": fixture.get("name"), "bbox": bbox})
    if not fixture_bboxes:
        return

    floor_bounds = floor_bounds_xy(arena_payload)
    for position_name, pose in positions.items():
        if not isinstance(pose, dict) or "x" not in pose or "y" not in pose:
            continue
        x = float(pose["x"])
        y = float(pose["y"])
        blocking_fixture = next(
            (fixture for fixture in fixture_bboxes if point_in_bbox(x, y, fixture["bbox"])),
            None,
        )
        if blocking_fixture is None:
            continue

        adjusted = adjust_position_outside_bbox(
            pose,
            blocking_fixture["bbox"],
            floor_bounds=floor_bounds,
            fixture_bboxes=fixture_bboxes,
        )
        if adjusted is None:
            print(
                f"[WARN] position {position_name!r} remains inside fixture "
                f"{blocking_fixture['name']!r}; no valid mechanical offset found"
            )
            continue
        print(
            f"[ADJUST] position {position_name!r} moved out of fixture {blocking_fixture['name']!r}: "
            f"({x:.3f}, {y:.3f}) -> ({adjusted['x']:.3f}, {adjusted['y']:.3f})"
        )
        pose.update(adjusted)


def build_robot_spawn_regions(
    task_cfg: dict[str, Any],
    arena_payload: dict[str, Any],
    *,
    robot_name: str,
    include_robot: bool,
) -> list[dict[str, Any]]:
    if not include_robot:
        return []

    robot_cfg = task_cfg.get("robot") if isinstance(task_cfg.get("robot"), dict) else {}
    floor_bounds = floor_bounds_xy(arena_payload)
    if floor_bounds is None:
        print("[WARN] floor fixture not found; cannot build robot-on-floor spawn region")
        return []

    start_pose: tuple[float, float, float] | None = None
    for waypoint in robot_cfg.get("waypoints", []) or []:
        if not isinstance(waypoint, dict):
            continue
        pose = waypoint.get("pose_xy_yaw")
        if isinstance(pose, list) and len(pose) == 3:
            start_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
            break

    if start_pose is None and is_vector3(robot_cfg.get("translation")):
        euler = vector3(robot_cfg.get("euler"), robot_base_euler(robot_cfg, robot_name))
        translation = robot_cfg["translation"]
        start_pose = (float(translation[0]), float(translation[1]), float(euler[2]))

    if start_pose is None:
        return []

    floor_center_x = 0.5 * (floor_bounds[0] + floor_bounds[2])
    floor_center_y = 0.5 * (floor_bounds[1] + floor_bounds[3])
    start_x, start_y, start_yaw = start_pose
    shift = [float(start_x - floor_center_x), float(start_y - floor_center_y), 0.0]
    yaw_shift = float(start_yaw - robot_base_euler(robot_cfg, robot_name)[2])
    return [
        {
            "object": robot_name,
            "target": "floor",
            "random_type": "A_on_B_region_sampler",
            "random_config": {
                "pos_range": [shift, list(shift)],
                "yaw_rotation": [yaw_shift, yaw_shift],
            },
        }
    ]


def build_scene_only_skills(robot_name: str, include_robot: bool) -> list[dict[str, Any]]:
    if not include_robot:
        return []
    return [{robot_name: [{"base": [], "left": [], "right": []}]}]


def build_arena_payload(arena_cfg: dict[str, Any], scene_dir: Path, asset_root: Path) -> dict[str, Any]:
    scene_name = scene_dir.parent.name if scene_dir.name == INTERDATA_SCENE_DIR_NAME else scene_dir.name
    fixtures = []
    for item in arena_cfg.get("fixtures", []):
        try:
            fixtures.append(convert_fixture(item, scene_dir, asset_root))
        except FileNotFoundError:
            if is_optional_missing_fixture(item):
                print(f"[SKIP] optional missing fixture {item.get('name')!r}")
                continue
            raise
    return {
        "name": sanitize_name(arena_cfg.get("name", scene_name) if not is_interdata_scene_source(scene_dir) else scene_name),
        "fixtures": fixtures,
    }


def build_task_payload(
    scene_dir: Path,
    task_cfg: dict[str, Any],
    arena_payload: dict[str, Any],
    arena_out_path: Path,
    *,
    asset_root_cfg: Path,
    asset_root_abs: Path,
    object_mode: str,
    default_prim_path_child: str,
    include_robot: bool,
    robot_name: str,
    max_episode_length: int,
) -> dict[str, Any]:
    scene_name = scene_dir.parent.name if scene_dir.name == INTERDATA_SCENE_DIR_NAME else scene_dir.name
    source_objects = task_cfg.get("task_objects")
    if not isinstance(source_objects, list):
        source_objects = task_cfg.get("objects")
    if not isinstance(source_objects, list):
        source_objects = []

    objects = [
        convert_task_object(
            item,
            scene_dir,
            asset_root_abs,
            object_mode=object_mode,
            default_prim_path_child=default_prim_path_child,
        )
        for item in source_objects
    ]
    source_regions = task_cfg.get("regions") if isinstance(task_cfg.get("regions"), list) else []
    positions = build_positions(task_cfg)
    task_name = sanitize_name(task_cfg.get("name", f"{scene_name}_task") if not is_interdata_scene_source(scene_dir) else f"{scene_name}_task")
    arena_file_rel = display_path(arena_out_path)

    task = {
        "name": task_name,
        "asset_root": asset_root_cfg.as_posix(),
        "task": "BananaBaseTask",
        "task_id": 0,
        "offset": None,
        "render": True,
        "arena_file": arena_file_rel,
        "env_map": {
            "envmap_lib": DEFAULT_ENVMAP_LIB,
            "apply_randomization": False,
            "intensity_range": [5000, 5000],
            "rotation_range": [0, 0],
        },
        "robots": build_robot(task_cfg, robot_name, include_robot, asset_root_abs=asset_root_abs),
        "objects": objects,
        "regions": build_robot_spawn_regions(
            task_cfg,
            arena_payload,
            robot_name=robot_name,
            include_robot=include_robot,
        ),
        "cameras": [],
        "data": {
            "task_dir": f"download/{scene_name}",
            "language_instruction": f"Load converted download scene {scene_name}.",
            "detailed_language_instruction": (
                f"Load converted download scene {scene_name}. Natural-language task steps are stored "
                "under source_tasks and can be converted with --generate-skills-with-api."
            ),
            "collect_info": f"download_{scene_name}",
            "version": "v1.0",
            "update": True,
            "max_episode_length": max_episode_length,
        },
        "positions": positions,
        "skills": build_scene_only_skills(robot_name, include_robot),
        "source_regions": [convert_region(region) for region in source_regions],
        "source_tasks": load_source_tasks(scene_dir),
    }
    return {"tasks": [task]}


def load_source_tasks(scene_dir: Path) -> list[dict[str, Any]]:
    if scene_dir.name == INTERDATA_SCENE_DIR_NAME:
        data = load_yaml(scene_dir / "task.yaml")
        tasks = data.get("tasks") if isinstance(data, dict) else []
        return tasks if isinstance(tasks, list) else []
    inter_task = scene_dir / "interndata_scene" / "task.yaml"
    if not inter_task.exists():
        return []
    data = load_yaml(inter_task)
    tasks = data.get("tasks") if isinstance(data, dict) else []
    return tasks if isinstance(tasks, list) else []


def validate_arena_payload(payload: dict[str, Any], asset_root: Path) -> list[str]:
    errors: list[str] = []
    for idx, fixture in enumerate(payload.get("fixtures", [])):
        cls = fixture.get("target_class")
        if cls not in SIMBOX_SUPPORTED_CLASSES:
            errors.append(f"arena.fixtures[{idx}] unsupported target_class {cls!r}")
        if cls == "PlaneObject":
            if fixture.get("collision_enabled") is not True:
                errors.append(f"arena.fixtures[{idx}] PlaneObject collision_enabled must be true")
            thickness = fixture.get("collision_thickness")
            if not isinstance(thickness, (int, float)) or thickness <= 0:
                errors.append(f"arena.fixtures[{idx}] PlaneObject collision_thickness must be positive")
        if cls == "GeometryObject":
            if fixture.get("collision_enabled") is not True:
                errors.append(f"arena.fixtures[{idx}] GeometryObject collision_enabled must be true")
            if fixture.get("collision_approximation") != "bbox":
                errors.append(f"arena.fixtures[{idx}] GeometryObject collision_approximation must be 'bbox'")
        rel_path = fixture.get("path")
        if rel_path and not (asset_root / rel_path).exists():
            errors.append(f"arena.fixtures[{idx}] path not found: {rel_path}")
    return errors


def iter_skill_cfgs(skills: list[Any]):
    for phase in skills:
        if not isinstance(phase, dict):
            continue
        for _robot_name, queues in phase.items():
            if not isinstance(queues, list):
                continue
            for queue_dict in queues:
                if not isinstance(queue_dict, dict):
                    continue
                for _controller, skill_list in queue_dict.items():
                    if not isinstance(skill_list, list):
                        continue
                    for skill in skill_list:
                        if isinstance(skill, dict):
                            yield skill


def validate_task_payload(payload: dict[str, Any], asset_root: Path, arena_payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    task = payload["tasks"][0]
    object_names = {obj["name"] for obj in task.get("objects", [])}
    fixture_names = {fixture["name"] for fixture in arena_payload.get("fixtures", [])}
    robot_names = {robot["name"] for robot in task.get("robots", [])}
    position_names = set((task.get("positions") or {}).keys())
    all_object_names = object_names | fixture_names
    region_object_names = all_object_names | robot_names
    object_info = {obj["name"]: obj for obj in task.get("objects", [])}

    for idx, obj in enumerate(task.get("objects", [])):
        cls = obj.get("target_class")
        if cls not in SIMBOX_SUPPORTED_CLASSES:
            errors.append(f"objects[{idx}] unsupported target_class {cls!r}")
        rel_path = obj.get("path")
        if rel_path and not (asset_root / rel_path).exists():
            errors.append(f"objects[{idx}] path not found: {rel_path}")
        if cls == "RigidObject" and not obj.get("prim_path_child"):
            errors.append(f"objects[{idx}] RigidObject missing prim_path_child")

    for idx, region in enumerate(task.get("regions", [])):
        if region.get("object") not in region_object_names:
            errors.append(f"regions[{idx}] unknown object {region.get('object')!r}")
        if region.get("target") not in all_object_names:
            errors.append(f"regions[{idx}] unknown target {region.get('target')!r}")

    ids: set[str] = set()
    for skill in iter_skill_cfgs(task.get("skills", [])):
        skill_id = skill.get("id")
        if skill_id:
            skill_id = str(skill_id)
            if skill_id in ids:
                errors.append(f"duplicate skill id {skill_id!r}")
            ids.add(skill_id)
        for obj_name in skill.get("objects", []) or []:
            if obj_name not in all_object_names:
                errors.append(f"skill {skill.get('name')!r} references unknown object {obj_name!r}")
        if skill.get("name") == "pick":
            obj_name = (skill.get("objects") or [None])[0]
            if obj_name in object_info:
                obj = object_info[obj_name]
                if obj.get("target_class") != "RigidObject":
                    errors.append(f"pick skill references non-rigid object {obj_name!r}")
        if skill.get("name") == "navigate" and skill.get("goal") and skill.get("goal") not in position_names:
            errors.append(f"navigate skill references unknown goal {skill.get('goal')!r}")
    for skill in iter_skill_cfgs(task.get("skills", [])):
        for dep_id in skill.get("depends_on", []) or []:
            if str(dep_id) not in ids:
                errors.append(f"skill {skill.get('id')!r} depends on unknown id {dep_id!r}")
    return errors


def build_llm_context(
    scene_dir: Path,
    arena_payload: dict[str, Any],
    task_payload: dict[str, Any],
) -> dict[str, Any]:
    task = task_payload["tasks"][0]
    return {
        "scene": scene_dir.name,
        "robot_name": task["robots"][0]["name"] if task.get("robots") else DEFAULT_ROBOT_NAME,
        "objects": [
            {
                "name": obj.get("name"),
                "category": obj.get("category"),
                "target_class": obj.get("target_class"),
                "path": obj.get("path"),
                "has_grasp_annotation": (repo_path(Path(task["asset_root"])) / obj.get("path", "")).with_name(
                    "Aligned_grasp_sparse.npy"
                ).exists(),
            }
            for obj in task.get("objects", [])
        ],
        "fixtures": [
            {
                "name": fixture.get("name"),
                "category": fixture.get("category"),
                "support_surface": fixture.get("support_surface"),
                "target_class": fixture.get("target_class"),
            }
            for fixture in arena_payload.get("fixtures", [])
        ],
        "positions": task.get("positions", {}),
        "source_tasks": task.get("source_tasks", []),
    }


def extract_yaml_mapping(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    data = yaml.safe_load(stripped)
    if not isinstance(data, dict):
        raise ValueError("Codex output is not a YAML mapping")
    allowed = {"positions", "skills", "regions"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Codex output has unsupported top-level keys: {sorted(unknown)}")
    return data


def load_toml(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib

        with path.open("rb") as handle:
            return tomllib.load(handle)
    try:
        import tomli
    except ImportError:
        return load_minimal_toml(path)
    with path.open("rb") as handle:
        return tomli.load(handle)


def parse_minimal_toml_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    return value


def load_minimal_toml(path: Path) -> dict[str, Any]:
    """Parse the small subset of TOML needed from Codex config.toml."""
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section.startswith("model_providers."):
                provider = section.removeprefix("model_providers.").strip('"')
                current = data.setdefault("model_providers", {}).setdefault(provider, {})
            else:
                current = {}
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip().strip('"')] = parse_minimal_toml_value(value)
    return data


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def read_codex_api_settings(
    *,
    config_path: Path | None,
    auth_path: Path | None,
    provider_name: str | None,
    model: str | None,
) -> dict[str, str]:
    home = codex_home()
    config = load_toml(config_path or home / "config.toml")
    auth_file = auth_path or home / "auth.json"
    auth = json.loads(auth_file.read_text(encoding="utf-8"))

    selected_provider = provider_name or config.get("model_provider")
    if not selected_provider:
        raise ValueError("Codex config missing model_provider; pass --api-provider")
    providers = config.get("model_providers") or {}
    provider_cfg = providers.get(selected_provider)
    if not isinstance(provider_cfg, dict):
        raise ValueError(f"Codex config missing model_providers.{selected_provider}")
    base_url = provider_cfg.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"Codex provider {selected_provider!r} missing base_url")

    api_key = os.environ.get("OPENAI_API_KEY") or auth.get("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError(f"Codex auth file {auth_file} missing OPENAI_API_KEY")

    return {
        "provider": selected_provider,
        "base_url": base_url.rstrip("/"),
        "wire_api": str(provider_cfg.get("wire_api") or "chat"),
        "model": model or str(config.get("model") or "gpt-5.5"),
        "api_key": api_key,
    }


def request_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "codex-api-config-converter/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body}") from exc
    return json.loads(response_body)


def extract_text_from_api_response(response: dict[str, Any]) -> str:
    if isinstance(response.get("choices"), list) and response["choices"]:
        message = response["choices"][0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(parts)

    output = response.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)

    text = response.get("output_text")
    if isinstance(text, str):
        return text
    raise ValueError(f"Could not extract text from API response keys: {sorted(response.keys())}")


def run_api_skill_generation(
    scene_dir: Path,
    arena_payload: dict[str, Any],
    task_payload: dict[str, Any],
    *,
    api_config: Path | None,
    api_auth: Path | None,
    api_provider: str | None,
    api_model: str | None,
    api_timeout: float,
    prompt_out_dir: Path | None,
) -> dict[str, Any]:
    settings = read_codex_api_settings(
        config_path=api_config,
        auth_path=api_auth,
        provider_name=api_provider,
        model=api_model,
    )
    context = build_llm_context(scene_dir, arena_payload, task_payload)
    prompt = (
        "You are generating SimBox task YAML fragments from a downloaded scene description.\n"
        "Follow the source-grounded skill facts exactly.\n"
        "Only use object names, fixture names, and position goal names present in the JSON context.\n"
        "Generate pick/place when the source task semantics call for grasping or moving an object.\n"
        "Prefer the first source_tasks entry that can be represented with available objects and fixtures.\n"
        "The generated skills must be directly usable YAML, with structure matching the template.\n\n"
        f"{SKILL_SUMMARY}\n\n"
        f"{SKILL_FORMAT_TEMPLATE}\n\n"
        "JSON context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
    )

    if prompt_out_dir is not None:
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_out_dir / f"{scene_dir.name}_skill_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

    if settings["wire_api"] == "responses":
        url = f"{settings['base_url']}/responses"
        payload = {
            "model": settings["model"],
            "input": prompt,
        }
    else:
        url = f"{settings['base_url']}/chat/completions"
        payload = {
            "model": settings["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

    response = request_json(url, payload, settings["api_key"], api_timeout)
    raw = extract_text_from_api_response(response)
    if prompt_out_dir is not None:
        raw_path = prompt_out_dir / f"{scene_dir.name}_skill_raw.yaml"
        raw_path.write_text(raw, encoding="utf-8")
    print(
        f"[API] Generated skill fragment for {scene_dir.name} via provider "
        f"{settings['provider']} ({settings['base_url']})"
    )
    return extract_yaml_mapping(raw)


def merge_llm_fragment(task_payload: dict[str, Any], fragment: dict[str, Any]) -> None:
    task = task_payload["tasks"][0]
    robot_name = task["robots"][0]["name"] if task.get("robots") else DEFAULT_ROBOT_NAME
    if "positions" in fragment and fragment["positions"] is not None:
        positions = task.setdefault("positions", {})
        for pose in fragment["positions"].values():
            if isinstance(pose, dict) and "yaw" in pose:
                pose["yaw"] = wrap_to_pi(float(pose["yaw"]))
        positions.update(fragment["positions"])
    if "regions" in fragment and fragment["regions"] is not None:
        regions = task.setdefault("regions", [])
        seen = {json.dumps(region, sort_keys=True) for region in regions if isinstance(region, dict)}
        for region in fragment["regions"]:
            if not isinstance(region, dict):
                regions.append(region)
                continue
            key = json.dumps(region, sort_keys=True)
            if key not in seen:
                regions.append(region)
                seen.add(key)
    if "skills" in fragment and fragment["skills"] is not None:
        for phase in fragment["skills"]:
            if isinstance(phase, dict) and robot_name not in phase and len(phase) == 1:
                queues = next(iter(phase.values()))
                phase.clear()
                phase[robot_name] = queues
        task["skills"] = fragment["skills"]
        task["data"]["language_instruction"] = f"Generated task for {task['name']} from natural-language source steps."
        task["data"]["detailed_language_instruction"] = (
            "Skills were generated by Codex from source_tasks. Validate grasp annotations and runtime feasibility "
            "before large-scale use."
        )


def preserve_generated_task_content(task_payload: dict[str, Any], existing_task_path: Path) -> None:
    if not existing_task_path.exists():
        return
    try:
        existing_payload = load_yaml(existing_task_path)
    except Exception as exc:  # pragma: no cover - defensive for hand-edited YAML
        print(f"[WARN] failed to preserve existing skills from {display_path(existing_task_path)}: {exc}")
        return
    if not isinstance(existing_payload, dict):
        return
    existing_tasks = existing_payload.get("tasks")
    if not isinstance(existing_tasks, list) or not existing_tasks or not isinstance(existing_tasks[0], dict):
        return

    existing_task = existing_tasks[0]
    task = task_payload["tasks"][0]
    for key in ("regions", "cameras"):
        existing_value = existing_task.get(key)
        if existing_value is not None:
            task[key] = existing_value
    existing_positions = existing_task.get("positions")
    if isinstance(existing_positions, dict):
        task["positions"].update(existing_positions)
    existing_skills = existing_task.get("skills")
    if isinstance(existing_skills, list) and existing_skills:
        task["skills"] = existing_skills

    existing_data = existing_task.get("data")
    if isinstance(existing_data, dict):
        for key in ("language_instruction", "detailed_language_instruction"):
            if key in existing_data:
                task["data"][key] = existing_data[key]


def discover_scene_dirs(args: argparse.Namespace) -> list[Path]:
    if args.scene_dir:
        return [repo_path(Path(path)) for path in args.scene_dir]
    root = repo_path(args.download_root)
    if args.all:
        return sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".") and (path / "arena.yaml").exists() and (path / "task.yaml").exists()
        )
    raise SystemExit("Pass --scene-dir PATH or --all")


def select_source_yaml_dir(scene_dir: Path) -> Path:
    interdata_dir = scene_dir / INTERDATA_SCENE_DIR_NAME
    if (interdata_dir / "arena.yaml").exists() and (interdata_dir / "task.yaml").exists():
        return interdata_dir
    return scene_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert download/<scene>/arena.yaml and task.yaml to current SimBox config YAML."
    )
    parser.add_argument("--scene-dir", action="append", help="Scene directory such as download/01_kitchen.")
    parser.add_argument("--all", action="store_true", help="Convert every complete scene under --download-root.")
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument(
        "--output-arena-dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted arena YAMLs. Defaults to each source scene directory "
            f"using {DEFAULT_ARENA_OUT_NAME}."
        ),
    )
    parser.add_argument(
        "--output-task-dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted task YAMLs. Defaults to each source scene directory "
            f"using {DEFAULT_TASK_OUT_NAME}."
        ),
    )
    parser.add_argument("--output-arena-name", default=DEFAULT_ARENA_OUT_NAME)
    parser.add_argument("--output-task-name", default=DEFAULT_TASK_OUT_NAME)
    parser.add_argument(
        "--asset-root-mode",
        choices=["scene", "download-root"],
        default="scene",
        help="Use each scene dir or the whole download root as SimBox asset_root.",
    )
    parser.add_argument(
        "--object-mode",
        choices=["geometry", "rigid"],
        default="geometry",
        help="Mechanical conversion mode for task_objects. geometry is load-only; rigid is manipulation template.",
    )
    parser.add_argument("--default-prim-path-child", default="Aligned")
    parser.add_argument("--robot-name", default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--no-robot", action="store_true", help="Generate a pure scene/object load config with no robot.")
    parser.add_argument("--max-episode-length", type=int, default=1000)
    parser.add_argument(
        "--generate-skills-with-api",
        action="store_true",
        help="Generate positions/skills by directly calling the API configured for Codex.",
    )
    parser.add_argument(
        "--reset-generated-skills",
        action="store_true",
        help="Do not preserve positions/skills from an existing converted task output.",
    )
    parser.add_argument(
        "--generate-skills-with-codex",
        action="store_true",
        help="Deprecated alias for --generate-skills-with-api; no Codex CLI process is used.",
    )
    parser.add_argument("--api-config", type=Path, help="Codex config.toml path. Defaults to $CODEX_HOME/config.toml.")
    parser.add_argument("--api-auth", type=Path, help="Codex auth.json path. Defaults to $CODEX_HOME/auth.json.")
    parser.add_argument("--api-provider", help="Codex model_providers key. Defaults to model_provider in config.toml.")
    parser.add_argument("--api-model", help="Model name. Defaults to model in config.toml.")
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--codex-model", dest="api_model", help=argparse.SUPPRESS)
    parser.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--codex-extra-arg",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-artifact-dir",
        type=Path,
        default=Path("output/download_scene_conversion/llm"),
        help="Directory for API prompts and raw outputs when skill generation is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def convert_scene(args: argparse.Namespace, scene_dir: Path) -> tuple[Path, Path]:
    source_dir = select_source_yaml_dir(scene_dir)
    arena_path = source_dir / "arena.yaml"
    task_path = source_dir / "task.yaml"
    if not arena_path.exists() or not task_path.exists():
        raise FileNotFoundError(f"{source_dir} must contain arena.yaml and task.yaml")

    arena_cfg = load_yaml(arena_path)
    task_cfg = load_yaml(task_path)
    if not isinstance(arena_cfg, dict) or not isinstance(task_cfg, dict):
        raise ValueError(f"{source_dir} arena/task YAML must be mappings")
    task_cfg = task_with_robot_pose_fallback(scene_dir, task_cfg)

    if args.output_arena_dir is None:
        output_arena = scene_dir / args.output_arena_name
    else:
        output_arena = repo_path(args.output_arena_dir) / f"{scene_dir.name}_arena.yaml"
    if args.output_task_dir is None:
        output_task = scene_dir / args.output_task_name
    else:
        output_task = repo_path(args.output_task_dir) / f"{scene_dir.name}_task.yaml"

    if args.asset_root_mode == "scene":
        asset_root_cfg = scene_dir.relative_to(REPO_ROOT) if scene_dir.is_absolute() else scene_dir
    else:
        root = repo_path(args.download_root)
        asset_root_cfg = root.relative_to(REPO_ROOT) if root.is_absolute() else root
    asset_root_abs = repo_path(asset_root_cfg)

    if source_dir != scene_dir:
        print(f"[SOURCE] {display_path(scene_dir)} uses {display_path(source_dir)} as task-ready YAML source")

    arena_payload = build_arena_payload(arena_cfg, source_dir, asset_root_abs)
    task_payload = build_task_payload(
        source_dir,
        task_cfg,
        arena_payload,
        output_arena,
        asset_root_cfg=asset_root_cfg,
        asset_root_abs=asset_root_abs,
        object_mode=args.object_mode,
        default_prim_path_child=args.default_prim_path_child,
        include_robot=not args.no_robot,
        robot_name=args.robot_name,
        max_episode_length=args.max_episode_length,
    )

    if not args.reset_generated_skills and not (args.generate_skills_with_api or args.generate_skills_with_codex):
        preserve_generated_task_content(task_payload, output_task)
    adjust_positions_outside_fixture_bboxes(task_payload, arena_payload, asset_root_abs)

    errors = validate_arena_payload(arena_payload, asset_root_abs)
    errors.extend(validate_task_payload(task_payload, asset_root_abs, arena_payload))
    if errors:
        raise ValueError(f"Mechanical conversion validation failed for {scene_dir}:\n- " + "\n- ".join(errors))

    if args.generate_skills_with_api or args.generate_skills_with_codex:
        fragment = run_api_skill_generation(
            scene_dir,
            arena_payload,
            task_payload,
            api_config=args.api_config,
            api_auth=args.api_auth,
            api_provider=args.api_provider,
            api_model=args.api_model,
            api_timeout=args.api_timeout,
            prompt_out_dir=repo_path(args.llm_artifact_dir),
        )
        merge_llm_fragment(task_payload, fragment)
        adjust_positions_outside_fixture_bboxes(task_payload, arena_payload, asset_root_abs)
        errors = validate_task_payload(task_payload, asset_root_abs, arena_payload)
        if errors:
            raise ValueError(f"Codex-generated fragment validation failed for {scene_dir}:\n- " + "\n- ".join(errors))

    write_yaml(output_arena, arena_payload, overwrite=args.overwrite, dry_run=args.dry_run)
    write_yaml(output_task, task_payload, overwrite=args.overwrite, dry_run=args.dry_run)
    return output_arena, output_task


def main() -> int:
    args = parse_args()
    scene_dirs = discover_scene_dirs(args)
    for scene_dir in scene_dirs:
        convert_scene(args, scene_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
