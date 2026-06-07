#!/usr/bin/env python3
"""Standalone headless renderer for existing Interndata scene YAMLs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any
import zlib

from isaacsim import SimulationApp
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fixed offline views from an existing task YAML.")
    parser.add_argument("--task", required=True, help="Path to download task.yaml or simbox_task.yaml.")
    parser.add_argument("--output-dir", default=os.environ.get("TASK_RENDER_OUTPUT_DIR", "outputs/task_views"))
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Skip robots from compatible task config. Robots are loaded by default.",
    )
    parser.add_argument("--include-robot", dest="no_robot", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=int(os.environ.get("TASK_RENDER_WIDTH", "2560")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("TASK_RENDER_HEIGHT", "1440")))
    parser.add_argument("--rt-subframes", type=int, default=int(os.environ.get("TASK_RENDER_RT_SUBFRAMES", "32")))
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=float(os.environ.get("TASK_RENDER_SETTLE_SECONDS", "1.0")),
        help="Physics simulation time to run after loading the scene before capturing views.",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        default=os.environ.get("TASK_RENDER_NO_PHYSICS", "").lower() in {"1", "true", "yes"},
        help="Load all assets as visual-only references without adding renderer collision proxies.",
    )
    parser.add_argument(
        "--no-snap-to-supports",
        action="store_true",
        default=os.environ.get("TASK_RENDER_NO_SNAP_TO_SUPPORTS", "").lower() in {"1", "true", "yes"},
        help="Disable visual bottom-to-support alignment from source candidate regions.",
    )
    parser.add_argument("--renderer", default=os.environ.get("TASK_RENDER_RENDERER", "RayTracedLighting"))
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    return args


ARGS = _parse_args()
DEFAULT_EXPERIENCE = "/isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit"
EXPERIENCE = os.environ.get("TASK_RENDER_EXPERIENCE", DEFAULT_EXPERIENCE)
LAUNCH_CONFIG = {
    "headless": True,
    "width": int(ARGS.width),
    "height": int(ARGS.height),
    "renderer": ARGS.renderer,
}
if EXPERIENCE and Path(EXPERIENCE).is_file():
    SIMULATION_APP = SimulationApp(LAUNCH_CONFIG, experience=EXPERIENCE)
else:
    SIMULATION_APP = SimulationApp(LAUNCH_CONFIG)


import carb  # noqa: E402  pylint: disable=wrong-import-position
import omni.usd  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core import World  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402  pylint: disable=wrong-import-position
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402  pylint: disable=wrong-import-position

from webrtc.config_compat import load_scene_config  # noqa: E402  pylint: disable=wrong-import-position


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "asset"


def _repo_path(path: str | Path, base: Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if base is not None:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path_obj).resolve()


def _asset_key(path: str | Path | None) -> str | None:
    if path is None:
        return None
    normalized = str(path).replace("\\", "/")
    marker = "assets/"
    if marker not in normalized:
        return normalized.lstrip("./")
    return normalized[normalized.index(marker) :]


def _semantic_instance_key(name: Any) -> str:
    safe = _safe_name(str(name))
    parts = [part for part in re.split(r"_+", safe) if part]
    semantic_parts = [part for part in parts if not part.isdigit() and not re.fullmatch(r"id\d+", part)]
    return "_".join(semantic_parts) or safe


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def _scale3(value: Any) -> tuple[float, float, float]:
    if isinstance(value, list) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (1.0, 1.0, 1.0)


def _apply_xform(prim, translation, euler_deg, scale=(1.0, 1.0, 1.0)) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*euler_deg))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _local_bbox(prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bbox = bbox_cache.ComputeLocalBound(prim).ComputeAlignedBox()
    size = bbox.GetSize()
    if min(float(size[0]), float(size[1]), float(size[2])) <= 0.0:
        return None
    return bbox.GetMidpoint(), size


def _world_bbox(prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    size = bbox.GetSize()
    if min(float(size[0]), float(size[1]), float(size[2])) <= 0.0:
        return None
    return bbox.GetMin(), bbox.GetMax()


def _snap_translation_to_support(
    prim,
    translation: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
    support_z: float | None,
) -> tuple[float, float, float]:
    if support_z is None:
        return translation
    bbox = _world_bbox(prim)
    if bbox is None:
        return translation
    bbox_min, _bbox_max = bbox
    current_bottom_z = float(bbox_min[2])
    if not math.isfinite(current_bottom_z):
        return translation
    clearance = 0.001
    delta_z = float(support_z) + clearance - current_bottom_z
    if abs(delta_z) < 1e-5:
        return translation
    snapped = (translation[0], translation[1], translation[2] + delta_z)
    _apply_xform(prim, snapped, euler_deg, scale)
    return snapped


def _add_box_collision(
    stage,
    prim_path: str,
    center: tuple[float, float, float] | Gf.Vec3d,
    size: tuple[float, float, float] | Gf.Vec3d,
) -> None:
    collision_geom = UsdGeom.Cube.Define(stage, f"{prim_path}/collision_proxy")
    collision_geom.CreateSizeAttr().Set(1.0)
    collision_prim = collision_geom.GetPrim()
    collision_xform = UsdGeom.Xformable(collision_prim)
    collision_xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    collision_xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    UsdGeom.Imageable(collision_prim).MakeInvisible()


def _make_dynamic_body(prim, mass: float) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr:
        mass_attr.Set(float(mass))
    else:
        mass_api.CreateMassAttr(float(mass))


def _prim_has_api(prim, api_schema) -> bool:
    try:
        return bool(prim.HasAPI(api_schema))
    except Exception:  # pragma: no cover - USD API differences across Isaac versions
        return bool(api_schema(prim))


def _rigid_body_prims(prim) -> list[Any]:
    rigid_prims = []
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        if _prim_has_api(child, UsdPhysics.RigidBodyAPI):
            rigid_prims.append(child)
    return rigid_prims


def _set_mass_if_present(prim, mass: float | None) -> None:
    if mass is None:
        return
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr:
        mass_attr.Set(float(mass))
    else:
        mass_api.CreateMassAttr(float(mass))


def _world_translation(stage, prim_path: str) -> tuple[float, float, float] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def _log_object_height_changes(stage, object_paths: list[str], before: dict[str, tuple[float, float, float] | None]) -> None:
    samples = []
    for path in object_paths[:8]:
        after = _world_translation(stage, path)
        start = before.get(path)
        if start is None or after is None:
            continue
        samples.append(f"{path.rsplit('/', 1)[-1]}:{start[2]:.3f}->{after[2]:.3f}")
    if samples:
        print(f"[task_renderer] object z after settle: {', '.join(samples)}", flush=True)


def _audit_support_alignment(
    stage,
    object_paths: list[str],
    support_heights: dict[str, float],
    *,
    tolerance: float = 0.006,
) -> None:
    if not support_heights:
        return
    checked = 0
    failures = []
    for path in object_paths:
        name = path.rsplit("/", 1)[-1]
        if name not in support_heights:
            continue
        prim = stage.GetPrimAtPath(path)
        bbox = _world_bbox(prim) if prim and prim.IsValid() else None
        if bbox is None:
            continue
        bottom_z = float(bbox[0][2])
        expected_z = float(support_heights[name])
        gap = bottom_z - expected_z
        checked += 1
        if abs(gap) > tolerance:
            failures.append(f"{name}:gap={gap:.4f} bottom={bottom_z:.4f} support={expected_z:.4f}")
    if failures:
        raise RuntimeError("[task_renderer] support alignment failed: " + "; ".join(failures))
    if checked:
        print(f"[task_renderer] support alignment ok for {checked} objects (tol={tolerance:.3f}m)", flush=True)


def _define_plane(stage, root_path: str, cfg: dict[str, Any], *, physics: str = "none") -> str:
    prim_path = f"{root_path}/{_safe_name(cfg.get('name', 'plane'))}"
    size = cfg.get("size") if isinstance(cfg.get("size"), list) else [1.0, 1.0]
    plane = UsdGeom.Plane.Define(stage, prim_path)
    plane.CreateWidthAttr().Set(float(size[0]))
    plane.CreateLengthAttr().Set(float(size[1]))
    translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = _vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    _apply_xform(plane.GetPrim(), translation, euler)
    if physics != "none":
        thickness = float(cfg.get("collision_thickness", 0.02))
        _add_box_collision(stage, prim_path, (0.0, 0.0, -0.5 * thickness), (float(size[0]), float(size[1]), thickness))
    return prim_path


def _load_reference(
    stage,
    root_path: str,
    cfg: dict[str, Any],
    asset_root: Path,
    *,
    physics: str = "none",
    support_z: float | None = None,
) -> str | None:
    usd_path = cfg.get("path") or cfg.get("usd_path")
    if not isinstance(usd_path, str) or not usd_path:
        return None
    abs_usd = _repo_path(usd_path, asset_root)
    if not abs_usd.exists():
        carb.log_warn(f"[task_renderer] missing USD for {cfg.get('name')}: {abs_usd}")
        return None

    prim_path = f"{root_path}/{_safe_name(cfg.get('name', abs_usd.stem))}"
    add_reference_to_stage(str(abs_usd), prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        carb.log_warn(f"[task_renderer] failed to create prim {prim_path}")
        return None
    translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = _vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    scale = _scale3(cfg.get("scale"))
    _apply_xform(prim, translation, euler, scale)
    if support_z is not None:
        snapped = _snap_translation_to_support(prim, translation, euler, scale, support_z)
        if snapped != translation:
            print(
                f"[task_renderer] snapped {cfg.get('name', prim_path)} "
                f"z {translation[2]:.4f}->{snapped[2]:.4f} support_z={support_z:.4f}",
                flush=True,
            )
    if physics == "static":
        bbox = _local_bbox(prim)
        if bbox is not None:
            center, size = bbox
            _add_box_collision(stage, prim_path, center, size)
    elif physics == "dynamic":
        rigid_prims = _rigid_body_prims(prim)
        if rigid_prims:
            mass = cfg.get("mass")
            for rigid_prim in rigid_prims:
                _set_mass_if_present(rigid_prim, float(mass) if mass is not None else None)
            print(
                f"[task_renderer] using authored rigid bodies for {cfg.get('name', prim_path)} "
                f"({len(rigid_prims)} rigid prims)",
                flush=True,
            )
        else:
            bbox = _local_bbox(prim)
            if bbox is not None:
                center, size = bbox
                _add_box_collision(stage, prim_path, center, size)
                _make_dynamic_body(prim, float(cfg.get("mass", 0.2)))
    return prim_path


def _load_items(
    stage,
    root_path: str,
    items: list[dict[str, Any]],
    asset_root: Path,
    *,
    physics: str = "none",
    support_heights: dict[str, float] | None = None,
) -> list[str]:
    loaded = []
    for cfg in items:
        if not isinstance(cfg, dict):
            continue
        if cfg.get("target_class") == "PlaneObject":
            loaded.append(_define_plane(stage, root_path, cfg, physics=physics))
            continue
        support_z = None
        if support_heights is not None:
            name = str(cfg.get("name", ""))
            if name in support_heights:
                support_z = support_heights[name]
            else:
                support_z = support_heights.get(_safe_name(name))
        prim_path = _load_reference(stage, root_path, cfg, asset_root, physics=physics, support_z=support_z)
        if prim_path:
            loaded.append(prim_path)
    return loaded


def _reference_asset_keys(prim) -> list[str]:
    references = prim.GetMetadata("references")
    if references is None:
        return []
    keys = []
    for attr in ("prependedItems", "explicitItems", "addedItems"):
        for reference in getattr(references, attr, []) or []:
            key = _asset_key(getattr(reference, "assetPath", None))
            if key:
                keys.append(key)
    return keys


def _prim_xform_values(prim) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    if not UsdGeom.Xformable(prim):
        return values
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        value = op.Get()
        if value is None:
            continue
        op_name = op.GetOpName()
        if op_name == "xformOp:translate":
            values["translation"] = [float(value[0]), float(value[1]), float(value[2])]
        elif op_name == "xformOp:rotateXYZ":
            values["euler"] = [float(value[0]), float(value[1]), float(value[2])]
        elif op_name == "xformOp:scale":
            values["scale"] = [float(value[0]), float(value[1]), float(value[2])]
    return values


def _scene_usda_transform_overrides(scene_cfg) -> dict[str, dict[str, list[float]]]:
    scene_usda = Path(scene_cfg.scene_dir) / "interndata_scene" / "scene.usda"
    if not scene_usda.exists():
        return {}
    try:
        scene_stage = Usd.Stage.Open(str(scene_usda))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        carb.log_warn(f"[task_renderer] cannot open scene USDA overrides {scene_usda}: {exc}")
        return {}

    overrides: dict[str, dict[str, list[float]]] = {}
    for prim in scene_stage.Traverse():
        values = _prim_xform_values(prim)
        if not values:
            continue
        for key in _reference_asset_keys(prim):
            overrides[key] = values
    if overrides:
        print(f"[task_renderer] loaded {len(overrides)} transform overrides from {scene_usda}", flush=True)
    return overrides


def _apply_transform_overrides(
    items: Any,
    overrides: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not overrides:
        return items if isinstance(items, list) else []
    patched = []
    applied = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cfg = dict(item)
        key = _asset_key(cfg.get("path") or cfg.get("usd_path"))
        if key in overrides:
            cfg.update(overrides[key])
            applied.append(str(cfg.get("name", key)))
        patched.append(cfg)
    if applied:
        print(f"[task_renderer] applied scene transforms: {', '.join(applied)}", flush=True)
    return patched


def _scene_usda_fixture_items(scene_cfg, existing_items: Any) -> list[dict[str, Any]]:
    scene_usda = Path(scene_cfg.scene_dir) / "interndata_scene" / "scene.usda"
    if not scene_usda.exists():
        return existing_items if isinstance(existing_items, list) else []
    try:
        scene_stage = Usd.Stage.Open(str(scene_usda))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        carb.log_warn(f"[task_renderer] cannot open scene USDA fixtures {scene_usda}: {exc}")
        return existing_items if isinstance(existing_items, list) else []

    patched = list(existing_items) if isinstance(existing_items, list) else []
    existing_keys = {
        key
        for item in patched
        if isinstance(item, dict)
        for key in [_asset_key(item.get("path") or item.get("usd_path"))]
        if key
    }
    existing_semantic_keys = {
        key
        for item in patched
        if isinstance(item, dict)
        for key in [_semantic_instance_key(item.get("name", ""))]
        if key
    }
    added = []
    skipped_semantic_duplicates = []
    for prim in scene_stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith("/World/Fixtures/"):
            continue
        ref_keys = [key for key in _reference_asset_keys(prim) if "/fixtures/" in key]
        if not ref_keys:
            continue
        key = ref_keys[0]
        if key in existing_keys:
            continue
        semantic_key = _semantic_instance_key(prim.GetName())
        if semantic_key in existing_semantic_keys:
            skipped_semantic_duplicates.append(prim.GetName())
            continue
        values = _prim_xform_values(prim)
        patched.append(
            {
                "name": _safe_name(prim.GetName()),
                "path": key,
                "target_class": "GeometryObject",
                "translation": values.get("translation", [0.0, 0.0, 0.0]),
                "euler": values.get("euler", [0.0, 0.0, 0.0]),
                "scale": values.get("scale", [1.0, 1.0, 1.0]),
            }
        )
        existing_keys.add(key)
        existing_semantic_keys.add(semantic_key)
        added.append(prim.GetName())
    if skipped_semantic_duplicates:
        print(
            "[task_renderer] skipped scene USDA duplicate fixtures: "
            + ", ".join(skipped_semantic_duplicates),
            flush=True,
        )
    if added:
        print(f"[task_renderer] added scene USDA fixtures: {', '.join(added)}", flush=True)
    return patched


def _load_source_interdata_task(scene_cfg) -> dict[str, Any]:
    task_path = Path(scene_cfg.scene_dir) / "interndata_scene" / "task.yaml"
    if not task_path.exists() and Path(scene_cfg.task_path).name == "task.yaml":
        task_path = Path(scene_cfg.task_path)
    if not task_path.exists():
        return {}
    try:
        with task_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:  # pragma: no cover - defensive for hand-edited YAML
        carb.log_warn(f"[task_renderer] cannot read source support regions {task_path}: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _support_heights_from_regions(scene_cfg) -> dict[str, float]:
    if ARGS.no_snap_to_supports:
        return {}
    payload = _load_source_interdata_task(scene_cfg)
    regions = payload.get("regions")
    if not isinstance(regions, list):
        return {}
    heights: dict[str, float] = {}
    for region in regions:
        if not isinstance(region, dict):
            continue
        object_name = region.get("object")
        candidates = region.get("candidates")
        if not isinstance(object_name, str) or not isinstance(candidates, list) or not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict) or "support_surface_y" not in candidate:
            continue
        try:
            support_z = float(candidate["support_surface_y"])
        except (TypeError, ValueError):
            continue
        heights[object_name] = support_z
        heights[_safe_name(object_name)] = support_z
    if heights:
        unique_count = len({name for name in heights if name == _safe_name(name)})
        print(f"[task_renderer] loaded support heights for {unique_count} objects", flush=True)
    return heights


def _names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item.get("name")) for item in items if isinstance(item, dict) and item.get("name")]


def _position_from_task(scene_cfg) -> tuple[float, float, float] | None:
    positions = scene_cfg.task.get("positions")
    if not isinstance(positions, dict):
        return None

    def from_name(name: Any) -> tuple[float, float, float] | None:
        if not isinstance(name, str):
            return None
        cfg = positions.get(name)
        if not isinstance(cfg, dict):
            return None
        if not all(key in cfg for key in ("x", "y", "yaw")):
            return None
        return (float(cfg["x"]), float(cfg["y"]), float(cfg["yaw"]))

    for phase in scene_cfg.task.get("skills", []):
        if not isinstance(phase, dict):
            continue
        for robot_queues in phase.values():
            if not isinstance(robot_queues, list):
                continue
            for queue_group in robot_queues:
                if not isinstance(queue_group, dict):
                    continue
                for skill in queue_group.get("base", []):
                    if isinstance(skill, dict) and skill.get("name") == "navigate":
                        pose = from_name(skill.get("goal"))
                        if pose is not None:
                            return pose

    for name in positions:
        pose = from_name(name)
        if pose is not None:
            return pose
    return None


def _robot_pose_from_regions(
    scene_cfg,
    robot: dict[str, Any],
    room: dict[str, float],
) -> tuple[float, float, float] | None:
    regions = scene_cfg.task.get("regions")
    if not isinstance(regions, list):
        return None
    robot_name = robot.get("name")
    base_euler = _vec3(robot.get("euler"), (0.0, 0.0, 0.0))
    cx = 0.5 * (room["min_x"] + room["max_x"])
    cy = 0.5 * (room["min_y"] + room["max_y"])
    for region in regions:
        if not isinstance(region, dict) or region.get("object") != robot_name:
            continue
        random_config = region.get("random_config")
        if not isinstance(random_config, dict):
            continue
        pos_range = random_config.get("pos_range")
        yaw_rotation = random_config.get("yaw_rotation")
        if not isinstance(pos_range, list) or not pos_range or not isinstance(pos_range[0], list):
            continue
        shift = pos_range[0]
        if len(shift) < 2:
            continue
        yaw_shift = 0.0
        if isinstance(yaw_rotation, list) and yaw_rotation:
            yaw_shift = float(yaw_rotation[0])
        return (cx + float(shift[0]), cy + float(shift[1]), base_euler[2] + yaw_shift)
    return None


def _robot_visual_items(scene_cfg, robots: list[dict[str, Any]], room: dict[str, float]) -> list[dict[str, Any]]:
    if not robots:
        return []
    visual_robots = []
    for robot in robots:
        pose = _robot_pose_from_regions(scene_cfg, robot, room)
        yaw_is_degrees = True
        if pose is None:
            task_pose = _position_from_task(scene_cfg)
            if task_pose is None:
                x = 0.5 * (room["min_x"] + room["max_x"])
                y = room["min_y"] + 0.25 * (room["max_y"] - room["min_y"])
                yaw = 180.0
            else:
                x, y, yaw = task_pose
                yaw_is_degrees = False
        else:
            x, y, yaw = pose
        yaw_deg = yaw if yaw_is_degrees else math.degrees(yaw)
        cfg = dict(robot)
        cfg["translation"] = [x, y, room["floor_z"]]
        euler = list(_vec3(cfg.get("euler"), (0.0, 0.0, 0.0)))
        euler[2] = yaw_deg
        cfg["euler"] = euler
        visual_robots.append(cfg)
        print(f"[task_renderer] robot visual pose x={x:.3f} y={y:.3f} yaw_deg={yaw_deg:.1f}", flush=True)
    return visual_robots


def _bounds_from_payload(scene_cfg) -> tuple[float, float, float, float, float, float]:
    points = []
    for cfg in list(scene_cfg.arena.get("fixtures", [])) + list(scene_cfg.task.get("objects", [])):
        if not isinstance(cfg, dict):
            continue
        translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
        size = cfg.get("size") if isinstance(cfg.get("size"), list) else [0.5, 0.5, 0.5]
        sx = float(size[0]) if len(size) > 0 else 0.5
        sy = float(size[1]) if len(size) > 1 else 0.5
        sz = float(size[2]) if len(size) > 2 else 0.5
        points.append((translation[0] - sx * 0.5, translation[1] - sy * 0.5, translation[2] - sz * 0.5))
        points.append((translation[0] + sx * 0.5, translation[1] + sy * 0.5, translation[2] + sz * 0.5))
    if not points:
        return (-2.0, -2.0, 0.0, 2.0, 2.0, 2.0)
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        min(p[2] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
        max(p[2] for p in points),
    )


def _setup_lighting(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/DomeLight"))
    dome.CreateIntensityAttr(850.0)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/KeyLight"))
    distant.CreateIntensityAttr(4500.0)
    distant.CreateAngleAttr(0.55)
    _apply_xform(distant.GetPrim(), (0.0, 0.0, 5.0), (-50.0, 0.0, 35.0))


def _room_frame(scene_cfg, fallback_bounds: tuple[float, float, float, float, float, float]) -> dict[str, float]:
    floor = None
    for cfg in scene_cfg.arena.get("fixtures", []):
        if not isinstance(cfg, dict):
            continue
        if cfg.get("target_class") == "PlaneObject" and str(cfg.get("name", "")).lower() == "floor":
            floor = cfg
            break
    if floor is not None:
        translation = _vec3(floor.get("translation"), (0.0, 0.0, 0.0))
        size = floor.get("size") if isinstance(floor.get("size"), list) else [4.0, 3.0]
        room = {
            "min_x": translation[0] - float(size[0]) * 0.5,
            "max_x": translation[0] + float(size[0]) * 0.5,
            "min_y": translation[1] - float(size[1]) * 0.5,
            "max_y": translation[1] + float(size[1]) * 0.5,
            "floor_z": translation[2],
            "height": 2.8,
        }
    else:
        min_x, min_y, min_z, max_x, max_y, max_z = fallback_bounds
        room = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "floor_z": min_z,
            "height": max(max_z - min_z, 2.8),
        }

    wall_tops = []
    for cfg in scene_cfg.arena.get("fixtures", []):
        if not isinstance(cfg, dict) or cfg.get("target_class") != "PlaneObject":
            continue
        if cfg.get("role") != "wall" and not str(cfg.get("name", "")).startswith("wall_"):
            continue
        translation = _vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
        size = cfg.get("size") if isinstance(cfg.get("size"), list) else [0.0, 0.0]
        if len(size) >= 2:
            wall_tops.append(translation[2] + float(size[1]) * 0.5)
    if wall_tops:
        room["height"] = max(wall_tops) - room["floor_z"]
    return room


def _camera_views(room: dict[str, float]) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    min_x = room["min_x"]
    max_x = room["max_x"]
    min_y = room["min_y"]
    max_y = room["max_y"]
    floor_z = room["floor_z"]
    room_height = room["height"]
    width = max(max_x - min_x, 1.0)
    depth = max(max_y - min_y, 1.0)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    scene_radius = 0.5 * math.hypot(width, depth)
    eye_z = floor_z + min(max(room_height * 0.78, 2.0), room_height - 0.08)
    target_z = floor_z + min(max(room_height * 0.22, 0.62), room_height - 0.75)
    inner = 0.22
    side = 0.18
    diagonal = 0.24
    overview_z = floor_z + min(max(room_height * 0.86, 2.35), room_height - 0.05)
    overhead_eye = (cx + width * 0.24, cy - depth * 0.20, overview_z)
    overhead_target = (cx, cy, target_z)
    return [
        ("south_interior", (cx, min_y + depth * inner, eye_z), (cx, cy + depth * side, target_z), (0.0, 0.0, 1.0)),
        ("north_interior", (cx, max_y - depth * inner, eye_z), (cx, cy - depth * side, target_z), (0.0, 0.0, 1.0)),
        ("west_interior", (min_x + width * inner, cy, eye_z), (cx + width * side, cy, target_z), (0.0, 0.0, 1.0)),
        ("east_interior", (max_x - width * inner, cy, eye_z), (cx - width * side, cy, target_z), (0.0, 0.0, 1.0)),
        ("diagonal_overview", (cx - scene_radius * diagonal, cy - scene_radius * diagonal, overview_z), (cx, cy, target_z), (0.0, 0.0, 1.0)),
        ("overhead_oblique", overhead_eye, overhead_target, (0.0, 0.0, 1.0)),
    ]


def _expected_rgb_path(view_dir: Path) -> Path:
    return view_dir / "rgb_0000.png"


def _capture_rgb(rep, writer, render_product, view_dir: Path) -> Path:
    expected_rgb = _expected_rgb_path(view_dir)
    if expected_rgb.exists():
        expected_rgb.unlink()
    writer.attach([render_product])
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=max(1, int(ARGS.rt_subframes)), pause_timeline=False)
        rep.orchestrator.wait_until_complete()
        if expected_rgb.exists():
            break
    writer.detach()
    if not expected_rgb.exists():
        raise RuntimeError(f"render did not produce {expected_rgb}")
    _validate_png_not_blank(expected_rgb)
    return expected_rgb


def _png_rgb_std(image_path: Path) -> float:
    data = image_path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"rendered image is not PNG: {image_path}")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat_chunks = []
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            interlace = chunk_data[12]
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if (width is None or height is None or bit_depth != 8 or color_type not in {2, 6} or interlace != 0):
        raise RuntimeError(f"unsupported PNG format for blank check: {image_path}")
    channels = 4 if color_type == 6 else 3
    row_bytes = int(width) * channels
    raw = zlib.decompress(b"".join(idat_chunks))
    prev = bytearray(row_bytes)
    pos = 0
    n = 0
    total = 0.0
    total_sq = 0.0
    stride = max(1, int(width) // 256)
    for _row in range(int(height)):
        filter_type = raw[pos]
        pos += 1
        current = bytearray(raw[pos : pos + row_bytes])
        pos += row_bytes
        for idx in range(row_bytes):
            left = current[idx - channels] if idx >= channels else 0
            up = prev[idx]
            up_left = prev[idx - channels] if idx >= channels else 0
            if filter_type == 1:
                current[idx] = (current[idx] + left) & 0xFF
            elif filter_type == 2:
                current[idx] = (current[idx] + up) & 0xFF
            elif filter_type == 3:
                current[idx] = (current[idx] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                current[idx] = (current[idx] + predictor) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"unsupported PNG filter {filter_type} in {image_path}")
        for x in range(0, int(width), stride):
            base = x * channels
            luma = (0.2126 * current[base] + 0.7152 * current[base + 1] + 0.0722 * current[base + 2]) / 255.0
            total += luma
            total_sq += luma * luma
            n += 1
        prev = current
    if n == 0:
        raise RuntimeError(f"rendered image has no pixels: {image_path}")
    mean = total / n
    variance = max(total_sq / n - mean * mean, 0.0)
    return math.sqrt(variance)


def _validate_png_not_blank(image_path: Path) -> None:
    try:
        byte_count = image_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"cannot read rendered image {image_path}: {exc}") from exc
    if byte_count < 1024:
        raise RuntimeError(f"rendered image is unexpectedly small: {image_path}")
    rgb_std = _png_rgb_std(image_path)
    if rgb_std < 0.015:
        raise RuntimeError(f"rendered image appears blank: {image_path} rgb_std={rgb_std:.5f}")


def _define_topdown_camera(stage, room: dict[str, float]) -> str:
    min_x = room["min_x"]
    max_x = room["max_x"]
    min_y = room["min_y"]
    max_y = room["max_y"]
    floor_z = room["floor_z"]
    room_height = room["height"]
    width = max(max_x - min_x, 1.0)
    depth = max(max_y - min_y, 1.0)
    aspect = max(float(ARGS.width) / max(float(ARGS.height), 1.0), 1.0)
    frustum_width = max(width, depth * aspect) * 1.18
    frustum_height = frustum_width / aspect
    camera_path = "/World/Cameras/TopDownCamera"
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    # Orthographic USD cameras use aperture to define the visible world extent.
    # Isaac's generated scenes store this in tenths of stage units.
    camera.CreateHorizontalApertureAttr().Set(float(frustum_width * 10.0))
    camera.CreateVerticalApertureAttr().Set(float(frustum_height * 10.0))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    _apply_xform(
        camera.GetPrim(),
        (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), floor_z + room_height + max(width, depth) * 1.35),
        (0.0, 0.0, 0.0),
    )
    return camera_path


def _render_views(stage, scene_cfg, output_dir: Path) -> list[Path]:
    import omni.replicator.core as rep  # pylint: disable=import-outside-toplevel

    room = _room_frame(scene_cfg, _bounds_from_payload(scene_cfg))
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    print(
        "[task_renderer] room frame "
        f"x=({room['min_x']:.3f},{room['max_x']:.3f}) "
        f"y=({room['min_y']:.3f},{room['max_y']:.3f}) "
        f"z=({room['floor_z']:.3f},{room['floor_z'] + room['height']:.3f})",
        flush=True,
    )
    for view_name, eye, target, up in _camera_views(room):
        view_dir = output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        camera = rep.create.camera(
            name=_safe_name(view_name),
            position=eye,
            look_at=target,
            look_at_up_axis=up,
            focal_length=16.0,
            clipping_range=(0.01, 1000.0),
        )
        render_product = rep.create.render_product(camera, (int(ARGS.width), int(ARGS.height)), force_new=True)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(view_dir),
            rgb=True,
            image_output_format="png",
            frame_padding=4,
        )
        rgb_path = _capture_rgb(rep, writer, render_product, view_dir)
        rendered.append(view_dir)
        print(f"[task_renderer] rendered {view_name} eye={eye} target={target} -> {rgb_path}", flush=True)
    topdown_camera = _define_topdown_camera(stage, room)
    view_dir = output_dir / "topdown"
    view_dir.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(topdown_camera, (int(ARGS.width), int(ARGS.height)), force_new=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(view_dir),
        rgb=True,
        image_output_format="png",
        frame_padding=4,
    )
    rgb_path = _capture_rgb(rep, writer, render_product, view_dir)
    rendered.append(view_dir)
    print(f"[task_renderer] rendered topdown camera={topdown_camera} -> {rgb_path}", flush=True)
    return rendered


def main() -> int:
    print(f"[task_renderer] starting with task={ARGS.task}", flush=True)
    enable_extension("omni.replicator.core")
    for _ in range(10):
        SIMULATION_APP.update()

    physics_dt = 1.0 / 30.0
    world = World(physics_dt=physics_dt, rendering_dt=physics_dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    include_robot = not ARGS.no_robot
    print("[task_renderer] loading compatible scene config", flush=True)
    scene_cfg = load_scene_config(ARGS.task, include_robot=include_robot)
    _setup_lighting(stage)
    transform_overrides = _scene_usda_transform_overrides(scene_cfg)
    support_heights = _support_heights_from_regions(scene_cfg)
    fixtures = _scene_usda_fixture_items(scene_cfg, scene_cfg.arena.get("fixtures", []))
    fixtures = _apply_transform_overrides(fixtures, transform_overrides)
    objects = _apply_transform_overrides(scene_cfg.task.get("objects", []), transform_overrides)
    render_scene_cfg = SimpleNamespace(
        **{
            **vars(scene_cfg),
            "arena": {**scene_cfg.arena, "fixtures": fixtures},
            "task": {**scene_cfg.task, "objects": objects},
        }
    )
    room = _room_frame(render_scene_cfg, _bounds_from_payload(render_scene_cfg))
    robots = scene_cfg.task.get("robots", []) if include_robot else []
    visual_robots = _robot_visual_items(
        render_scene_cfg,
        robots if isinstance(robots, list) else [],
        room,
    )
    loaded_arena = _load_items(
        stage,
        "/World/Arena",
        fixtures if isinstance(fixtures, list) else [],
        scene_cfg.asset_root,
        physics="none" if ARGS.no_physics else "static",
    )
    loaded_objects = _load_items(
        stage,
        "/World/Objects",
        objects if isinstance(objects, list) else [],
        scene_cfg.asset_root,
        physics="none" if ARGS.no_physics else "dynamic",
        support_heights=support_heights,
    )
    loaded_robots = _load_items(stage, "/World/Robots", visual_robots, scene_cfg.asset_root)
    expected_fixture_names = set(_names(fixtures))
    loaded_fixture_names = {path.rsplit("/", 1)[-1] for path in loaded_arena}
    missing_fixtures = sorted(expected_fixture_names - loaded_fixture_names)

    object_pose_before_settle = {path: _world_translation(stage, path) for path in loaded_objects}
    if not ARGS.no_physics:
        print("[task_renderer] resetting world physics after loading collision proxies", flush=True)
        world.reset()
    settle_steps = max(0, int(math.ceil(float(ARGS.settle_seconds) / physics_dt)))
    print(
        f"[task_renderer] settling physics for {settle_steps} steps "
        f"({settle_steps * physics_dt:.3f}s)",
        flush=True,
    )
    for _ in range(settle_steps):
        world.step(render=True)
    if not ARGS.no_physics:
        _log_object_height_changes(stage, loaded_objects, object_pose_before_settle)
    _audit_support_alignment(stage, loaded_objects, support_heights)
    output_dir = _repo_path(ARGS.output_dir)
    rendered = _render_views(stage, render_scene_cfg, output_dir)
    print(
        "[task_renderer] loaded "
        f"arena={len(loaded_arena)}/{len(_names(fixtures))} "
        f"objects={len(loaded_objects)}/{len(_names(objects))} "
        f"robots={len(loaded_robots)}/{len(_names(robots))} "
        f"from {scene_cfg.task_path}",
        flush=True,
    )
    if missing_fixtures:
        print(f"[task_renderer] missing fixtures: {', '.join(missing_fixtures)}", flush=True)
    print(f"[task_renderer] wrote {len(rendered)} views under {output_dir}", flush=True)
    return 0


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
