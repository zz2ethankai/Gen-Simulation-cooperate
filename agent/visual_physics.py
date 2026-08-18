#!/usr/bin/env python3
"""In-process Isaac Sim physics renderer for Agent scene observation.

It renders deterministic views from either a SimBox ``simbox_task.yaml`` or an
interdata ``task.yaml`` and records a physics audit plus a render status for the
user-facing ``python -m agent view`` command.

Ordering constraint: ``SimulationApp`` must be created before any ``omni.*`` /
``pxr.*`` import, so all of those imports happen lazily inside ``run_render`` /
``_render_main`` and helpers reach them through the ``PX`` shim or module
globals populated there.

Entry point::

    from agent import visual_physics
    visual_physics.run_render(settings)

``settings`` is a namespace carrying these fields:
``task``, ``output_dir``, ``width``, ``height``, ``rt_subframes``,
``settle_seconds``, ``gravity_mps2``, ``include_robot``, ``single_view``,
``renderer`` (all optional except ``task`` / ``output_dir``).
"""

from __future__ import annotations

import atexit
import importlib
import json
import math
import os
import re
import sys
import traceback
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from workflows.simbox.core.utils.camera_template import resolve_camera_template_pose
from workflows.simbox.core.robots.profile import (
    PlacementFamily,
    load_robot_profile_for_task,
    resolve_robot_asset_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIENCE = "/isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit"


class _LazyPxr:
    """Resolve ``pxr.*`` submodules on first attribute access (after app launch)."""

    _cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = importlib.import_module(f"pxr.{name}")
        return self._cache[name]


PX = _LazyPxr()

# Populated by ``_render_main`` after SimulationApp is created.
OMNI_USD = None
WORLD_CLS = None
ENABLE_EXTENSION = None

# Normalized render settings; set by ``run_render``.
SETTINGS: SimpleNamespace | None = None


def _opt(name: str, default: Any = None) -> Any:
    return getattr(SETTINGS, name, default)


def _warn(msg: str) -> None:
    print(f"[visual_physics] WARNING: {msg}", file=sys.stderr, flush=True)


def _normalize(settings: Any) -> SimpleNamespace:
    return SimpleNamespace(
        task=Path(getattr(settings, "task")),
        output_dir=Path(getattr(settings, "output_dir")),
        width=int(getattr(settings, "width", 2560)),
        height=int(getattr(settings, "height", 1440)),
        rt_subframes=int(getattr(settings, "rt_subframes", 32)),
        settle_seconds=float(getattr(settings, "settle_seconds", 1.0)),
        gravity_mps2=float(getattr(settings, "gravity_mps2", -9.81)),
        no_physics=bool(getattr(settings, "no_physics", False)),
        include_robot=bool(getattr(settings, "include_robot", True)),
        single_view=str(getattr(settings, "single_view", "")),
        renderer=str(getattr(settings, "renderer", "RayTracedLighting")),
        no_snap_to_supports=bool(getattr(settings, "no_snap_to_supports", False)),
    )


# --------------------------------------------------------------------------- #
# config loading (SimBox + InterData dispatch)
# --------------------------------------------------------------------------- #


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def repo_path(path: str | Path, base: Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if base is not None:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path_obj).resolve()


def load_simbox_task(task_path: Path) -> dict[str, Any]:
    payload = load_yaml(task_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{task_path} must contain a mapping")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        raise ValueError(f"{task_path} is not a supported SimBox task YAML")

    task = tasks[0]
    arena_file = task.get("arena_file")
    if not isinstance(arena_file, str) or not arena_file:
        raise ValueError(f"{task_path} missing tasks[0].arena_file")

    task_dir = task_path.parent
    arena_path = repo_path(Path(arena_file), task_dir)
    arena_payload = load_yaml(arena_path)
    if not isinstance(arena_payload, dict):
        raise ValueError(f"{arena_path} must contain a mapping")

    return {
        "source": "simbox",
        "scene_dir": task_dir.parent if task_dir.name == "simbox" else task_dir,
        "task_path": task_path,
        "arena_path": arena_path,
        "task": task,
        "arena": arena_payload,
        "asset_root": repo_path(Path(str(task.get("asset_root", task_dir))), task_dir),
    }


def load_interdata_task(task_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    task_dir = task_path.parent
    arena_file = payload.get("arena") or payload.get("arena_file") or "arena.yaml"
    if not isinstance(arena_file, str) or not arena_file:
        arena_file = "arena.yaml"
    arena_path = repo_path(Path(arena_file), task_dir)
    arena_payload = load_yaml(arena_path)
    if not isinstance(arena_payload, dict):
        raise ValueError(f"{arena_path} must contain a mapping")
    scene_dir = task_dir.parent if task_dir.name.startswith("interdata") else task_dir
    asset_root = payload.get("asset_root") or scene_dir
    return {
        "source": "interdata_task",
        "scene_dir": scene_dir,
        "task_path": task_path,
        "arena_path": arena_path,
        "task": payload,
        "arena": arena_payload,
        "asset_root": repo_path(Path(str(asset_root)), task_dir),
    }


def load_scene_config(task_path: str | Path, *, include_robot: bool = False) -> SimpleNamespace:
    del include_robot
    path = repo_path(Path(task_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    if (
        payload.get("format") == "task"
        or isinstance(payload.get("arena"), str)
        or isinstance(payload.get("regions"), list)
    ):
        return SimpleNamespace(**load_interdata_task(path, payload))
    if isinstance(payload.get("tasks"), list) and payload.get("tasks"):
        return SimpleNamespace(**load_simbox_task(path))
    raise ValueError(f"{path} is not a supported SimBox/InterData task YAML")


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "asset"


def asset_key(path: str | Path | None) -> str | None:
    if path is None:
        return None
    normalized = str(path).replace("\\", "/")
    marker = "assets/"
    if marker not in normalized:
        return normalized.lstrip("./")
    return normalized[normalized.index(marker):]


def semantic_instance_key(name: Any) -> str:
    safe = safe_name(str(name))
    parts = [part for part in re.split(r"_+", safe) if part]
    semantic_parts = [
        part for part in parts
        if not part.isdigit() and not re.fullmatch(r"id\d+", part)
    ]
    return "_".join(semantic_parts) or safe


def vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def scale3(value: Any) -> tuple[float, float, float]:
    if isinstance(value, list) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (1.0, 1.0, 1.0)


def normalized_collider_kind(value: Any, default: str) -> str:
    requested = str(value or default).strip().lower()
    return re.sub(r"[^a-z0-9]", "", requested)


def static_convex_collision_method(
    collider_kind: str, *, has_coacd_sidecar: bool
) -> str:
    if collider_kind not in {"coacd", "convexdecomposition"}:
        raise ValueError(f"not a convex static collider: {collider_kind}")
    if has_coacd_sidecar:
        return "coacd"
    if collider_kind == "convexdecomposition":
        return "physx_convex_decomposition"
    return "missing_coacd"


def object_physics_mode(cfg: dict[str, Any]) -> str:
    physics = cfg.get("physics")
    declarations = []
    if isinstance(physics, dict) and "rigid_body" in physics:
        declarations.append(("physics.rigid_body", physics["rigid_body"], False))
    if "static" in cfg:
        declarations.append(("static", cfg["static"], True))
    if "rigidbody" in cfg:
        declarations.append(("rigidbody", cfg["rigidbody"], False))
    for field, value, static_when_true in declarations:
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")
        is_static = value if static_when_true else not value
        return "static" if is_static else "dynamic"
    if cfg.get("target_class") in {"GeometryObject", "PlaneObject"}:
        return "static"
    return "dynamic"


def partition_objects_by_physics(
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    static_objects = []
    dynamic_objects = []
    for cfg in objects:
        destination = (
            static_objects
            if object_physics_mode(cfg) == "static"
            else dynamic_objects
        )
        destination.append(cfg)
    return static_objects, dynamic_objects


def apply_xform(prim, translation, euler_deg, scale=(1.0, 1.0, 1.0)) -> None:
    xform = PX.UsdGeom.Xformable(prim)
    translate_op = existing_or_add_xform_op(
        xform, "xformOp:translate", xform.AddTranslateOp
    )
    rotate_op = existing_or_add_xform_op(
        xform, "xformOp:rotateXYZ", xform.AddRotateXYZOp
    )
    scale_op = existing_or_add_xform_op(xform, "xformOp:scale", xform.AddScaleOp)
    set_xform_op_vec3(translate_op, translation)
    set_xform_op_vec3(rotate_op, euler_deg)
    set_xform_op_vec3(scale_op, scale)
    xform.SetXformOpOrder([translate_op, rotate_op, scale_op])


def existing_or_add_xform_op(xform, attribute_name: str, add_op):
    attribute = xform.GetPrim().GetAttribute(attribute_name)
    if attribute:
        return PX.UsdGeom.XformOp(attribute)
    return add_op()


def set_xform_op_vec3(op, values) -> None:
    vector_types = {
        PX.UsdGeom.XformOp.PrecisionDouble: PX.Gf.Vec3d,
        PX.UsdGeom.XformOp.PrecisionFloat: PX.Gf.Vec3f,
        PX.UsdGeom.XformOp.PrecisionHalf: PX.Gf.Vec3h,
    }
    vector_type = vector_types[op.GetPrecision()]
    op.Set(vector_type(*values))


def world_bbox(prim) -> tuple[Any, Any] | None:
    bbox_cache = PX.UsdGeom.BBoxCache(
        PX.Usd.TimeCode.Default(),
        [PX.UsdGeom.Tokens.default_, PX.UsdGeom.Tokens.render, PX.UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    size = bbox.GetSize()
    if min(float(size[0]), float(size[1]), float(size[2])) <= 0.0:
        return None
    return bbox.GetMin(), bbox.GetMax()


def support_body_box_geometry(
    bbox_min,
    bbox_max,
    *,
    support_surface_z: float,
    top_clearance: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minimum = tuple(float(bbox_min[index]) for index in range(3))
    maximum = tuple(float(bbox_max[index]) for index in range(3))
    support_z = float(support_surface_z)
    clearance = float(top_clearance)
    if not all(math.isfinite(value) for value in (*minimum, *maximum, support_z, clearance)):
        raise ValueError("support body bounds must be finite")
    if clearance < 0.0:
        raise ValueError("support body top clearance must be non-negative")
    body_top_z = min(maximum[2], support_z - clearance)
    size = (
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        body_top_z - minimum[2],
    )
    if min(size) <= 0.0:
        raise ValueError("support body bounds are empty after clipping")
    center = (
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.5 * (minimum[2] + body_top_z),
    )
    return center, size


def world_translation(stage, prim_path: str) -> tuple[float, float, float] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    matrix = PX.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        PX.Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def add_box_collision(
    stage,
    prim_path: str,
    center,
    size,
    *,
    proxy_name: str = "collision_proxy",
    world_space: bool = False,
):
    collision_geom = PX.UsdGeom.Cube.Define(stage, f"{prim_path}/{proxy_name}")
    collision_geom.CreateSizeAttr().Set(1.0)
    collision_prim = collision_geom.GetPrim()
    collision_xform = PX.UsdGeom.Xformable(collision_prim)
    if world_space:
        collision_xform.SetResetXformStack(True)
    collision_xform.AddTranslateOp().Set(PX.Gf.Vec3d(*center))
    collision_xform.AddScaleOp().Set(PX.Gf.Vec3f(*size))
    PX.UsdPhysics.CollisionAPI.Apply(collision_prim)
    PX.UsdGeom.Imageable(collision_prim).MakeInvisible()
    return collision_prim


def make_dynamic_body(prim, mass: float) -> None:
    PX.UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = PX.UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr:
        mass_attr.Set(float(mass))
    else:
        mass_api.CreateMassAttr(float(mass))


def prim_has_api(prim, api_schema) -> bool:
    try:
        return bool(prim.HasAPI(api_schema))
    except Exception:  # pragma: no cover - USD API differences across Isaac versions
        return bool(api_schema(prim))


def collect_collision_prims(prim) -> list[Any]:
    return [
        child
        for child in PX.Usd.PrimRange(prim)
        if prim_has_api(child, PX.UsdPhysics.CollisionAPI)
    ]


def is_visible_mesh(prim) -> bool:
    if not prim.IsA(PX.UsdGeom.Mesh):
        return False
    imageable = PX.UsdGeom.Imageable(prim)
    if imageable.ComputeVisibility() == PX.UsdGeom.Tokens.invisible:
        return False
    try:
        purpose = imageable.ComputePurpose()
    except AttributeError:
        purpose = imageable.GetPurposeAttr().Get() or PX.UsdGeom.Tokens.default_
    return purpose not in {PX.UsdGeom.Tokens.proxy, PX.UsdGeom.Tokens.guide}


def clear_physics_apis(prim) -> None:
    for child in list(PX.Usd.PrimRange(prim)):
        for api_schema in (
            PX.UsdPhysics.CollisionAPI,
            PX.UsdPhysics.MeshCollisionAPI,
            PX.UsdPhysics.RigidBodyAPI,
            PX.UsdPhysics.MassAPI,
            PX.PhysxSchema.PhysxConvexDecompositionCollisionAPI,
        ):
            if prim_has_api(child, api_schema):
                child.RemoveAPI(api_schema)


def configure_convex_decomposition(prim) -> dict[str, Any]:
    settings = {
        "hull_vertex_limit": int(
            os.environ.get("TASK_RENDER_CONVEX_HULL_VERTEX_LIMIT", "64")
        ),
        "max_convex_hulls": int(
            os.environ.get("TASK_RENDER_CONVEX_MAX_HULLS", "64")
        ),
        "min_thickness": float(
            os.environ.get("TASK_RENDER_CONVEX_MIN_THICKNESS", "0.001")
        ),
        "voxel_resolution": int(
            os.environ.get("TASK_RENDER_CONVEX_VOXEL_RESOLUTION", "500000")
        ),
        "error_percentage": float(
            os.environ.get("TASK_RENDER_CONVEX_ERROR_PERCENTAGE", "2.5")
        ),
        "shrink_wrap": os.environ.get(
            "TASK_RENDER_CONVEX_SHRINK_WRAP", "1"
        ).strip().lower()
        not in {"0", "false", "no"},
    }
    if settings["hull_vertex_limit"] < 4:
        raise RuntimeError("convex hull vertex limit must be at least 4")
    if settings["max_convex_hulls"] < 1:
        raise RuntimeError("maximum convex hull count must be positive")
    if not 10000 <= settings["voxel_resolution"] <= 10000000:
        raise RuntimeError("convex voxel resolution is outside the PhysX range")
    api = PX.PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    api.CreateHullVertexLimitAttr().Set(settings["hull_vertex_limit"])
    api.CreateMaxConvexHullsAttr().Set(settings["max_convex_hulls"])
    api.CreateMinThicknessAttr().Set(settings["min_thickness"])
    api.CreateVoxelResolutionAttr().Set(settings["voxel_resolution"])
    api.CreateErrorPercentageAttr().Set(settings["error_percentage"])
    api.CreateShrinkWrapAttr().Set(settings["shrink_wrap"])
    return settings


def replace_with_visual_mesh_collision(prim, approximation: str) -> list[Any]:
    """Discard stale package proxies and author collision from visible geometry."""
    clear_physics_apis(prim)
    collision_prims = []
    for child in PX.Usd.PrimRange(prim):
        if not is_visible_mesh(child):
            continue
        PX.UsdPhysics.CollisionAPI.Apply(child)
        mesh_collision = PX.UsdPhysics.MeshCollisionAPI.Apply(child)
        mesh_collision.CreateApproximationAttr().Set(approximation)
        if approximation == "convexDecomposition":
            configure_convex_decomposition(child)
        collision_prims.append(child)
    return collision_prims


def attach_coacd_collision(stage, prim, collision_usd: Path, *, profile: str):
    collision_path = f"{prim.GetPath()}/coacd_collision"
    collision_root = stage.DefinePrim(collision_path, "Xform")
    if not collision_root.GetReferences().AddReference(collision_usd.as_posix()):
        raise RuntimeError(f"failed to reference CoACD collision: {collision_usd}")
    collision_root.SetCustomDataByKey("collision:decomposition", "CoACD")
    collision_root.SetCustomDataByKey("collision:profile", profile)
    collision_prims = [
        child
        for child in PX.Usd.PrimRange(collision_root)
        if child.IsA(PX.UsdGeom.Mesh)
        and prim_has_api(child, PX.UsdPhysics.CollisionAPI)
        and str(
            PX.UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get() or ""
        )
        == "convexHull"
    ]
    if not collision_prims:
        raise RuntimeError(f"CoACD sidecar contains no convex hulls: {collision_usd}")
    return collision_root, collision_prims


def coacd_sidecar(runtime_usd: Path, profile: str) -> Path | None:
    name = "coacd_static.usda" if profile == "static_ground" else "coacd_dynamic.usda"
    direct = runtime_usd.parent / "collision" / name
    if direct.is_file():
        return direct
    metadata_path = runtime_usd.parent / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        relative = metadata.get("collision_source")
        if isinstance(relative, str) and Path(relative).name == name:
            packaged = runtime_usd.parent / relative
            if packaged.is_file():
                return packaged
    packaged = runtime_usd.parent / "source_package" / "collision" / name
    return packaged if packaged.is_file() else None


def rigid_body_prims(prim) -> list[Any]:
    rigid_prims = []
    for child in PX.Usd.PrimRange(prim):
        if child == prim:
            continue
        if prim_has_api(child, PX.UsdPhysics.RigidBodyAPI):
            rigid_prims.append(child)
    return rigid_prims


def snap_translation_to_support(
    prim,
    translation: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
    support_z: float | None,
) -> tuple[float, float, float]:
    if support_z is None:
        return translation
    bbox = world_bbox(prim)
    if bbox is None:
        return translation
    bbox_min, _ = bbox
    current_bottom_z = float(bbox_min[2])
    if not math.isfinite(current_bottom_z):
        return translation
    # Static CoACD collision can top out a few mm ABOVE the declared support
    # surface; snapping flush leaves objects spawning pre-penetrated, which at
    # a coarse physics dt lets PhysX eject them downward through a hull seam.
    clearance = 0.001
    delta_z = float(support_z) + clearance - current_bottom_z
    if abs(delta_z) < 1e-5:
        return translation
    snapped = (translation[0], translation[1], translation[2] + delta_z)
    apply_xform(prim, snapped, euler_deg, scale)
    return snapped


def audit_wall_attachments(stage, fixture_paths: list[str], fixture_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure final visual/collision bounds against the declared room wall."""
    config_by_name = {
        safe_name(cfg.get("name", "")): cfg
        for cfg in fixture_configs
        if isinstance(cfg, dict) and cfg.get("long_edge_against_wall")
    }
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in fixture_paths:
        name = path.rsplit("/", 1)[-1]
        cfg = config_by_name.get(name)
        if cfg is None:
            continue
        report = cfg.get("asset_wall_attachment")
        if not isinstance(report, dict):
            failures.append(f"{name}: missing attachment contract")
            continue
        prim = stage.GetPrimAtPath(path)
        bounds = world_bbox(prim) if prim and prim.IsValid() else None
        if bounds is None:
            failures.append(f"{name}: no measurable world bounds")
            continue
        minimum, maximum = bounds
        wall = str(cfg.get("preferred_wall") or "").lower()
        boundary = float(report.get("room_boundary_coordinate"))
        if wall == "west":
            signed_gap = float(minimum[0]) - boundary
        elif wall == "east":
            signed_gap = boundary - float(maximum[0])
        elif wall == "south":
            signed_gap = float(minimum[1]) - boundary
        elif wall == "north":
            signed_gap = boundary - float(maximum[1])
        else:
            failures.append(f"{name}: invalid wall {wall!r}")
            continue
        wall_clearance = float(report.get("wall_collision_clearance_m", 0.0))
        passed = -0.004 <= signed_gap <= 0.015
        records.append(
            {
                "name": name,
                "wall": wall,
                "signed_gap_m": round(signed_gap, 6),
                "allowed_gap_m": 0.015,
                "available_wall_collision_clearance_m": round(wall_clearance, 6),
                "status": "passed" if passed else "failed",
            }
        )
        if not passed:
            failures.append(f"{name}: wall={wall} signed_gap={signed_gap:.6f}m")
    if failures:
        raise RuntimeError("runtime wall attachment audit failed: " + "; ".join(failures))
    print(
        f"[visual_physics] runtime wall attachment audit ok for {len(records)} fixtures",
        flush=True,
    )
    return records


def audit_dynamic_collisions(stage, object_paths: list[str]) -> list[dict[str, Any]]:
    """Reject dynamic objects that did not enter PhysX as convex decompositions."""
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in object_paths:
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            failures.append(f"{path}:missing_root")
            continue
        coacd_root = stage.GetPrimAtPath(f"{path}/coacd_collision")
        uses_coacd = bool(coacd_root and coacd_root.IsValid())
        mesh_count = 0
        convex_mesh_count = 0
        collision_count = 0
        traversal_root = coacd_root if uses_coacd else root
        for prim in PX.Usd.PrimRange(traversal_root):
            if prim_has_api(prim, PX.UsdPhysics.CollisionAPI):
                collision_count += 1
            is_collision_mesh = (
                prim.IsA(PX.UsdGeom.Mesh) if uses_coacd else is_visible_mesh(prim)
            )
            if not is_collision_mesh:
                continue
            mesh_count += 1
            approximation = str(
                PX.UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or ""
            )
            expected_approximation = (
                "convexHull" if uses_coacd else "convexDecomposition"
            )
            if (
                prim_has_api(prim, PX.UsdPhysics.CollisionAPI)
                and approximation == expected_approximation
            ):
                convex_mesh_count += 1
        rigid_body_count = len(rigid_body_prims(root)) + int(
            prim_has_api(root, PX.UsdPhysics.RigidBodyAPI)
        )
        name = path.rsplit("/", 1)[-1]
        record = {
            "name": name,
            "mesh_count": mesh_count,
            "collision_prim_count": collision_count,
            "convex_decomposition_mesh_count": convex_mesh_count,
            "rigid_body_count": rigid_body_count,
            "status": "passed",
            "method": "CoACD" if uses_coacd else "PhysXConvexDecomposition",
        }
        reasons = []
        if mesh_count <= 0:
            reasons.append("no_mesh")
        if convex_mesh_count != mesh_count:
            reasons.append(f"collision_hulls={convex_mesh_count}/{mesh_count}")
        if rigid_body_count <= 0:
            reasons.append("no_rigid_body")
        if reasons:
            record["status"] = "failed"
            record["errors"] = reasons
            failures.append(f"{name}:{','.join(reasons)}")
        records.append(record)
    if failures:
        raise RuntimeError(
            "[visual_physics] dynamic convex collision audit failed: "
            + "; ".join(failures)
        )
    print(
        f"[visual_physics] convex collision audit ok for {len(records)} objects",
        flush=True,
    )
    return records


def audit_static_collisions(stage, fixture_paths: list[str]) -> list[dict[str, Any]]:
    """Audit ground fixtures separately from room-shell triangle collisions."""
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in fixture_paths:
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            failures.append(f"{path}:missing_root")
            continue
        if root.GetCustomDataByKey("collision:required") is False:
            records.append(
                {
                    "name": path.rsplit("/", 1)[-1],
                    "method": "CollisionDisabled",
                    "collision_prim_count": 0,
                    "convex_hull_count": 0,
                    "rigid_body_count": 0,
                    "status": "skipped",
                }
            )
            continue
        coacd_root = stage.GetPrimAtPath(f"{path}/coacd_collision")
        uses_coacd = bool(coacd_root and coacd_root.IsValid())
        traversal_root = coacd_root if uses_coacd else root
        collision_prims = collect_collision_prims(traversal_root)
        convex_hulls = [
            prim
            for prim in collision_prims
            if prim.IsA(PX.UsdGeom.Mesh)
            and str(
                PX.UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or ""
            )
            == "convexHull"
        ]
        rigid_bodies = len(rigid_body_prims(root)) + int(
            prim_has_api(root, PX.UsdPhysics.RigidBodyAPI)
        )
        errors = []
        if not collision_prims:
            errors.append("no_collision_geometry")
        if uses_coacd and len(convex_hulls) != len(collision_prims):
            errors.append(f"collision_hulls={len(convex_hulls)}/{len(collision_prims)}")
        if rigid_bodies:
            errors.append(f"unexpected_rigid_bodies={rigid_bodies}")
        record = {
            "name": path.rsplit("/", 1)[-1],
            "method": "CoACD" if uses_coacd else "StaticTriangleMesh",
            "collision_prim_count": len(collision_prims),
            "convex_hull_count": len(convex_hulls),
            "rigid_body_count": rigid_bodies,
            "status": "failed" if errors else "passed",
        }
        if errors:
            record["errors"] = errors
            failures.append(f"{record['name']}:{','.join(errors)}")
        records.append(record)
    if failures:
        raise RuntimeError(
            "[visual_physics] static collision audit failed: " + "; ".join(failures)
        )
    print(
        f"[visual_physics] static collision audit ok for {len(records)} fixtures "
        f"(CoACD={sum(record['method'] == 'CoACD' for record in records)})",
        flush=True,
    )
    return records


def log_object_height_changes(stage, object_paths: list[str], before) -> None:
    samples = []
    for path in object_paths[:8]:
        after = world_translation(stage, path)
        start = before.get(path)
        if start is None or after is None:
            continue
        samples.append(f"{path.rsplit('/', 1)[-1]}:{start[2]:.3f}->{after[2]:.3f}")
    if samples:
        print(
            f"[visual_physics] object z after settle: {', '.join(samples)}", flush=True
        )


def audit_support_alignment(
    stage,
    object_paths: list[str],
    support_heights: dict[str, float],
    *,
    tolerance: float = 0.006,
) -> list[dict[str, Any]]:
    if not support_heights:
        return []
    checked = 0
    failures = []
    records: list[dict[str, Any]] = []
    for path in object_paths:
        name = path.rsplit("/", 1)[-1]
        if name not in support_heights:
            continue
        prim = stage.GetPrimAtPath(path)
        bbox = world_bbox(prim) if prim and prim.IsValid() else None
        if bbox is None:
            continue
        bottom_z = float(bbox[0][2])
        expected_z = float(support_heights[name])
        gap = bottom_z - expected_z
        checked += 1
        records.append(
            {
                "name": name,
                "bottom_z": round(bottom_z, 6),
                "support_z": round(expected_z, 6),
                "gap_m": round(gap, 6),
                "tolerance_m": tolerance,
                "status": "passed" if abs(gap) <= tolerance else "failed",
            }
        )
        if abs(gap) > tolerance:
            failures.append(
                f"{name}:gap={gap:.4f} bottom={bottom_z:.4f} support={expected_z:.4f}"
            )
    if failures:
        raise RuntimeError(
            "[visual_physics] support alignment failed: " + "; ".join(failures)
        )
    if checked:
        print(
            f"[visual_physics] support alignment ok for {checked} objects (tol={tolerance:.3f}m)",
            flush=True,
        )
    return records


def stabilize_support_alignment(
    stage,
    object_paths: list[str],
    support_heights: dict[str, float],
    *,
    tolerance: float = 0.006,
    maximum_repair: float = 0.05,
) -> list[dict[str, Any]]:
    """Correct small post-settle contact offsets while preserving XY and rotation."""

    repairs = []
    for path in object_paths:
        name = path.rsplit("/", 1)[-1]
        if name not in support_heights:
            continue
        prim = stage.GetPrimAtPath(path)
        bbox = world_bbox(prim) if prim and prim.IsValid() else None
        if bbox is None:
            continue
        gap = float(bbox[0][2]) - float(support_heights[name])
        if abs(gap) <= tolerance or abs(gap) > maximum_repair:
            continue
        translate = prim.GetAttribute("xformOp:translate")
        current = translate.Get() if translate else None
        if current is None:
            continue
        delta_z = 0.001 - gap
        translate.Set(
            PX.Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]) + delta_z)
        )
        repairs.append(
            {
                "name": name,
                "gap_before_m": round(gap, 6),
                "translation_delta_z_m": round(delta_z, 6),
                "policy": "support_relation_post_settle_contact_stabilization",
            }
        )
    if repairs:
        print(
            f"[visual_physics] stabilized support contact for {len(repairs)} objects",
            flush=True,
        )
    return repairs


# --------------------------------------------------------------------------- #
# material / prim authoring
# --------------------------------------------------------------------------- #


def resolve_texture_path(cfg: dict[str, Any], asset_root: Path | None) -> Path | None:
    texture = cfg.get("texture")
    if not isinstance(texture, dict) or asset_root is None:
        return None
    texture_file = texture.get("texture_file")
    candidates: list[Path] = []
    if isinstance(texture_file, str) and texture_file:
        candidates.extend(
            [
                asset_root / texture_file,
                asset_root / "interdata" / texture_file,
                asset_root.parent / texture_file,
            ]
        )
    lib = texture.get("texture_lib")
    tex_id = str(texture.get("texture_id", 1))
    if isinstance(lib, str) and lib:
        lib_name = lib.replace("\\", "/").split("/")[-1]
        candidates.extend(
            [
                asset_root / "texture_libs" / lib_name / f"{tex_id}.png",
                asset_root / "interdata" / "texture_libs" / lib_name / f"{tex_id}.png",
                asset_root.parent / "texture_libs" / lib_name / f"{tex_id}.png",
                asset_root / lib / f"{tex_id}.png",
                asset_root.parent / lib / f"{tex_id}.png",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def bind_texture_material(stage, prim, texture_path: Path) -> None:
    material_path = f"{prim.GetPath()}_TextureMaterial"
    material = PX.UsdShade.Material.Define(stage, material_path)

    shader = PX.UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", PX.Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", PX.Sdf.ValueTypeNames.Float).Set(0.0)

    reader = PX.UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader_st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", PX.Sdf.ValueTypeNames.Token).Set("st")

    texture = PX.UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", PX.Sdf.ValueTypeNames.Asset).Set(
        PX.Sdf.AssetPath(texture_path.as_posix())
    )
    texture.CreateInput("sourceColorSpace", PX.Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("st", PX.Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", PX.Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", PX.Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    PX.UsdShade.MaterialBindingAPI(prim).Bind(material)


def bind_constant_material(stage, prim, appearance: dict[str, Any]) -> None:
    material_path = f"{prim.GetPath()}_AppearanceMaterial"
    material = PX.UsdShade.Material.Define(stage, material_path)
    shader = PX.UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = vec3(appearance.get("color"), (0.8, 0.8, 0.8))
    shader.CreateInput("diffuseColor", PX.Sdf.ValueTypeNames.Color3f).Set(
        PX.Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", PX.Sdf.ValueTypeNames.Float).Set(
        float(appearance.get("roughness", 0.5))
    )
    shader.CreateInput("metallic", PX.Sdf.ValueTypeNames.Float).Set(
        float(appearance.get("metallic", 0.0))
    )
    shader.CreateInput("opacity", PX.Sdf.ValueTypeNames.Float).Set(
        float(appearance.get("opacity", 1.0))
    )
    shader.CreateInput("ior", PX.Sdf.ValueTypeNames.Float).Set(
        float(appearance.get("ior", 1.5))
    )
    shader.CreateInput("opacityThreshold", PX.Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    PX.UsdShade.MaterialBindingAPI(prim).Bind(material)


def ensure_plane_uv_primvar(plane, texture_scale: Any) -> None:
    sx, sy = 1.0, 1.0
    if isinstance(texture_scale, (list, tuple)) and len(texture_scale) >= 2:
        sx = max(0.01, float(texture_scale[0]))
        sy = max(0.01, float(texture_scale[1]))
    primvar = PX.UsdGeom.PrimvarsAPI(plane.GetPrim()).CreatePrimvar(
        "st", PX.Sdf.ValueTypeNames.TexCoord2fArray, PX.UsdGeom.Tokens.faceVarying
    )
    primvar.Set(
        [
            PX.Gf.Vec2f(0.0, 0.0),
            PX.Gf.Vec2f(sx, 0.0),
            PX.Gf.Vec2f(sx, sy),
            PX.Gf.Vec2f(0.0, sy),
        ]
    )


def texture_score(path: Path) -> tuple[int, str]:
    lower = path.as_posix().lower()
    score = 0
    if "asset_pool_generated_albedo" in lower:
        score += 130
    if any(
        token in lower for token in ("albedo", "basecolor", "base_color", "diffuse")
    ):
        score += 100
    if lower.endswith(
        (
            "/color.png",
            "/color.jpg",
            "/color.jpeg",
            "/color.webp",
            "color.png",
            "color.jpg",
            "color.jpeg",
            "color.webp",
        )
    ):
        score += 40
    if "/semantic_" in lower or path.name.lower().startswith("semantic_"):
        score -= 100
    return score, path.name


def reference_texture_path(abs_usd: Path) -> Path | None:
    roots = [abs_usd.parent, abs_usd.parent / "textures"]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in (
            "asset_pool_generated_albedo.*",
            "*albedo*.*",
            "*basecolor*.*",
            "*base_color*.*",
            "*diffuse*.*",
            "Color.*",
            "color.*",
            "semantic_*.*",
        ):
            candidates.extend(
                path
                for path in root.glob(pattern)
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
    unique = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        return None
    unique.sort(key=texture_score)
    return unique[-1]


def prim_has_uv_texture(prim) -> bool:
    for child in PX.Usd.PrimRange(prim):
        if child.GetTypeName() != "Shader":
            continue
        attr = child.GetAttribute("info:id")
        try:
            if attr and attr.Get() == "UsdUVTexture":
                return True
        except Exception:
            continue
    return False


def ensure_mesh_uv_primvar(mesh_prim) -> None:
    if mesh_prim.GetAttribute("primvars:st").HasAuthoredValueOpinion():
        return
    mesh = PX.UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return
    coords = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    mins = [min(p[i] for p in coords) for i in range(3)]
    maxs = [max(p[i] for p in coords) for i in range(3)]
    spans = [max(maxs[i] - mins[i], 1e-6) for i in range(3)]
    axes = sorted(range(3), key=lambda idx: spans[idx], reverse=True)[:2]
    values = [
        PX.Gf.Vec2f(
            (p[axes[0]] - mins[axes[0]]) / spans[axes[0]],
            (p[axes[1]] - mins[axes[1]]) / spans[axes[1]],
        )
        for p in coords
    ]
    primvars = PX.UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar(
        "st", PX.Sdf.ValueTypeNames.TexCoord2fArray, PX.UsdGeom.Tokens.vertex
    )
    st.Set(values)


def bind_reference_texture_if_missing(stage, prim, abs_usd: Path) -> None:
    if prim_has_uv_texture(prim):
        return
    texture_path = reference_texture_path(abs_usd)
    if texture_path is None:
        return
    mesh_prims = [
        child for child in PX.Usd.PrimRange(prim) if child.GetTypeName() == "Mesh"
    ]
    if not mesh_prims:
        bind_texture_material(stage, prim, texture_path)
        return
    for mesh_prim in mesh_prims:
        ensure_mesh_uv_primvar(mesh_prim)
        bind_texture_material(stage, mesh_prim, texture_path)
    print(
        f"[visual_physics] bound reference texture {texture_path.name} for {prim.GetName()}",
        flush=True,
    )


def define_plane(stage, root_path: str, cfg: dict[str, Any], *, physics: str = "none", asset_root: Path | None = None) -> str:
    prim_path = f"{root_path}/{safe_name(cfg.get('name', 'plane'))}"
    size = cfg.get("size") if isinstance(cfg.get("size"), list) else [1.0, 1.0]
    plane = PX.UsdGeom.Plane.Define(stage, prim_path)
    plane.CreateWidthAttr().Set(float(size[0]))
    plane.CreateLengthAttr().Set(float(size[1]))
    plane.CreateDoubleSidedAttr().Set(True)
    translation = vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    plane_prim = plane.GetPrim()
    collision_required = physics != "none" and bool(cfg.get("collision_enabled", True))
    plane_prim.SetCustomDataByKey("collision:required", collision_required)
    apply_xform(plane_prim, translation, euler)
    texture_path = resolve_texture_path(cfg, asset_root)
    if texture_path is not None:
        texture_cfg = cfg.get("texture") if isinstance(cfg.get("texture"), dict) else {}
        ensure_plane_uv_primvar(plane, texture_cfg.get("texture_scale"))
        bind_texture_material(stage, plane_prim, texture_path)
    elif isinstance(cfg.get("appearance"), dict):
        bind_constant_material(stage, plane_prim, cfg["appearance"])
    if collision_required:
        thickness = float(cfg.get("collision_thickness", 0.02))
        add_box_collision(
            stage,
            prim_path,
            (0.0, 0.0, -0.5 * thickness),
            (float(size[0]), float(size[1]), thickness),
        )
    return prim_path


# --------------------------------------------------------------------------- #
# scene item loading
# --------------------------------------------------------------------------- #


def load_reference(stage, root_path: str, cfg: dict[str, Any], asset_root: Path, *, physics: str = "none", support_z: float | None = None) -> str | None:
    usd_path = cfg.get("path") or cfg.get("usd_path")
    if not isinstance(usd_path, str) or not usd_path:
        return None
    abs_usd = repo_path(usd_path, asset_root)
    if not abs_usd.exists():
        _warn(f"missing USD for {cfg.get('name')}: {abs_usd}")
        return None

    prim_path = f"{root_path}/{safe_name(cfg.get('name', abs_usd.stem))}"
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.SetCustomDataByKey(
        "collision:required",
        physics != "none" and bool(cfg.get("collision_enabled", True)),
    )
    if not prim.GetReferences().AddReference(str(abs_usd)):
        _warn(f"failed to reference {abs_usd} at {prim_path}")
        stage.RemovePrim(prim_path)
        return None
    if not prim or not prim.IsValid():
        _warn(f"failed to create prim {prim_path}")
        return None
    translation = vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
    euler = vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0))
    scale = scale3(cfg.get("scale"))
    apply_xform(prim, translation, euler, scale)
    bind_reference_texture_if_missing(stage, prim, abs_usd)
    if support_z is not None:
        snapped = snap_translation_to_support(prim, translation, euler, scale, support_z)
        if snapped != translation:
            print(
                f"[visual_physics] snapped {cfg.get('name', prim_path)} "
                f"z {translation[2]:.4f}->{snapped[2]:.4f} support_z={support_z:.4f}",
                flush=True,
            )
    if physics == "static":
        requested_collider = str(cfg.get("collider") or "coacd").strip()
        collider_kind = normalized_collider_kind(requested_collider, "coacd")
        coacd_usd = coacd_sidecar(abs_usd, "static_ground")
        if collider_kind in {"trianglemesh", "staticmesh", "mesh", "none"}:
            collision_prims = replace_with_visual_mesh_collision(prim, "none")
            print(
                f"[visual_physics] using exact static triangle collision for "
                f"{cfg.get('name', prim_path)} ({len(collision_prims)} meshes)",
                flush=True,
            )
        elif collider_kind == "meshsimplification":
            collision_prims = replace_with_visual_mesh_collision(prim, "meshSimplification")
        elif collider_kind == "supportbodybbox":
            bounds = world_bbox(prim)
            if bounds is None:
                raise RuntimeError(
                    f"support asset {cfg.get('name', prim_path)} has no finite bounds"
                )
            if cfg.get("support_surface_z") is None:
                raise RuntimeError(
                    f"support asset {cfg.get('name', prim_path)} is missing support_surface_z"
                )
            center, size = support_body_box_geometry(
                *bounds,
                support_surface_z=float(cfg["support_surface_z"]),
                top_clearance=float(cfg.get("support_body_top_clearance", 0.0)),
            )
            clear_physics_apis(prim)
            collision_prims = [
                add_box_collision(
                    stage,
                    prim_path,
                    center,
                    size,
                    proxy_name="support_body_collision_proxy",
                    world_space=True,
                )
            ]
            print(
                f"[visual_physics] using clipped support-body box collision for "
                f"{cfg.get('name', prim_path)} top_z={center[2] + 0.5 * size[2]:.4f}",
                flush=True,
            )
        elif collider_kind in {"coacd", "convexdecomposition"}:
            collision_method = static_convex_collision_method(
                collider_kind, has_coacd_sidecar=coacd_usd is not None
            )
            if collision_method == "coacd":
                assert coacd_usd is not None
                clear_physics_apis(prim)
                _, collision_prims = attach_coacd_collision(
                    stage, prim, coacd_usd, profile="static_ground"
                )
                print(
                    f"[visual_physics] using detailed CoACD ground collision for "
                    f"{cfg.get('name', prim_path)} ({len(collision_prims)} hulls)",
                    flush=True,
                )
            elif collision_method == "physx_convex_decomposition":
                collision_prims = replace_with_visual_mesh_collision(
                    prim, "convexDecomposition"
                )
                print(
                    f"[visual_physics] using PhysX convex decomposition for "
                    f"{cfg.get('name', prim_path)} ({len(collision_prims)} meshes)",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"static asset {cfg.get('name', prim_path)} requests CoACD "
                    "but has no static_ground collision sidecar"
                )
        else:
            raise RuntimeError(
                f"unsupported static collider {requested_collider!r} for "
                f"{cfg.get('name', prim_path)}"
            )
        if not collision_prims:
            raise RuntimeError(
                f"asset {cfg.get('name', prim_path)} has no mesh for static collision"
            )
    elif physics == "dynamic":
        requested_collider = str(cfg.get("collider") or "convexDecomposition")
        collider_kind = normalized_collider_kind(requested_collider, "convexDecomposition")
        if collider_kind not in {"coacd", "convexdecomposition"}:
            raise RuntimeError(
                f"dynamic asset {cfg.get('name', prim_path)} requires "
                f"CoACD or convexDecomposition, got {requested_collider!r}"
            )
        coacd_usd = coacd_sidecar(abs_usd, "dynamic_object")
        if coacd_usd is not None:
            clear_physics_apis(prim)
            _, collision_prims = attach_coacd_collision(stage, prim, coacd_usd, profile="dynamic_object")
            print(
                f"[visual_physics] using CoACD collision for "
                f"{cfg.get('name', prim_path)} ({len(collision_prims)} hulls)",
                flush=True,
            )
        else:
            if collider_kind == "coacd":
                raise RuntimeError(
                    f"dynamic asset {cfg.get('name', prim_path)} declares CoACD "
                    "but has no dynamic_object collision sidecar"
                )
            collision_prims = replace_with_visual_mesh_collision(prim, "convexDecomposition")
        if not collision_prims:
            raise RuntimeError(
                f"asset {cfg.get('name', prim_path)} has no mesh for convex collision"
            )
        make_dynamic_body(prim, float(cfg.get("mass", 0.2)))
    return prim_path


def load_items(stage, root_path: str, items: list[dict[str, Any]], asset_root: Path, *, physics: str = "none", support_heights: dict[str, float] | None = None) -> list[str]:
    loaded = []
    for cfg in items:
        if not isinstance(cfg, dict):
            continue
        item_name = str(cfg.get("name") or "<unnamed>")
        print(f"[visual_physics] loading {root_path}/{item_name}", flush=True)
        if cfg.get("target_class") == "PlaneObject":
            loaded.append(define_plane(stage, root_path, cfg, physics=physics, asset_root=asset_root))
            print(f"[visual_physics] loaded {root_path}/{item_name}", flush=True)
            continue
        support_z = None
        if support_heights is not None:
            name = str(cfg.get("name", ""))
            if name in support_heights:
                support_z = support_heights[name]
            else:
                support_z = support_heights.get(safe_name(name))
        prim_path = load_reference(stage, root_path, cfg, asset_root, physics=physics, support_z=support_z)
        if prim_path:
            loaded.append(prim_path)
            print(f"[visual_physics] loaded {root_path}/{item_name}", flush=True)
    return loaded


def reference_asset_keys(prim) -> list[str]:
    references = prim.GetMetadata("references")
    if references is None:
        return []
    keys = []
    for attr in ("prependedItems", "explicitItems", "addedItems"):
        for reference in getattr(references, attr, []) or []:
            key = asset_key(getattr(reference, "assetPath", None))
            if key:
                keys.append(key)
    return keys


def prim_xform_values(prim) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    if not PX.UsdGeom.Xformable(prim):
        return values
    for op in PX.UsdGeom.Xformable(prim).GetOrderedXformOps():
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


def scene_usda_transform_overrides(scene_cfg) -> dict[str, dict[str, list[float]]]:
    scene_usda = Path(scene_cfg.scene_dir) / "interndata_scene" / "scene.usda"
    if not scene_usda.exists():
        return {}
    try:
        scene_stage = PX.Usd.Stage.Open(str(scene_usda))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        _warn(f"cannot open scene USDA overrides {scene_usda}: {exc}")
        return {}

    overrides: dict[str, dict[str, list[float]]] = {}
    for prim in scene_stage.Traverse():
        values = prim_xform_values(prim)
        if not values:
            continue
        for key in reference_asset_keys(prim):
            overrides[key] = values
    if overrides:
        print(
            f"[visual_physics] loaded {len(overrides)} transform overrides from {scene_usda}",
            flush=True,
        )
    return overrides


def apply_transform_overrides(items: Any, overrides: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not overrides:
        return items if isinstance(items, list) else []
    patched = []
    applied = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cfg = dict(item)
        key = asset_key(cfg.get("path") or cfg.get("usd_path"))
        if key in overrides:
            cfg.update(overrides[key])
            applied.append(str(cfg.get("name", key)))
        patched.append(cfg)
    if applied:
        print(
            f"[visual_physics] applied scene transforms: {', '.join(applied)}",
            flush=True,
        )
    return patched


def scene_usda_fixture_items(scene_cfg, existing_items: Any) -> list[dict[str, Any]]:
    scene_usda = Path(scene_cfg.scene_dir) / "interndata_scene" / "scene.usda"
    if not scene_usda.exists():
        return existing_items if isinstance(existing_items, list) else []
    try:
        scene_stage = PX.Usd.Stage.Open(str(scene_usda))
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        _warn(f"cannot open scene USDA fixtures {scene_usda}: {exc}")
        return existing_items if isinstance(existing_items, list) else []

    patched = list(existing_items) if isinstance(existing_items, list) else []
    existing_keys = {
        key
        for item in patched
        if isinstance(item, dict)
        for key in [asset_key(item.get("path") or item.get("usd_path"))]
        if key
    }
    existing_semantic_keys = {
        key
        for item in patched
        if isinstance(item, dict)
        for key in [semantic_instance_key(item.get("name", ""))]
        if key
    }
    added = []
    skipped_semantic_duplicates = []
    for prim in scene_stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith("/World/Fixtures/"):
            continue
        ref_keys = [key for key in reference_asset_keys(prim) if "/fixtures/" in key]
        if not ref_keys:
            continue
        key = ref_keys[0]
        if key in existing_keys:
            continue
        semantic_key = semantic_instance_key(prim.GetName())
        if semantic_key in existing_semantic_keys:
            skipped_semantic_duplicates.append(prim.GetName())
            continue
        values = prim_xform_values(prim)
        patched.append(
            {
                "name": safe_name(prim.GetName()),
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
            "[visual_physics] skipped scene USDA duplicate fixtures: "
            + ", ".join(skipped_semantic_duplicates),
            flush=True,
        )
    if added:
        print(
            f"[visual_physics] added scene USDA fixtures: {', '.join(added)}", flush=True
        )
    return patched


def fixture_pose_by_name(scene_cfg) -> dict[str, dict[str, Any]]:
    fixtures = scene_cfg.arena.get("fixtures")
    if not isinstance(fixtures, list):
        return {}
    return {
        str(item.get("name")): item
        for item in fixtures
        if isinstance(item, dict) and item.get("name")
    }


def region_by_object_or_name(scene_cfg) -> dict[str, dict[str, Any]]:
    regions = scene_cfg.task.get("regions")
    if not isinstance(regions, list):
        return {}
    out = {}
    for region in regions:
        if not isinstance(region, dict):
            continue
        name = region.get("name")
        obj = region.get("object")
        if isinstance(name, str):
            out[name] = region
        if isinstance(obj, str):
            out[obj] = region
    return out


def support_heights_from_task_regions(scene_cfg) -> dict[str, float]:
    heights = {}
    for region in region_by_object_or_name(scene_cfg).values():
        if not isinstance(region, dict):
            continue
        object_name = region.get("object")
        if not isinstance(object_name, str):
            continue
        random_config = region.get("random_config") or {}
        support_z = random_config.get("support_surface_z", region.get("support_surface_z"))
        if support_z is None:
            continue
        try:
            value = float(support_z)
        except (TypeError, ValueError):
            continue
        heights[object_name] = value
        heights[safe_name(object_name)] = value
    return heights


def support_patch_sizes_from_regions(scene_cfg) -> dict[str, tuple[float, float]]:
    sizes: dict[str, tuple[float, float]] = {}
    for region in region_by_object_or_name(scene_cfg).values():
        if not isinstance(region, dict):
            continue
        object_name = region.get("object")
        size = region.get("size")
        if not isinstance(object_name, str) or not isinstance(size, list) or len(size) < 2:
            continue
        try:
            width, depth = float(size[0]), float(size[1])
        except (TypeError, ValueError):
            continue
        if min(width, depth) <= 0.0:
            continue
        sizes[object_name] = (width, depth)
        sizes[safe_name(object_name)] = (width, depth)
    return sizes


def apply_region_placements(items: Any, scene_cfg) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    fixtures = fixture_pose_by_name(scene_cfg)
    regions = region_by_object_or_name(scene_cfg)
    patched = []
    applied = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cfg = dict(item)
        region = regions.get(str(cfg.get("spawn_region", ""))) or regions.get(
            str(cfg.get("name", ""))
        )
        if isinstance(region, dict):
            random_config = region.get("random_config") or {}
            target_name = region.get("target") or region.get("B") or region.get("parent_fixture")
            target = fixtures.get(str(target_name)) if target_name else None
            target_translation = target.get("translation") if isinstance(target, dict) else None
            translation = resolve_region_translation(region, target_translation)
            if translation is not None:
                cfg["translation"] = translation
                yaw_rotation = random_config.get("yaw_rotation")
                if isinstance(yaw_rotation, list) and yaw_rotation:
                    euler = list(vec3(cfg.get("euler") or cfg.get("rotation"), (0.0, 0.0, 0.0)))
                    euler[2] += float(yaw_rotation[0])
                    cfg["euler"] = euler
                applied.append(str(cfg.get("name", target_name)))
        patched.append(cfg)
    if applied:
        print(
            f"[visual_physics] applied region placements: {', '.join(applied)}",
            flush=True,
        )
    expected = {
        str(item.get("name") or "")
        for item in items
        if isinstance(item, dict) and item.get("spawn_region")
    }
    missing = sorted(expected - set(applied))
    if missing:
        raise RuntimeError(
            "task objects have no resolved support-region placement: "
            + ", ".join(missing)
        )
    return patched


def load_source_interdata_task(scene_cfg) -> dict[str, Any]:
    task_path = Path(scene_cfg.scene_dir) / "interndata_scene" / "task.yaml"
    if not task_path.exists() and Path(scene_cfg.task_path).name == "task.yaml":
        task_path = Path(scene_cfg.task_path)
    if not task_path.exists():
        return {}
    try:
        with task_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:  # pragma: no cover - defensive for hand-edited YAML
        _warn(f"cannot read source support regions {task_path}: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def support_heights_from_regions(scene_cfg) -> dict[str, float]:
    if SETTINGS.no_snap_to_supports:
        return {}
    payload = load_source_interdata_task(scene_cfg)
    regions = payload.get("regions")
    if not isinstance(regions, list):
        return support_heights_from_task_regions(scene_cfg)
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
        heights[safe_name(object_name)] = support_z
    if not heights:
        heights = support_heights_from_task_regions(scene_cfg)
    if heights:
        unique_count = len({name for name in heights if name == safe_name(name)})
        print(
            f"[visual_physics] loaded support heights for {unique_count} objects",
            flush=True,
        )
    return heights


def names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        str(item.get("name"))
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]


def position_from_task(scene_cfg) -> tuple[float, float, float] | None:
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


def robot_pose_from_regions(
    scene_cfg,
    robot: dict[str, Any],
) -> tuple[float, float, float] | None:
    regions = scene_cfg.task.get("regions")
    if not isinstance(regions, list):
        return None
    robot_name = robot.get("name")
    base_euler = vec3(robot.get("euler"), (0.0, 0.0, 0.0))
    fixtures = fixture_pose_by_name(scene_cfg)
    for region in regions:
        if not isinstance(region, dict) or region.get("object") != robot_name:
            continue
        random_config = region.get("random_config")
        if not isinstance(random_config, dict):
            continue
        yaw_rotation = random_config.get("yaw_rotation")
        target_name = region.get("target") or region.get("B")
        target = fixtures.get(str(target_name)) if target_name else None
        target_translation = (
            target.get("translation") if isinstance(target, dict) else None
        )
        translation = resolve_region_translation(region, target_translation)
        if translation is None:
            continue
        yaw_shift = 0.0
        if isinstance(yaw_rotation, list) and yaw_rotation:
            yaw_shift = float(yaw_rotation[0])
        return (translation[0], translation[1], base_euler[2] + yaw_shift)
    return None


def robot_mount_support_height(
    scene_cfg,
    robot: dict[str, Any],
) -> float:
    robot_name = str(robot.get("name") or "")
    fixtures = fixture_pose_by_name(scene_cfg)
    for region in scene_cfg.task.get("regions", []):
        if not isinstance(region, dict) or region.get("object") != robot_name:
            continue
        target_name = region.get("target") or region.get("B")
        target = fixtures.get(str(target_name)) if target_name else None
        if not isinstance(target, dict) or target.get("support_surface_z") is None:
            raise ValueError(
                f"support-mounted robot {robot_name!r} requires its region target "
                "fixture to declare support_surface_z"
            )
        shift_z = range_midpoint3(
            (region.get("random_config") or {}).get("pos_range")
        )[2]
        return float(target["support_surface_z"]) + shift_z
    raise ValueError(
        f"support-mounted robot {robot_name!r} requires a task placement region"
    )


def robot_visual_items(
    scene_cfg,
    robots: list[dict[str, Any]],
    room: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not robots:
        return [], {}
    visual_robots = []
    support_heights = {}
    for robot in robots:
        profile = load_robot_profile_for_task(robot, Path(scene_cfg.task_path))
        pose = robot_pose_from_regions(scene_cfg, robot)
        yaw_is_degrees = True
        if pose is None:
            task_pose = position_from_task(scene_cfg)
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
        cfg["path"] = str(resolve_robot_asset_path(profile))
        cfg["target_class"] = profile.target_class
        cfg["translation"] = [x, y, room["floor_z"]]
        euler = list(vec3(cfg.get("euler"), (0.0, 0.0, 0.0)))
        euler[2] = yaw_deg
        cfg["euler"] = euler
        visual_robots.append(cfg)
        if profile.placement.family is PlacementFamily.SUPPORT_MOUNTED:
            support_heights[str(cfg.get("name") or "")] = robot_mount_support_height(
                scene_cfg, robot
            )
        print(
            f"[visual_physics] robot visual pose x={x:.3f} y={y:.3f} yaw_deg={yaw_deg:.1f}",
            flush=True,
        )
    return visual_robots, support_heights


def bounds_from_payload(scene_cfg) -> tuple[float, float, float, float, float, float]:
    points = []
    for cfg in list(scene_cfg.arena.get("fixtures", [])) + list(scene_cfg.task.get("objects", [])):
        if not isinstance(cfg, dict):
            continue
        translation = vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
        size = cfg.get("size") if isinstance(cfg.get("size"), list) else [0.5, 0.5, 0.5]
        sx = float(size[0]) if len(size) > 0 else 0.5
        sy = float(size[1]) if len(size) > 1 else 0.5
        sz = float(size[2]) if len(size) > 2 else 0.5
        points.append(
            (
                translation[0] - sx * 0.5,
                translation[1] - sy * 0.5,
                translation[2] - sz * 0.5,
            )
        )
        points.append(
            (
                translation[0] + sx * 0.5,
                translation[1] + sy * 0.5,
                translation[2] + sz * 0.5,
            )
        )
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


# --------------------------------------------------------------------------- #
# region placement (world coordinate resolution)
# --------------------------------------------------------------------------- #


def range_midpoint3(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        return [0.0, 0.0, 0.0]
    lower = value[0] if isinstance(value[0], list) else []
    upper = value[1] if len(value) > 1 and isinstance(value[1], list) else lower
    return [
        (number_at(lower, index) + number_at(upper, index)) * 0.5
        for index in range(3)
    ]


def resolve_region_translation(region: dict[str, Any], target_translation: Any) -> list[float] | None:
    random_config = region.get("random_config")
    if not isinstance(random_config, dict):
        random_config = {}
    shift = range_midpoint3(random_config.get("pos_range"))
    support_z = random_config.get("support_surface_z", region.get("support_surface_z"))
    runtime = region.get("runtime_placement")
    if not isinstance(runtime, dict):
        runtime = {}
    frame = str(runtime.get("frame") or "")

    if frame == "parent_world_xy_offset" and valid_vec(target_translation, 3):
        offset = runtime.get("offset_xy")
        if valid_vec(offset, 2):
            return [
                float(target_translation[0]) + float(offset[0]) + shift[0],
                float(target_translation[1]) + float(offset[1]) + shift[1],
                resolved_z(support_z, float(target_translation[2]) + shift[2]),
            ]

    if frame == "isaac_room_center_xy":
        center = runtime.get("center_xy")
        if valid_vec(center, 2):
            return [
                float(center[0]) + shift[0],
                float(center[1]) + shift[1],
                resolved_z(support_z, shift[2]),
            ]

    if valid_vec(target_translation, 3):
        return [
            float(target_translation[0]) + shift[0],
            float(target_translation[1]) + shift[1],
            resolved_z(support_z, float(target_translation[2]) + shift[2]),
        ]

    center = region.get("center")
    if valid_vec(center, 2):
        return [
            float(center[0]) + shift[0],
            float(center[1]) + shift[1],
            resolved_z(support_z, shift[2]),
        ]
    return None


def valid_vec(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= length


def number_at(value: list[Any], index: int) -> float:
    try:
        return float(value[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def resolved_z(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------- #
# cameras / lighting / rendering
# --------------------------------------------------------------------------- #


def setup_lighting(stage, scene_cfg) -> None:
    dome = PX.UsdLux.DomeLight.Define(stage, PX.Sdf.Path("/World/Lights/DomeLight"))
    dome.CreateIntensityAttr(float(os.environ.get("INTERDATA_DOME_LIGHT_INTENSITY", "1100")))
    distant = PX.UsdLux.DistantLight.Define(stage, PX.Sdf.Path("/World/Lights/KeyLight"))
    distant.CreateIntensityAttr(float(os.environ.get("INTERDATA_KEY_LIGHT_INTENSITY", "800")))
    distant.CreateAngleAttr(float(os.environ.get("INTERDATA_KEY_LIGHT_ANGLE", "3")))
    apply_xform(distant.GetPrim(), (0.0, 0.0, 5.0), (-50.0, 0.0, 35.0))
    window_intensity = float(os.environ.get("INTERDATA_WINDOW_LIGHT_INTENSITY", "240"))
    for index, cfg in enumerate(scene_cfg.arena.get("fixtures", [])):
        if not isinstance(cfg, dict) or cfg.get("role") != "window_glass":
            continue
        translation = vec3(cfg.get("translation"), (0.0, 0.0, 1.5))
        inward = vec3(cfg.get("inward_normal"), (0.0, 1.0, 0.0))
        light = PX.UsdLux.SphereLight.Define(stage, PX.Sdf.Path(f"/World/Lights/WindowLight_{index}"))
        light.CreateIntensityAttr(window_intensity)
        light.CreateRadiusAttr(0.3)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(5500.0)
        apply_xform(
            light.GetPrim(),
            (
                translation[0] + inward[0] * 0.25,
                translation[1] + inward[1] * 0.25,
                translation[2],
            ),
            (0.0, 0.0, 0.0),
        )


def room_frame(scene_cfg, derived_bounds) -> dict[str, float]:
    floor = None
    for cfg in scene_cfg.arena.get("fixtures", []):
        if not isinstance(cfg, dict):
            continue
        if cfg.get("target_class") == "PlaneObject" and str(cfg.get("name", "")).lower() == "floor":
            floor = cfg
            break
    if floor is not None:
        translation = vec3(floor.get("translation"), (0.0, 0.0, 0.0))
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
        min_x, min_y, min_z, max_x, max_y, max_z = derived_bounds
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
        translation = vec3(cfg.get("translation"), (0.0, 0.0, 0.0))
        size = cfg.get("size") if isinstance(cfg.get("size"), list) else [0.0, 0.0]
        if len(size) >= 2:
            wall_tops.append(translation[2] + float(size[1]) * 0.5)
    if wall_tops:
        room["height"] = max(wall_tops) - room["floor_z"]
    return room


def camera_obstacle_rects(scene_cfg) -> list[tuple[float, float, float, float]]:
    rects = []
    for cfg in scene_cfg.arena.get("fixtures", []):
        if not isinstance(cfg, dict) or cfg.get("target_class") == "PlaneObject":
            continue
        translation = cfg.get("translation")
        extents = cfg.get("asset_world_extents")
        if (
            not isinstance(translation, list)
            or len(translation) < 2
            or not isinstance(extents, list)
            or len(extents) < 2
        ):
            continue
        local_x, local_y = float(extents[0]), float(extents[1])
        euler = cfg.get("euler") if isinstance(cfg.get("euler"), list) else []
        yaw = float(euler[2]) if len(euler) >= 3 else 0.0
        radians = math.radians(yaw)
        size_x = abs(local_x * math.cos(radians)) + abs(local_y * math.sin(radians))
        size_y = abs(local_x * math.sin(radians)) + abs(local_y * math.cos(radians))
        x, y = float(translation[0]), float(translation[1])
        rects.append(
            (
                x - size_x * 0.5,
                y - size_y * 0.5,
                x + size_x * 0.5,
                y + size_y * 0.5,
            )
        )
    return rects


def camera_obstacle_clearance(point, rects, *, padding: float) -> float:
    if not rects:
        return float("inf")
    clearances = []
    for rect in rects:
        expanded = (rect[0] - padding, rect[1] - padding, rect[2] + padding, rect[3] + padding)
        if expanded[0] <= point[0] <= expanded[2] and expanded[1] <= point[1] <= expanded[3]:
            clearances.append(
                -min(
                    point[0] - expanded[0],
                    expanded[2] - point[0],
                    point[1] - expanded[1],
                    expanded[3] - point[1],
                )
            )
            continue
        dx = max(expanded[0] - point[0], point[0] - expanded[2], 0.0)
        dy = max(expanded[1] - point[1], point[1] - expanded[3], 0.0)
        clearances.append(math.hypot(dx, dy))
    return min(clearances)


def opening_camera_views(room: dict[str, float], scene_cfg):
    metadata = scene_cfg.arena.get("metadata") if isinstance(scene_cfg.arena.get("metadata"), dict) else {}
    openings = metadata.get("architectural_openings") or []
    if not isinstance(openings, list):
        return []
    min_x, max_x = room["min_x"], room["max_x"]
    min_y, max_y = room["min_y"], room["max_y"]
    floor_z = room["floor_z"]
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    room_size = metadata.get("room_size_m")
    if isinstance(room_size, list) and len(room_size) >= 2:
        width, depth = float(room_size[0]), float(room_size[1])
    else:
        width, depth = max_x - min_x, max_y - min_y
    interior_min_x, interior_max_x = cx - width * 0.5, cx + width * 0.5
    interior_min_y, interior_max_y = cy - depth * 0.5, cy + depth * 0.5
    obstacle_rects = camera_obstacle_rects(scene_cfg)

    def choose_eye(candidates):
        def score(candidate):
            point, _ = candidate
            boundary_clearance = min(
                point[0] - min_x,
                max_x - point[0],
                point[1] - min_y,
                max_y - point[1],
            )
            return min(
                boundary_clearance,
                camera_obstacle_clearance(point, obstacle_rects, padding=0.3),
            )

        return max(candidates, key=score)

    def threshold_and_inward(opening: dict[str, Any]):
        side = str(opening.get("wall") or "")
        offset = float(opening.get("center_offset_m") or 0.0)
        if side == "south":
            return (interior_min_x + offset, interior_min_y), (0.0, 1.0)
        if side == "north":
            return (interior_max_x - offset, interior_max_y), (0.0, -1.0)
        if side == "east":
            return (interior_max_x, interior_min_y + offset), (-1.0, 0.0)
        if side == "west":
            return (interior_min_x, interior_max_y - offset), (1.0, 0.0)
        return None

    views = []
    doors = [item for item in openings if item.get("kind") == "door"]
    if doors:
        door = max(doors, key=lambda item: float(item.get("width_m") or 0.0))
        resolved = threshold_and_inward(door)
        if resolved is not None:
            threshold, inward = resolved
            inward_span = depth if abs(inward[1]) > 0.0 else width
            tangent_span = width if abs(inward[1]) > 0.0 else depth
            tangent = (inward[1], -inward[0])
            base_distance = min(1.1, max(0.82, 0.18 * inward_span))
            candidates = [
                (
                    (
                        threshold[0] + inward[0] * distance + tangent[0] * shift,
                        threshold[1] + inward[1] * distance + tangent[1] * shift,
                    ),
                    tangent,
                )
                for distance in (base_distance, min(base_distance + 0.2, 1.3))
                for shift in (0.0, 0.08 * tangent_span, -0.08 * tangent_span)
            ]
            eye_xy, _ = choose_eye(candidates)
            eye = (eye_xy[0], eye_xy[1], floor_z + min(1.65, room["height"] * 0.62))
            target_distance = min(inward_span * 0.75, max(2.0, inward_span * 0.58))
            target = (
                eye_xy[0] + inward[0] * target_distance,
                eye_xy[1] + inward[1] * target_distance,
                floor_z + min(0.95, room["height"] * 0.36),
            )
            views.append(("doorway_interior", eye, target, (0.0, 0.0, 1.0)))

    windows = [item for item in openings if item.get("kind") == "window"]
    if windows:
        window = max(windows, key=lambda item: float(item.get("width_m") or 0.0) * float(item.get("height_m") or 0.0))
        resolved = threshold_and_inward(window)
        if resolved is not None:
            threshold, inward = resolved
            inward_span = depth if abs(inward[1]) > 0.0 else width
            tangent_span = width if abs(inward[1]) > 0.0 else depth
            clockwise_tangent = (inward[1], -inward[0])
            base_distance = min(1.0, max(0.75, 0.14 * inward_span))
            candidates = []
            for distance in (base_distance, min(base_distance + 0.2, 1.2)):
                for tangent_sign in (1.0, -1.0, 0.0):
                    tangent = (
                        clockwise_tangent[0] * tangent_sign,
                        clockwise_tangent[1] * tangent_sign,
                    )
                    shift = 0.18 * tangent_span if tangent_sign else 0.0
                    candidates.append(
                        (
                            (
                                threshold[0] + inward[0] * distance + tangent[0] * shift,
                                threshold[1] + inward[1] * distance + tangent[1] * shift,
                            ),
                            tangent,
                        )
                    )
            eye_xy, eye_tangent = choose_eye(candidates)
            eye = (eye_xy[0], eye_xy[1], floor_z + min(1.65, room["height"] * 0.66))
            target_distance = min(inward_span * 0.75, max(2.0, inward_span * 0.55))
            target = (
                eye_xy[0] + inward[0] * target_distance - eye_tangent[0] * (0.03 * tangent_span),
                eye_xy[1] + inward[1] * target_distance - eye_tangent[1] * (0.03 * tangent_span),
                floor_z + min(0.95, room["height"] * 0.36),
            )
            views.append(("window_overview", eye, target, (0.0, 0.0, 1.0)))
    return views


def camera_views(room: dict[str, float], scene_cfg, stage=None):
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
    target_z = floor_z + min(max(room_height * 0.22, 0.62), room_height - 0.75)
    views = [
        (
            "diagonal_overview",
            (
                cx - width * 0.44,
                cy - depth * 0.44,
                floor_z + room_height + scene_radius * 0.78,
            ),
            (cx, cy, target_z),
            (0.0, 0.0, 1.0),
        ),
    ]
    opening_views = opening_camera_views(room, scene_cfg)
    views.extend(opening_views)
    opening_view_names = {view[0] for view in opening_views}
    eye_z = floor_z + min(max(room_height * 0.62, 1.55), room_height - 0.15)
    views.append(
        (
            "room_interior",
            (cx, min_y + depth * 0.12, eye_z),
            (cx, cy + depth * 0.04, target_z),
            (0.0, 0.0, 1.0),
        )
    )
    if "doorway_interior" not in opening_view_names:
        views.append(
            (
                "south_interior",
                (cx, min_y + depth * 0.12, eye_z),
                (cx, cy, target_z),
                (0.0, 0.0, 1.0),
            )
        )
    if "window_overview" not in opening_view_names:
        views.append(
            (
                "east_interior",
                (max_x - width * 0.12, cy, eye_z),
                (cx, cy, target_z),
                (0.0, 0.0, 1.0),
            )
        )
    extra = os.environ.get("TASK_RENDER_EXTRA_VIEWS_JSON")
    if extra:
        try:
            for item in json.loads(extra):
                name = str(item["name"])
                if item.get("template"):
                    if stage is None:
                        raise ValueError("camera template requires a loaded USD stage")
                    robot_name = str(item["robot"])
                    target_name = str(item["target"])
                    robot_xyz = world_translation(
                        stage, f"/World/Robots/{safe_name(robot_name)}"
                    )
                    target_xyz = world_translation(
                        stage, f"/World/Objects/{safe_name(target_name)}"
                    )
                    robot_cfg = next(
                        (
                            value
                            for value in scene_cfg.task.get("robots", [])
                            if value.get("name") == robot_name
                        ),
                        {},
                    )
                    pose = robot_pose_from_regions(scene_cfg, robot_cfg)
                    yaw_deg = (
                        float(pose[2])
                        if pose is not None
                        else float(vec3(robot_cfg.get("euler"), (0.0, 0.0, 0.0))[2])
                    )
                    if robot_xyz is None or target_xyz is None:
                        raise ValueError(
                            f"camera template subjects are missing: robot={robot_name} "
                            f"target={target_name}"
                        )
                    resolved = resolve_camera_template_pose(
                        str(item["template"]),
                        robot_xyz,
                        yaw_deg,
                        target_xyz,
                        item.get("template_params"),
                        [
                            room["min_x"],
                            room["max_x"],
                            room["min_y"],
                            room["max_y"],
                        ],
                    )
                    eye = tuple(resolved["eye"])
                    target = tuple(resolved["target"])
                else:
                    eye = tuple(float(v) for v in item["eye"])
                    target = tuple(float(v) for v in item["target"])
                up = tuple(float(v) for v in item.get("up", [0.0, 0.0, 1.0]))
                views.append((name, eye, target, up))
        except Exception as exc:
            print(
                f"[visual_physics] failed to parse TASK_RENDER_EXTRA_VIEWS_JSON: {exc}",
                flush=True,
            )
    if SETTINGS.single_view:
        views = [view for view in views if view[0] == SETTINGS.single_view]
    return views


def expected_rgb_path(view_dir: Path) -> Path:
    return view_dir / "rgb_0000.png"


def capture_rgb(rep, writer, render_product, view_dir: Path) -> Path:
    expected_rgb = expected_rgb_path(view_dir)
    if expected_rgb.exists():
        expected_rgb.unlink()
    attached = False
    try:
        writer.attach([render_product])
        attached = True
        for _ in range(3):
            rep.orchestrator.step(
                rt_subframes=max(1, int(SETTINGS.rt_subframes)),
                pause_timeline=False,
            )
            rep.orchestrator.wait_until_complete()
            if expected_rgb.exists():
                break
    finally:
        try:
            if attached:
                writer.detach()
        finally:
            render_product.destroy()
    if not expected_rgb.exists():
        raise RuntimeError(f"render did not produce {expected_rgb}")
    validate_png_not_blank(expected_rgb)
    return expected_rgb


def png_rgb_std(image_path: Path) -> float:
    data = image_path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"rendered image is not PNG: {image_path}")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat_chunks = []
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
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
    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type not in {2, 6}
        or interlace != 0
    ):
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
    for _ in range(int(height)):
        filter_type = raw[pos]
        pos += 1
        current = bytearray(raw[pos:pos + row_bytes])
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
            luma = (
                0.2126 * current[base]
                + 0.7152 * current[base + 1]
                + 0.0722 * current[base + 2]
            ) / 255.0
            total += luma
            total_sq += luma * luma
            n += 1
        prev = current
    if n == 0:
        raise RuntimeError(f"rendered image has no pixels: {image_path}")
    mean = total / n
    variance = max(total_sq / n - mean * mean, 0.0)
    return math.sqrt(variance)


def validate_png_not_blank(image_path: Path) -> None:
    try:
        byte_count = image_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"cannot read rendered image {image_path}: {exc}") from exc
    if byte_count < 1024:
        raise RuntimeError(f"rendered image is unexpectedly small: {image_path}")
    rgb_std = png_rgb_std(image_path)
    if rgb_std < 0.015:
        raise RuntimeError(
            f"rendered image appears blank: {image_path} rgb_std={rgb_std:.5f}"
        )


def define_topdown_camera(stage, room: dict[str, float]) -> str:
    min_x = room["min_x"]
    max_x = room["max_x"]
    min_y = room["min_y"]
    max_y = room["max_y"]
    floor_z = room["floor_z"]
    room_height = room["height"]
    width = max(max_x - min_x, 1.0)
    depth = max(max_y - min_y, 1.0)
    aspect = max(float(SETTINGS.width) / max(float(SETTINGS.height), 1.0), 1.0)
    frustum_width = max(width, depth * aspect) * 1.18
    frustum_height = frustum_width / aspect
    camera_path = "/World/Cameras/TopDownCamera"
    camera = PX.UsdGeom.Camera.Define(stage, PX.Sdf.Path(camera_path))
    camera.CreateProjectionAttr().Set(PX.UsdGeom.Tokens.orthographic)
    camera.CreateHorizontalApertureAttr().Set(float(frustum_width * 10.0))
    camera.CreateVerticalApertureAttr().Set(float(frustum_height * 10.0))
    camera.CreateClippingRangeAttr().Set(PX.Gf.Vec2f(0.01, 1000.0))
    apply_xform(
        camera.GetPrim(),
        (
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            floor_z + room_height + max(width, depth) * 1.35,
        ),
        (0.0, 0.0, 0.0),
    )
    return camera_path


def _render_views_with_replicator(rep, stage, scene_cfg, output_dir: Path) -> list[Path]:
    room = room_frame(scene_cfg, bounds_from_payload(scene_cfg))
    focal_length_mm = float(os.environ.get("TASK_RENDER_FOCAL_LENGTH_MM", "16.0"))
    if not math.isfinite(focal_length_mm) or focal_length_mm <= 0.0:
        raise ValueError("TASK_RENDER_FOCAL_LENGTH_MM must be a positive number")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    print(
        "[visual_physics] room frame "
        f"x=({room['min_x']:.3f},{room['max_x']:.3f}) "
        f"y=({room['min_y']:.3f},{room['max_y']:.3f}) "
        f"z=({room['floor_z']:.3f},{room['floor_z'] + room['height']:.3f})",
        flush=True,
    )
    for view_name, eye, target, up in camera_views(room, scene_cfg, stage):
        view_dir = output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        camera = rep.create.camera(
            name=safe_name(view_name),
            position=eye,
            look_at=target,
            look_at_up_axis=up,
            focal_length=focal_length_mm,
            clipping_range=(0.01, 1000.0),
        )
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(view_dir),
            rgb=True,
            image_output_format="png",
            frame_padding=4,
        )
        render_product = rep.create.render_product(
            camera,
            (int(SETTINGS.width), int(SETTINGS.height)),
            force_new=True,
        )
        rgb_path = capture_rgb(rep, writer, render_product, view_dir)
        rendered.append(view_dir)
        print(
            f"[visual_physics] rendered {view_name} eye={eye} target={target} -> {rgb_path}",
            flush=True,
        )
    if SETTINGS.single_view and SETTINGS.single_view != "topdown":
        return rendered
    topdown_camera = define_topdown_camera(stage, room)
    view_dir = output_dir / "topdown"
    view_dir.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(view_dir),
        rgb=True,
        image_output_format="png",
        frame_padding=4,
    )
    render_product = rep.create.render_product(
        topdown_camera,
        (int(SETTINGS.width), int(SETTINGS.height)),
        force_new=True,
    )
    rgb_path = capture_rgb(rep, writer, render_product, view_dir)
    rendered.append(view_dir)
    print(
        f"[visual_physics] rendered topdown camera={topdown_camera} -> {rgb_path}",
        flush=True,
    )
    return rendered


def _finish_replicator(rep) -> None:
    rep.orchestrator.stop()
    rep.orchestrator.wait_until_complete()
    data_queue = rep.backends.io_queue.data_queue
    if data_queue.q is None:
        return
    # This hook otherwise joins workers after Kit has already torn down Carb.
    atexit.unregister(data_queue.destroy)
    data_queue.destroy()


def render_views(stage, scene_cfg, output_dir: Path) -> list[Path]:
    import omni.replicator.core as rep  # pylint: disable=import-outside-toplevel

    try:
        return _render_views_with_replicator(rep, stage, scene_cfg, output_dir)
    finally:
        _finish_replicator(rep)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def author_support_patches(stage, object_paths: list[str], *, support_heights: dict[str, float], support_patch_sizes: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    thickness = max(0.001, float(os.environ.get("TASK_RENDER_SUPPORT_PATCH_M", "0.01")))
    margin = max(0.0, float(os.environ.get("TASK_RENDER_SUPPORT_PATCH_MARGIN_M", "0.01")))
    requested = []
    for path in object_paths:
        name = path.rsplit("/", 1)[-1]
        support_z = support_heights.get(name)
        size = support_patch_sizes.get(name)
        translation = world_translation(stage, path)
        if support_z is None or size is None or translation is None:
            continue
        prim = stage.GetPrimAtPath(path)
        bbox = world_bbox(prim) if prim and prim.IsValid() else None
        footprint_x = float(bbox[1][0]) + 2.0 * margin if bbox else 0.0
        footprint_y = float(bbox[1][1]) + 2.0 * margin if bbox else 0.0
        patch_x = max(float(size[0]), footprint_x)
        patch_y = max(float(size[1]), footprint_y)
        requested.append(
            {
                "objects": [name],
                "support_z": float(support_z),
                "min_x": float(translation[0]) - patch_x * 0.5,
                "max_x": float(translation[0]) + patch_x * 0.5,
                "min_y": float(translation[1]) - patch_y * 0.5,
                "max_y": float(translation[1]) + patch_y * 0.5,
            }
        )

    # Coplanar overlapping boxes create duplicate PhysX contacts and can eject
    # light objects from a shelf during settling.  Merge only intersecting
    # patches; disjoint supports remain independent.
    patches = merge_support_patch_bounds(requested)
    records = []
    for index, patch in enumerate(patches, start=1):
        center_x = 0.5 * (patch["min_x"] + patch["max_x"])
        center_y = 0.5 * (patch["min_y"] + patch["max_y"])
        size_x = patch["max_x"] - patch["min_x"]
        size_y = patch["max_y"] - patch["min_y"]
        patch_path = f"/World/SupportPatches/patch_{index:03d}"
        add_box_collision(
            stage,
            patch_path,
            (center_x, center_y, patch["support_z"] - 0.5 * thickness),
            (size_x, size_y, thickness),
        )
        records.append(
            {
                "objects": patch["objects"],
                "support_z": round(patch["support_z"], 6),
                "size_xy": [round(size_x, 6), round(size_y, 6)],
                "path": patch_path,
            }
        )
    print(f"[visual_physics] authored {len(records)} support collision patches", flush=True)
    return records


def merge_support_patch_bounds(patches: list[dict[str, object]], *, height_tolerance: float = 0.001, overlap_tolerance: float = 0.001) -> list[dict[str, object]]:
    merged = [dict(patch) for patch in patches]
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(merged):
            for right_index in range(left_index + 1, len(merged)):
                right = merged[right_index]
                same_height = abs(float(left["support_z"]) - float(right["support_z"])) <= height_tolerance
                overlaps = not (
                    float(left["max_x"]) + overlap_tolerance < float(right["min_x"])
                    or float(right["max_x"]) + overlap_tolerance < float(left["min_x"])
                    or float(left["max_y"]) + overlap_tolerance < float(right["min_y"])
                    or float(right["max_y"]) + overlap_tolerance < float(left["min_y"])
                )
                if not same_height or not overlaps:
                    continue
                left["objects"] = sorted(set(left.get("objects", [])) | set(right.get("objects", [])))
                for key in ("min_x", "min_y"):
                    left[key] = min(float(left[key]), float(right[key]))
                for key in ("max_x", "max_y"):
                    left[key] = max(float(left[key]), float(right[key]))
                merged.pop(right_index)
                changed = True
                break
            if changed:
                break
    return merged


def _render_main(app, settings: SimpleNamespace) -> int:
    global OMNI_USD, WORLD_CLS, ENABLE_EXTENSION
    OMNI_USD = importlib.import_module("omni.usd")
    from omni.isaac.core import World as _WorldCls  # pylint: disable=import-outside-toplevel
    from omni.isaac.core.utils.extensions import enable_extension as _enable  # pylint: disable=import-outside-toplevel
    WORLD_CLS = _WorldCls
    ENABLE_EXTENSION = _enable

    print(f"[visual_physics] starting with task={settings.task}", flush=True)
    ENABLE_EXTENSION("omni.replicator.core")
    for _ in range(10):
        app.update()

    physics_dt = 1.0 / 30.0
    world = WORLD_CLS(physics_dt=physics_dt, rendering_dt=physics_dt, stage_units_in_meters=1.0)
    if not settings.no_physics:
        physics_context = world.get_physics_context()
        physics_context.enable_gpu_dynamics(False)
        physics_context.set_broadphase_type("MBP")
        physics_context.set_gravity(float(settings.gravity_mps2))
        print(
            "[visual_physics] using CPU PhysX for deterministic concave-static "
            "and CoACD rigid collisions with "
            f"gravity={float(settings.gravity_mps2):.3f} m/s^2",
            flush=True,
        )
    stage = OMNI_USD.get_context().get_stage()
    PX.UsdGeom.SetStageUpAxis(stage, PX.UsdGeom.Tokens.z)

    include_robot = settings.include_robot
    print("[visual_physics] loading compatible scene config", flush=True)
    scene_cfg = load_scene_config(settings.task, include_robot=include_robot)
    print(
        f"[visual_physics] config loaded source={scene_cfg.source} asset_root={scene_cfg.asset_root}",
        flush=True,
    )
    print("[visual_physics] setup lighting", flush=True)
    setup_lighting(stage, scene_cfg)
    print("[visual_physics] load overrides", flush=True)
    transform_overrides = scene_usda_transform_overrides(scene_cfg)
    print("[visual_physics] load support heights", flush=True)
    support_heights = support_heights_from_regions(scene_cfg)
    support_patch_sizes = support_patch_sizes_from_regions(scene_cfg)
    print("[visual_physics] collect fixtures", flush=True)
    fixtures = scene_usda_fixture_items(scene_cfg, scene_cfg.arena.get("fixtures", []))
    fixtures = apply_transform_overrides(fixtures, transform_overrides)
    objects = apply_transform_overrides(scene_cfg.task.get("objects", []), transform_overrides)
    objects = apply_region_placements(objects, scene_cfg)
    static_objects, dynamic_objects = partition_objects_by_physics(objects)
    render_scene_cfg = SimpleNamespace(
        **{
            **vars(scene_cfg),
            "arena": {**scene_cfg.arena, "fixtures": fixtures},
            "task": {**scene_cfg.task, "objects": objects},
        }
    )
    room = room_frame(render_scene_cfg, bounds_from_payload(render_scene_cfg))
    robots = scene_cfg.task.get("robots", []) if include_robot else []
    visual_robots, robot_support_heights = robot_visual_items(
        render_scene_cfg,
        robots if isinstance(robots, list) else [],
        room,
    )
    print(
        f"[visual_physics] loading items fixtures={len(fixtures) if isinstance(fixtures, list) else 0} "
        f"objects={len(objects) if isinstance(objects, list) else 0}",
        flush=True,
    )
    loaded_arena = load_items(
        stage,
        "/World/Arena",
        fixtures if isinstance(fixtures, list) else [],
        scene_cfg.asset_root,
        physics="none" if settings.no_physics else "static",
    )
    loaded_static_objects = load_items(
        stage,
        "/World/Objects",
        static_objects,
        scene_cfg.asset_root,
        physics="none" if settings.no_physics else "static",
    )
    loaded_dynamic_objects = load_items(
        stage,
        "/World/Objects",
        dynamic_objects,
        scene_cfg.asset_root,
        physics="none" if settings.no_physics else "dynamic",
        support_heights=support_heights,
    )
    loaded_objects = loaded_static_objects + loaded_dynamic_objects
    support_patches = (
        []
        if settings.no_physics
        else author_support_patches(
            stage,
            loaded_dynamic_objects,
            support_heights=support_heights,
            support_patch_sizes=support_patch_sizes,
        )
    )
    loaded_robots = load_items(
        stage,
        "/World/Robots",
        visual_robots,
        scene_cfg.asset_root,
        support_heights=robot_support_heights,
    )
    expected_fixture_names = {safe_name(name) for name in names(fixtures)}
    loaded_fixture_names = {path.rsplit("/", 1)[-1] for path in loaded_arena}
    missing_fixtures = sorted(expected_fixture_names - loaded_fixture_names)
    expected_object_names = {safe_name(name) for name in names(objects)}
    loaded_object_names = {path.rsplit("/", 1)[-1] for path in loaded_objects}
    missing_objects = sorted(expected_object_names - loaded_object_names)
    if missing_objects:
        raise RuntimeError(
            "Layer-2 assets were not loaded into the rendered stage: "
            + ", ".join(missing_objects)
        )

    object_pose_before_settle = {
        path: world_translation(stage, path) for path in loaded_dynamic_objects
    }
    if not settings.no_physics:
        static_collision_audit = audit_static_collisions(
            stage, loaded_arena + loaded_static_objects
        )
        wall_attachment_audit = audit_wall_attachments(stage, loaded_arena, fixtures)
        convex_collision_audit = audit_dynamic_collisions(
            stage, loaded_dynamic_objects
        )
        print(
            "[visual_physics] resetting world physics after loading collision proxies",
            flush=True,
        )
        world.reset()
    else:
        static_collision_audit = []
        wall_attachment_audit = []
        convex_collision_audit = []
    settle_steps = max(0, int(math.ceil(float(settings.settle_seconds) / physics_dt)))
    print(
        f"[visual_physics] settling physics for {settle_steps} steps "
        f"({settle_steps * physics_dt:.3f}s)",
        flush=True,
    )
    for _ in range(settle_steps):
        world.step(render=True)
    if not settings.no_physics:
        log_object_height_changes(
            stage, loaded_dynamic_objects, object_pose_before_settle
        )
    output_dir = repo_path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    support_alignment_repairs = (
        stabilize_support_alignment(stage, loaded_dynamic_objects, support_heights)
        if not settings.no_physics
        else []
    )
    print("[visual_physics] auditing settled support alignment", flush=True)
    try:
        support_alignment_audit = audit_support_alignment(
            stage, loaded_dynamic_objects, support_heights
        )
    except Exception as exc:
        (output_dir / "physics_audit.json").write_text(
            json.dumps(
                {
                    "physics_enabled": not settings.no_physics,
                    "gravity_mps2": float(settings.gravity_mps2),
                    "settle_seconds": float(settings.settle_seconds),
                    "convex_decomposition": convex_collision_audit,
                    "static_collision": static_collision_audit,
                    "wall_attachment": wall_attachment_audit,
                    "support_alignment_repairs": support_alignment_repairs,
                    "status": "failed",
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[visual_physics] support alignment audit failed: {exc}", flush=True)
        raise
    settle_displacement = []
    for path in loaded_dynamic_objects:
        before = object_pose_before_settle.get(path)
        after = world_translation(stage, path)
        if before is None or after is None:
            continue
        settle_displacement.append(
            {
                "name": path.rsplit("/", 1)[-1],
                "translation_before": [round(value, 6) for value in before],
                "translation_after": [round(value, 6) for value in after],
                "displacement_m": round(
                    math.sqrt(sum((after[index] - before[index]) ** 2 for index in range(3))),
                    6,
                ),
            }
        )
    audit_path = output_dir / "physics_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "physics_enabled": not settings.no_physics,
                "gravity_mps2": float(settings.gravity_mps2),
                "settle_seconds": float(settings.settle_seconds),
                "static_collision": static_collision_audit,
                "wall_attachment": wall_attachment_audit,
                "convex_decomposition": convex_collision_audit,
                "support_alignment": support_alignment_audit,
                "support_alignment_repairs": support_alignment_repairs,
                "support_patches": support_patches,
                "settle_displacement": settle_displacement,
                "status": "passed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[visual_physics] physics audit -> {audit_path}", flush=True)
    rendered = render_views(stage, render_scene_cfg, output_dir)
    print(
        "[visual_physics] loaded "
        f"arena={len(loaded_arena)}/{len(names(fixtures))} "
        f"objects={len(loaded_objects)}/{len(names(objects))} "
        f"robots={len(loaded_robots)}/{len(names(robots))} "
        f"from {scene_cfg.task_path}",
        flush=True,
    )
    if missing_fixtures:
        print(
            f"[visual_physics] missing fixtures: {', '.join(missing_fixtures)}",
            flush=True,
        )
    print(f"[visual_physics] wrote {len(rendered)} views under {output_dir}", flush=True)
    return 0


def run_render(settings: Any) -> int:
    """Launch a headless Isaac Sim app and render the configured task in-process."""
    global SETTINGS
    SETTINGS = _normalize(settings)
    if SETTINGS.settle_seconds <= 0.0:
        raise ValueError(
            "--settle-seconds must be greater than zero for physics rendering"
        )
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("ISAACSIM_ACCEPT_EULA", "yes")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    from isaacsim import SimulationApp  # must precede omni/pxr imports

    experience = os.environ.get("TASK_RENDER_EXPERIENCE", DEFAULT_EXPERIENCE)
    active_gpu = int(os.environ.get("INTERNDATA_ISAAC_ACTIVE_GPU", "0"))
    if active_gpu < 0:
        raise ValueError("INTERNDATA_ISAAC_ACTIVE_GPU must be non-negative")
    launch_config = {
        "headless": True,
        "active_gpu": active_gpu,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "width": int(SETTINGS.width),
        "height": int(SETTINGS.height),
        "renderer": SETTINGS.renderer,
        "fast_shutdown": True,
    }
    saved_argv = list(sys.argv)
    sys.argv = [saved_argv[0]]
    try:
        if experience and Path(experience).is_file():
            app = SimulationApp(launch_config, experience=experience)
        else:
            app = SimulationApp(launch_config)
    finally:
        sys.argv = saved_argv
    result = 0
    error = None
    try:
        result = _render_main(app, SETTINGS)
    except Exception as exc:
        traceback.print_exc()
        result = 1
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        output_dir = Path(SETTINGS.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "render_status.json").write_text(
            json.dumps(
                {"return_code": int(result), "error": error},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        app.close(wait_for_replicator=False)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(run_render(SimpleNamespace(task=sys.argv[1], output_dir=sys.argv[2])))
