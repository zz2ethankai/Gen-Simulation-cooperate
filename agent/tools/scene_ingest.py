"""Normalize interdata scene configs into SimBox-runable task documents.

The agent scene pipeline emits an intermediate format (``interdata/task.yaml`` +
``interdata/arena.yaml``) that the data engine cannot consume directly. This module
translates that intermediate format into ``simbox_task.yaml`` + ``simbox_arena.yaml``
under an Agent-owned output root. Source scenes remain read-only; callers consume
the explicit paths returned by :class:`ConversionReport`.

The converter is intentionally pure: no Isaac/pxr imports at module load. The pxr
probes are lazy and return ``None`` on any failure so conversion always falls back
to structural defaults (the server, which has Isaac, populates the probed fields).
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..settings import SCENE_INGEST_DEFAULTS, merge_mappings


REPO_ROOT = Path(__file__).resolve().parents[2]

SUPPORT_FIXTURE_DEFAULT = "central_work_table"
DROPPED_REGION_SUFFIXES = ("book_candidate_region", "notebook_candidate_region")
ROBOT_START_REGION_NAME = "robot_start_region"
TASK_CLASS = "BananaBaseTask"


class SceneConversionError(Exception):
    """A scene failed conversion with a structured failure code."""

    def __init__(self, message: str, failure_code: str):
        super().__init__(message)
        self.message = message
        self.failure_code = failure_code


@dataclass
class ConversionReport:
    scene_dir: Path
    task_id: str
    out_task_path: Path
    out_arena_path: Path
    status: str = "failed"  # converted | skipped | failed
    failure_code: str | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_dir": str(self.scene_dir),
            "task_id": self.task_id,
            "out_task_path": str(self.out_task_path),
            "out_arena_path": str(self.out_arena_path),
            "status": self.status,
            "failure_code": self.failure_code,
            "message": self.message,
            "warnings": self.warnings,
        }


def _scene_ingest_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    section = dict(settings or {}).get("scene_ingest", {})
    if not isinstance(section, Mapping):
        raise ValueError("Agent config scene_ingest must be a mapping")
    return merge_mappings(SCENE_INGEST_DEFAULTS, section)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _output_root(cfg: Mapping[str, Any]) -> Path:
    value = str(cfg.get("output_root") or "").strip()
    if not value:
        raise ValueError("Agent config scene_ingest.output_root must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _owned_scene_dir(scene_dir: Path, cfg: Mapping[str, Any]) -> Path:
    source = scene_dir.resolve()
    output_root = _output_root(cfg)
    try:
        relative = source.relative_to(REPO_ROOT.resolve())
    except ValueError:
        source_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
        relative = Path(f"{source.name}-{source_key}")
    owned = (output_root / relative).resolve()
    if owned == source or source in owned.parents:
        raise ValueError(
            "scene_ingest.output_root must not place derived files inside the source scene"
        )
    if output_root != owned and output_root not in owned.parents:
        raise ValueError("derived scene path escapes scene_ingest.output_root")
    return owned


# --------------------------------------------------------------------------- #
# Discovery / idempotency
# --------------------------------------------------------------------------- #


def discover_interdata_scenes(roots: Iterable[Path]) -> list[Path]:
    """Return scene directories that contain an ``interdata/task.yaml``."""
    scenes: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for task_path in root.rglob("interdata/task.yaml"):
            scenes.add(task_path.resolve().parent.parent)
    return sorted(scenes)


def _source_mtime(scene_dir: Path) -> float:
    interdata = scene_dir / "interdata"
    candidates = [interdata / "task.yaml", interdata / "arena.yaml"]
    return max((path.stat().st_mtime for path in candidates if path.is_file()), default=0.0)


def needs_conversion(
    scene_dir: Path,
    out_task_path: Path,
    out_arena_path: Path,
) -> bool:
    """True if either output is missing or older than a source document."""
    if not out_task_path.is_file() or not out_arena_path.is_file():
        return True
    source_mtime = _source_mtime(scene_dir)
    return source_mtime > min(out_task_path.stat().st_mtime, out_arena_path.stat().st_mtime)


# --------------------------------------------------------------------------- #
# Top-level conversion
# --------------------------------------------------------------------------- #


def convert_scene(scene_dir: Path, settings: Mapping[str, Any] | None = None) -> ConversionReport:
    scene_dir = Path(scene_dir).resolve()
    cfg = _scene_ingest_settings(settings)
    interdata_dir = scene_dir / "interdata"
    task_path = interdata_dir / "task.yaml"
    arena_path = interdata_dir / "arena.yaml"
    task_id = _task_id(scene_dir, cfg)
    out_dir_rel = Path(str(cfg["out_dir_rel"]))
    if out_dir_rel.is_absolute() or ".." in out_dir_rel.parts:
        raise ValueError("scene_ingest.out_dir_rel must be a relative path without '..'")
    out_dir = _owned_scene_dir(scene_dir, cfg) / out_dir_rel / task_id
    out_task_path = out_dir / "simbox_task.yaml"
    out_arena_path = out_dir / "simbox_arena.yaml"

    report = ConversionReport(
        scene_dir=scene_dir,
        task_id=task_id,
        out_task_path=out_task_path,
        out_arena_path=out_arena_path,
    )

    if not task_path.is_file() or not arena_path.is_file():
        missing = [name for name, path in (("task.yaml", task_path), ("arena.yaml", arena_path)) if not path.is_file()]
        report.failure_code = "SOURCE_MISSING"
        report.message = f"interdata source missing: {', '.join(missing)} in {interdata_dir}"
        return report

    if not needs_conversion(scene_dir, out_task_path, out_arena_path):
        report.status = "skipped"
        report.message = "conversion outputs already up to date"
        return report

    try:
        interdata = _load_mapping(task_path, "SOURCE_INVALID_TASK")
        arena = _load_mapping(arena_path, "SOURCE_INVALID_ARENA")
        if not interdata.get("objects"):
            raise SceneConversionError("interdata task.yaml has no objects", "SOURCE_NO_OBJECTS")

        task_doc, warnings = build_task_document(interdata, arena, scene_dir, cfg, out_task_path, out_arena_path)
        arena_doc = build_arena_document(arena, cfg)
        _fix_arena_texture_libs(arena_doc, scene_dir, warnings)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_task_path.write_text(
            yaml.safe_dump({"tasks": [task_doc]}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        out_arena_path.write_text(
            yaml.safe_dump(arena_doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        report.status = "converted"
        report.message = "converted interdata scene to SimBox task"
        report.warnings = warnings
    except SceneConversionError as exc:
        report.failure_code = exc.failure_code
        report.message = exc.message
    except Exception as exc:  # pragma: no cover - defensive
        report.failure_code = "CONVERSION_ERROR"
        report.message = f"{type(exc).__name__}: {exc}"
    return report


def convert_all(roots: Iterable[Path], settings: Mapping[str, Any] | None = None) -> list[ConversionReport]:
    scenes = discover_interdata_scenes(roots)
    return [convert_scene(scene, settings) for scene in scenes]


def _load_mapping(path: Path, failure_code: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SceneConversionError(f"cannot parse {path}: {exc}", failure_code) from exc
    if not isinstance(value, dict):
        raise SceneConversionError(f"expected a mapping at {path}", failure_code)
    return value


def _task_id(scene_dir: Path, cfg: Mapping[str, Any]) -> str:
    if str(cfg.get("task_id_source", "scene_name")) == "scene_name":
        return scene_dir.name
    return scene_dir.name


# --------------------------------------------------------------------------- #
# Document builders
# --------------------------------------------------------------------------- #


def build_task_document(
    interdata: Mapping[str, Any],
    arena: Mapping[str, Any],
    scene_dir: Path,
    cfg: Mapping[str, Any],
    out_task_path: Path,
    out_arena_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    task_id = _task_id(scene_dir, cfg)
    support_fixture = str(cfg.get("support_fixture") or SUPPORT_FIXTURE_DEFAULT)

    table_center = _resolve_table_center(arena, scene_dir, cfg, warnings)
    table_trans = _table_translation(arena, support_fixture)

    object_docs, active_objects = _build_objects(interdata, scene_dir, cfg, warnings)
    robot_name = _robot_name(interdata)
    robot_doc, robot_profile = _build_robot(interdata)
    cameras = _build_cameras(interdata, robot_profile)

    default_robot_support = (
        "floor"
        if robot_profile.placement.family.value == "floor_standing"
        else support_fixture
    )
    regions, robot_start = _build_regions(
        interdata,
        arena,
        table_center,
        table_trans,
        support_fixture,
        robot_name,
        default_robot_support,
        warnings,
    )
    source_regions = _build_source_regions(robot_start, robot_name, default_robot_support)

    env_map = _build_env_map(scene_dir, cfg, warnings)
    data = _build_data(interdata, cfg, task_id)

    doc: dict[str, Any] = {
        "name": task_id,
        "asset_root": _repo_relative(scene_dir),
        "task": TASK_CLASS,
        "task_id": 0,
        "offset": None,
        "render": True,
        "arena_file": _repo_relative(out_arena_path),
        "env_map": env_map,
        "robots": [robot_doc],
        "objects": object_docs,
        "cameras": cameras,
        "regions": regions,
        "source_regions": source_regions,
        "delivery_active_objects": active_objects,
        "skills": [{robot_name: [{"base": [], "left": [], "right": []}]}],
        "data": data,
        "max_episode_length": int(cfg.get("max_episode_length", 10000)),
        "source_tasks": [],
        "metadata": {
            "scene_name": scene_dir.name,
            "source_format": "interdata",
            "source_task": str((scene_dir / "interdata" / "task.yaml").resolve()),
            "converted_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return doc, warnings


_TEXTURE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _resolve_texture_lib(scene_dir: Path, texture: Mapping[str, Any]) -> str | None:
    """Return a texture_lib (relative to scene_dir/asset_root) that globs to a
    directory of texture images, or None if none exists under the scene.

    interdata arenas carry ``texture_lib: floor_textures`` as a bare name while
    the images live under ``interdata/texture_libs/floor_textures/``. The engine's
    ``PlaneObject.apply_texture`` globs ``asset_root/<texture_lib>/*``, so the value
    must resolve to a real directory relative to the asset root.
    """
    lib = texture.get("texture_lib")
    if not lib or not isinstance(lib, str):
        return None
    for candidate in sorted(scene_dir.glob(f"**/{lib}")):
        if not candidate.is_dir():
            continue
        if any(path.suffix.lower() in _TEXTURE_IMAGE_EXTS for path in candidate.iterdir()):
            return candidate.relative_to(scene_dir).as_posix()
    return None


def _fix_arena_texture_libs(arena_doc: dict[str, Any], scene_dir: Path, warnings: list[str]) -> None:
    for fixture in arena_doc.get("fixtures", []) or []:
        texture = fixture.get("texture")
        if not isinstance(texture, dict):
            continue
        lib = texture.get("texture_lib")
        resolved = _resolve_texture_lib(scene_dir, texture)
        if resolved:
            texture["texture_lib"] = resolved
            texture_file = texture.get("texture_file")
            if isinstance(texture_file, str) and texture_file:
                texture["texture_file"] = f"{resolved}/{Path(texture_file).name}"
        elif lib:
            warnings.append(
                f"TEXTURE_LIB_UNRESOLVED: {fixture.get('name')} texture_lib={lib} "
                "(no image dir found under asset_root; engine texture load will fail)"
            )


def build_arena_document(arena: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = []
    for fixture in arena.get("fixtures", []) or []:
        if not isinstance(fixture, dict) or not fixture.get("name"):
            continue
        converted = {}
        for key, value in fixture.items():
            if key == "usd_path":
                converted["path"] = copy.deepcopy(value)
            else:
                converted[key] = copy.deepcopy(value)
        fixtures.append(converted)
    return {"fixtures": fixtures}


def _build_objects(
    interdata: Mapping[str, Any],
    scene_dir: Path,
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    docs: list[dict[str, Any]] = []
    active: list[str] = []
    for item in interdata.get("objects", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        usd_rel = item.get("usd_path")
        usd_path = scene_dir / str(usd_rel) if usd_rel else None
        if usd_path is None or not usd_path.is_file():
            warnings.append(f"OBJECT_PATH_MISSING: {name} usd_path={usd_rel}")
            continue

        prim_path_child = _derive_prim_path_child(item)
        attach = probe_attach_prim_path_children(usd_path, prim_path_child)
        if not attach:
            warnings.append(
                f"ATTACH_PROBE_FAILED: {name} (engine will auto-discover collision prims)"
            )

        doc: dict[str, Any] = {
            "name": name,
            "target_class": str(item.get("target_class") or cfg.get("target_class_task_object") or "RigidObject"),
            "path": str(usd_rel),
            "prim_path_child": prim_path_child,
            "attach_prim_path_children": attach,
            "role": _task_role(item),
        }
        for key in (
            "euler",
            "scale",
            "parent_fixture",
            "asset_category",
            "description",
            "collider",
            "rigidbody",
            "friction",
            "grasp_annotation_path",
            "grasp_annotation_format",
            "grasp_annotation_count",
            "grasp_annotation_frame",
            "grasp_annotation_scale",
            "spawn_region",
            "placement",
        ):
            if key in item:
                doc[key] = copy.deepcopy(item[key])
        initial_pose = item.get("initial_pose")
        if isinstance(initial_pose, Mapping):
            rotation = initial_pose.get("rotation")
            if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
                base_euler = doc.get("euler") or [0.0, 0.0, 0.0]
                doc["euler"] = [
                    float(base_euler[0]) + float(rotation[0]),
                    float(base_euler[1]) + float(rotation[1]),
                    float(base_euler[2]) + float(rotation[2]),
                ]
        if "mass_kg" in item:
            doc["mass"] = copy.deepcopy(item["mass_kg"])
        if isinstance(item.get("source_physics"), dict):
            doc["source_physics"] = copy.deepcopy(item["source_physics"])
        doc["physics"] = {
            "rigid_body": bool(item.get("rigidbody", True)),
            "collision_enabled": bool(item.get("collision_enabled", True)),
        }
        docs.append(doc)
        active.append(name)
    return docs, active


def _derive_prim_path_child(item: Mapping[str, Any]) -> str:
    frame = str(item.get("grasp_annotation_frame") or "")
    if frame.startswith("/World/"):
        segment = frame[len("/World/"):].strip("/")
        if segment:
            return segment.split("/")[0]
    return "Aligned"


def _task_role(item: Mapping[str, Any]) -> str:
    role = str(item.get("role") or "")
    if role == "task_active":
        return "task_object"
    return role or "task_object"


def _robot_name(interdata: Mapping[str, Any]) -> str:
    robot = interdata.get("robot")
    if isinstance(robot, dict) and robot.get("name"):
        return str(robot["name"])
    raise SceneConversionError("interdata robot.name is required", "SOURCE_ROBOT_INSTANCE_MISSING")


def _build_robot(interdata: Mapping[str, Any]):
    from workflows.simbox.core.robots.profile import load_robot_profile_for_task

    robot = interdata.get("robot") or {}
    if not isinstance(robot, Mapping):
        raise SceneConversionError("interdata robot must be a mapping", "SOURCE_INVALID_ROBOT")
    config_file = robot.get("robot_config_file")
    if not config_file:
        raise SceneConversionError(
            "interdata robot.robot_config_file is required",
            "SOURCE_ROBOT_PROFILE_MISSING",
        )
    try:
        profile = load_robot_profile_for_task(robot, REPO_ROOT / "agent" / "tools" / "scene_ingest.py")
    except (OSError, ValueError) as exc:
        raise SceneConversionError(str(exc), "SOURCE_ROBOT_PROFILE_INVALID") from exc
    doc: dict[str, Any] = {
        "name": _robot_name(interdata),
        "robot_config_file": str(config_file),
        "use_batch": bool(robot.get("use_batch", True)),
        "collision_activation_distance": float(robot.get("collision_activation_distance", 0.05)),
        "ignore_substring": list(robot.get("ignore_substring") or []),
    }
    if isinstance(robot.get("euler"), list) and len(robot["euler"]) >= 3:
        doc["euler"] = [float(value) for value in robot["euler"][:3]]
    return doc, profile


def _build_cameras(
    interdata: Mapping[str, Any],
    robot_profile,
) -> list[dict[str, Any]]:
    robot_name = _robot_name(interdata)
    return [camera.to_task_camera(robot_name) for camera in robot_profile.camera_rig]


def _jitter(region: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    random_config = region.get("random_config")
    if isinstance(random_config, Mapping):
        pos_range = random_config.get("pos_range")
        if (
            isinstance(pos_range, (list, tuple))
            and len(pos_range) == 2
            and isinstance(pos_range[0], (list, tuple))
            and len(pos_range[0]) >= 2
            and isinstance(pos_range[1], (list, tuple))
            and len(pos_range[1]) >= 2
        ):
            return [float(pos_range[0][0]), float(pos_range[0][1])], [
                float(pos_range[1][0]),
                float(pos_range[1][1]),
            ]
    return [-0.01, -0.01], [0.01, 0.01]


def _region_world_xy(
    region: Mapping[str, Any],
    table_trans: list[float],
) -> list[float] | None:
    """Resolve an object region's desired world XY.

    ``runtime_placement.offset_xy`` (``frame: parent_world_xy_offset``) is the
    authoritative position: the offset from the support fixture's world pivot.
    Falls back to the region's own ``center`` when ``offset_xy`` is absent.
    """
    placement = region.get("runtime_placement")
    if isinstance(placement, Mapping):
        if placement.get("frame") == "parent_world_xy_offset":
            offset_xy = placement.get("offset_xy")
            if isinstance(offset_xy, (list, tuple)) and len(offset_xy) >= 2:
                return [
                    table_trans[0] + float(offset_xy[0]),
                    table_trans[1] + float(offset_xy[1]),
                ]
    center = region.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return [float(center[0]), float(center[1])]
    return None


def _build_regions(
    interdata: Mapping[str, Any],
    arena: Mapping[str, Any],
    table_center: list[float],
    table_trans: list[float],
    support_fixture: str,
    robot_name: str,
    default_robot_support: str,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    arena_center: dict[str, list[float]] = {}
    for region in arena.get("regions", []) or []:
        if isinstance(region, dict) and region.get("name") and isinstance(region.get("center"), (list, tuple)):
            arena_center[str(region["name"])] = [float(value) for value in region["center"][:2]]

    regions: list[dict[str, Any]] = []
    robot_start: Mapping[str, Any] | None = None
    for region in interdata.get("regions", []) or []:
        if not isinstance(region, dict) or not region.get("name"):
            continue
        name = str(region["name"])
        if name in DROPPED_REGION_SUFFIXES:
            warnings.append(f"DROPPED_REGION: {name} references an object absent from task.yaml")
            continue
        if name == ROBOT_START_REGION_NAME or str(region.get("object")) == robot_name:
            robot_start = region
            continue

        desired = _region_world_xy(region, table_trans)
        if desired is None:
            arena_xy = arena_center.get(name)
            if arena_xy is not None:
                desired = [float(arena_xy[0]), float(arena_xy[1])]
        if desired is None:
            warnings.append(f"REGION_NO_CENTER: {name}")
            continue
        base = [desired[0] - table_center[0], desired[1] - table_center[1]]
        jmin, jmax = _jitter(region)
        pos_range = [
            [round(base[0] + jmin[0], 6), round(base[1] + jmin[1], 6), 0.0],
            [round(base[0] + jmax[0], 6), round(base[1] + jmax[1], 6), 0.0],
        ]
        yaw = region.get("yaw_range") or [0.0, 0.0]
        runtime_support = str(
            region.get("target")
            or region.get("B")
            or region.get("parent_fixture")
            or support_fixture
        )
        semantic_support = str(
            region.get("parent_fixture")
            or region.get("support_target_fixture")
            or support_fixture
        )
        converted = {
            "name": name,
            "object": str(region.get("object") or region.get("A") or ""),
            "A": str(region.get("A") or region.get("object") or ""),
            "target": runtime_support,
            "random_type": str(
                region.get("random_type") or region.get("type") or "A_on_B_region_sampler"
            ),
            "random_config": {
                "pos_range": pos_range,
                "yaw_rotation": [float(yaw[0]), float(yaw[1])],
            },
            "B": runtime_support,
            "parent_fixture": semantic_support,
            "support_surface_z": region.get("support_surface_z"),
        }
        for key in ("center", "size", "runtime_placement", "sampling", "support_surface_source"):
            if key in region:
                converted[key] = copy.deepcopy(region[key])
        source_random = region.get("random_config")
        if isinstance(source_random, Mapping) and source_random.get("support_surface_z") is not None:
            converted["random_config"]["support_surface_z"] = copy.deepcopy(
                source_random["support_surface_z"]
            )
        regions.append(converted)

    robot_center = list(robot_start.get("center") or [0.0, 0.0]) if robot_start else [0.0, 0.0]
    robot_yaw = list(robot_start.get("yaw_range") or [0.0, 0.0]) if robot_start else [0.0, 0.0]
    robot_runtime_support = str(
        (robot_start or {}).get("target")
        or (robot_start or {}).get("B")
        or (robot_start or {}).get("parent_fixture")
        or default_robot_support
    )
    robot_semantic_support = str(
        (robot_start or {}).get("parent_fixture")
        or (robot_start or {}).get("support_target_fixture")
        or default_robot_support
    )
    robot_random = (robot_start or {}).get("random_config")
    robot_pos_range = robot_random.get("pos_range") if isinstance(robot_random, Mapping) else None
    if not (
        isinstance(robot_pos_range, list)
        and len(robot_pos_range) == 2
        and all(isinstance(value, list) and len(value) >= 2 for value in robot_pos_range)
    ):
        pos = [float(robot_center[0]), float(robot_center[1]), 0.0]
        robot_pos_range = [pos, pos]
    else:
        robot_pos_range = [
            [float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 0.0]
            for value in robot_pos_range
        ]
    converted_robot_region = {
        "name": str((robot_start or {}).get("name") or ROBOT_START_REGION_NAME),
        "object": robot_name,
        "A": robot_name,
        "target": robot_runtime_support,
        "random_type": str(
            (robot_start or {}).get("random_type")
            or (robot_start or {}).get("type")
            or "A_on_B_region_sampler"
        ),
        "random_config": {
            "pos_range": robot_pos_range,
            "yaw_rotation": [float(robot_yaw[0]), float(robot_yaw[1])],
        },
        "B": robot_runtime_support,
        "parent_fixture": robot_semantic_support,
        "center": [float(robot_center[0]), float(robot_center[1])],
        "support_surface_z": robot_start.get("support_surface_z") if robot_start else 0.0,
    }
    for key in ("size", "runtime_placement", "sampling"):
        if robot_start and key in robot_start:
            converted_robot_region[key] = copy.deepcopy(robot_start[key])
    regions.append(converted_robot_region)
    return regions, robot_start


def _build_source_regions(
    robot_start: Mapping[str, Any] | None,
    robot_name: str,
    default_robot_support: str,
) -> list[dict[str, Any]]:
    center = list(robot_start.get("center") or [0.0, 0.0]) if robot_start else [0.0, 0.0]
    yaw = list(robot_start.get("yaw_range") or [0.0, 0.0]) if robot_start else [0.0, 0.0]
    support = str(
        (robot_start or {}).get("target")
        or (robot_start or {}).get("B")
        or (robot_start or {}).get("parent_fixture")
        or default_robot_support
    )
    cx = float(center[0])
    cy = float(center[1])
    return [
        {
            "name": "robot_initial_region",
            "type": "A_on_B_region_sampler",
            "A": "robot",
            "B": support,
            "center": [cx, cy],
            "center_xyz": [cx, 0.0, cy],
            "yaw_range": [float(yaw[0]), float(yaw[1])],
            "robot_base": robot_name,
            "support_surface_z": (robot_start or {}).get("support_surface_z", 0.0),
        }
    ]


def _build_env_map(
    scene_dir: Path,
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    envmap_lib = str(cfg.get("envmap_lib") or "environment/envmaps")
    hdr_files = list((scene_dir / envmap_lib).glob("*.hdr")) if (scene_dir / envmap_lib).is_dir() else []
    if not hdr_files:
        warnings.append(f"ENVMAP_MISSING: no *.hdr under {scene_dir / envmap_lib}; engine will fall back to default")
    return {
        "envmap_lib": envmap_lib,
        "apply_randomization": False,
    }


def _build_data(interdata: Mapping[str, Any], cfg: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    tasks = interdata.get("tasks") or []
    language = ""
    detailed = ""
    if tasks and isinstance(tasks[0], dict):
        first = tasks[0]
        name = str(first.get("task_name") or "")
        description = str(first.get("task_description") or "")
        language = description or name
        detailed = f"{name}: {description}".strip(": ") if name and description else (name or description)
    return {
        "language_instruction": language,
        "detailed_language_instruction": detailed,
        "task_dir": f"runs/{task_id}",
        "collect_info": task_id,
        "version": "v1.0",
        "update": True,
        "max_episode_length": int(cfg.get("max_episode_length", 10000)),
    }


def _table_fixture(arena: Mapping[str, Any], support_fixture: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in arena.get("fixtures", []) or []
            if isinstance(item, dict) and item.get("name") == support_fixture
        ),
        None,
    )


def _table_translation(arena: Mapping[str, Any], support_fixture: str) -> list[float]:
    """World XY of the support fixture pivot, the anchor for ``offset_xy``."""
    fixture = _table_fixture(arena, support_fixture)
    translation = (fixture or {}).get("translation") or [0.0, 0.0, 0.0]
    return [float(translation[0]), float(translation[1])]


def _resolve_table_center(
    arena: Mapping[str, Any],
    scene_dir: Path,
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> list[float]:
    support_fixture = str(cfg.get("support_fixture") or SUPPORT_FIXTURE_DEFAULT)
    fixture = _table_fixture(arena, support_fixture)
    if fixture is None:
        warnings.append(f"SUPPORT_FIXTURE_MISSING: {support_fixture} not in arena; using origin")
        return [0.0, 0.0]
    usd_rel = fixture.get("usd_path") or fixture.get("path")
    usd_path = scene_dir / str(usd_rel) if usd_rel else None
    translation = fixture.get("translation") or [0.0, 0.0, 0.0]
    euler = fixture.get("euler") or [0.0, 0.0, 0.0]
    scale = fixture.get("scale") or [1.0, 1.0, 1.0]
    if usd_path is not None and usd_path.is_file():
        center = probe_table_bbox_center(usd_path, translation, euler, scale)
        if center is not None:
            return [center[0], center[1]]
        warnings.append(f"TABLE_BBOX_PROBE_FAILED: {support_fixture}; using fixture translation")
    else:
        warnings.append(f"TABLE_USD_MISSING: {usd_rel}; using fixture translation")
    return [float(translation[0]), float(translation[1])]


# --------------------------------------------------------------------------- #
# pxr probes (lazy; return None on any failure so conversion always falls back)
# --------------------------------------------------------------------------- #


def probe_attach_prim_path_children(obj_usd: Path, prim_path_child: str) -> list[str]:
    """Return collision-prim paths (relative to the object root) under ``prim_path_child``.

    Mirrors ``attach_collision_utils.collision_candidate_paths`` so the engine's
    explicit attach list matches what CuRobo will see. Empty on probe failure.
    """
    try:
        from pxr import Usd, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return []
    try:
        stage = Usd.Stage.Open(str(obj_usd))
        if stage is None:
            return []

        def _find_child_prim(prim: Any) -> Any | None:
            if prim.GetName() == prim_path_child:
                return prim
            for child in prim.GetChildren():
                found = _find_child_prim(child)
                if found is not None:
                    return found
            return None

        target = _find_child_prim(stage.GetPseudoRoot())
        if target is None:
            return []
        results: list[str] = []

        def _collect(prim: Any, prefix: str) -> None:
            collision = UsdPhysics.CollisionAPI(prim)
            if collision and bool(collision.GetCollisionEnabledAttr().Get()):
                results.append(prefix)
            for child in prim.GetChildren():
                _collect(child, f"{prefix}/{child.GetName()}")

        _collect(target, prim_path_child)
        return sorted(set(results))
    except Exception:
        return []


def probe_table_bbox_center(
    fixture_usd: Path,
    translation: Any,
    euler: Any,
    scale: Any,
) -> tuple[float, float, float] | None:
    """Return the fixture's world bbox center after the arena transform is applied."""
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415
    except ImportError:
        return None
    try:
        stage = Usd.Stage.Open(str(fixture_usd))
        if stage is None:
            return None
        root_children = stage.GetPseudoRoot().GetChildren()
        if not root_children:
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode(0), [UsdGeom.Tokens.default_])
        bound = cache.ComputeLocalBound(root_children[0])
        rng = bound.ComputeAlignedRange()
        center_local = [
            (rng.GetMin()[i] + rng.GetMax()[i]) / 2.0 for i in range(3)
        ]
        sx, sy, sz = (float(v) for v in (scale or [1.0, 1.0, 1.0]))
        cx = center_local[0] * sx
        cy = center_local[1] * sy
        cz = center_local[2] * sz
        rad = math.radians(float((euler or [0.0, 0.0, 0.0])[2]))
        rx = cx * math.cos(rad) - cy * math.sin(rad)
        ry = cx * math.sin(rad) + cy * math.cos(rad)
        t = translation or [0.0, 0.0, 0.0]
        return (float(t[0]) + rx, float(t[1]) + ry, float(t[2]) + cz)
    except Exception:
        return None
