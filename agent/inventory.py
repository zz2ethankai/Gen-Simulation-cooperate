"""Discover task, scene, asset and robot capabilities from SimBox YAMLs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from workflows.simbox.core.robots.profile import load_robot_profile_for_task

from .contracts import AssetCapability, RobotInstanceCapability, SceneCapabilityManifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_ROOTS = [
    REPO_ROOT / "InternDataAssets" / "Bench_2.1_isaacsim" / "scene_4",
]
DEFAULT_INDEX_PATH = REPO_ROOT / "output" / "agent_inventory.json"
INVENTORY_SCHEMA_VERSION = 3


class InventorySchemaError(ValueError):
    """The derived inventory cache does not match the current contract."""


def _source_identity(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def stable_task_id(path: Path) -> str:
    digest = hashlib.sha256(_source_identity(path).encode("utf-8")).hexdigest()
    return f"task_{digest}"


def _bool_value(*values: Any) -> bool | None:
    for value in values:
        if value is not None:
            return bool(value)
    return None


def _load_task(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = value.get("tasks") if isinstance(value, dict) else None
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        raise ValueError(f"task YAML must contain a non-empty tasks list: {path}")
    return tasks[0]


def _scene_id(path: Path, roots: Iterable[Path]) -> str:
    for root in roots:
        try:
            return path.resolve().relative_to(root.resolve()).parts[0]
        except (ValueError, IndexError):
            continue
    return path.parent.parent.parent.name


def _flatten_skills(task: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for robot_entry in task.get("skills", []) or []:
        if not isinstance(robot_entry, dict):
            continue
        for stages in robot_entry.values():
            for stage in stages or []:
                if not isinstance(stage, dict):
                    continue
                for skills in stage.values():
                    for skill in skills or []:
                        if isinstance(skill, dict) and skill.get("name"):
                            names.add(str(skill["name"]).lower())
    return sorted(names)


def _asset_capability(cfg: dict[str, Any], active: set[str]) -> AssetCapability:
    physics = cfg.get("physics") if isinstance(cfg.get("physics"), dict) else {}
    source_physics = cfg.get("source_physics") if isinstance(cfg.get("source_physics"), dict) else {}
    container = cfg.get("container_affordance") if isinstance(cfg.get("container_affordance"), dict) else {}
    attach_paths = cfg.get("attach_prim_path_children") or []
    if isinstance(attach_paths, str):
        attach_paths = [attach_paths]
    affordances: list[str] = []
    if container.get("can_receive_objects"):
        affordances.append("container")
    if container.get("open_top"):
        affordances.append("open_top")
    if cfg.get("name") in active:
        affordances.append("pickable")
    if cfg.get("role") in {"support", "target_support", "static_context_object"}:
        affordances.append("support")
    rigid = _bool_value(cfg.get("rigidbody"), physics.get("rigid_body"), source_physics.get("rigid_body"))
    collision = _bool_value(
        cfg.get("collision_enabled"),
        physics.get("collision_enabled"),
        source_physics.get("collision_enabled"),
    )
    if cfg.get("name") in active:
        attach_status = "explicit" if attach_paths else "runtime_discovery_required"
    else:
        attach_status = "not_required"
    return AssetCapability(
        name=str(cfg.get("name", "")),
        category=str(cfg.get("asset_category") or cfg.get("category") or "unknown"),
        role=str(cfg.get("role") or "unknown"),
        target_class=str(cfg.get("target_class") or "unknown"),
        asset_path=str(cfg.get("path")) if cfg.get("path") else None,
        parent_fixture=str(cfg.get("parent_fixture")) if cfg.get("parent_fixture") else None,
        rigid_body=rigid,
        collision_enabled=collision,
        attach_proxy_status=attach_status,
        attach_prim_path_children=[str(item) for item in attach_paths],
        affordances=sorted(set(affordances)),
    )


def _robot_instance_capability(
    cfg: dict[str, Any],
    task_path: Path,
) -> RobotInstanceCapability:
    instance_name = str(cfg.get("name", "")).strip()
    config_file = str(cfg.get("robot_config_file", "")).strip()
    if not instance_name:
        raise ValueError(f"robot instance requires a non-empty name: {task_path}")
    if not config_file:
        raise ValueError(
            f"robot instance {instance_name!r} requires robot_config_file: {task_path}"
        )
    profile = load_robot_profile_for_task(cfg, task_path)
    placement_family = str(
        getattr(profile.placement.family, "value", profile.placement.family)
    )
    return RobotInstanceCapability(
        instance_name=instance_name,
        profile_id=profile.profile_id,
        robot_config_file=config_file,
        target_class=profile.target_class,
        placement_family=placement_family,
        available_arms=sorted(profile.arms),
        capabilities=sorted(profile.capabilities),
        collision_world_modes=sorted(profile.collision_world_modes),
        profile_hash=profile.profile_hash,
    )


def discover_task_paths(
    scene_roots: Iterable[Path] | None = None,
    additional_paths: Iterable[Path] = (),
) -> list[Path]:
    roots = [Path(root) for root in (scene_roots or DEFAULT_SCENE_ROOTS)]
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        paths.update(path.resolve() for path in root.glob("*/assets/basic/*/simbox_task*.yaml"))
    paths.update(Path(path).resolve() for path in additional_paths if Path(path).is_file())
    return sorted(paths)


def build_inventory(
    scene_roots: Iterable[Path] | None = None,
    settings: dict[str, Any] | None = None,
) -> list[SceneCapabilityManifest]:
    roots = [Path(root) for root in (scene_roots or DEFAULT_SCENE_ROOTS)]
    converted_task_paths: list[Path] = []
    converted_scene_ids: dict[Path, str] = {}
    converted_reports: list[Any] = []
    if settings:
        from .tools.scene_ingest import convert_all

        scene_ingest = settings.get("scene_ingest", {})
        if isinstance(scene_ingest, dict) and bool(scene_ingest.get("enabled", True)):
            for report in convert_all(roots, settings=settings):
                if report.status == "failed":
                    print(
                        f"[scene_ingest] {report.task_id}: {report.failure_code} - {report.message}",
                        flush=True,
                    )
                elif report.out_task_path.is_file():
                    converted_path = report.out_task_path.resolve()
                    converted_task_paths.append(converted_path)
                    converted_scene_ids[converted_path] = report.scene_dir.name
                    converted_reports.append(report)
    manifests: list[SceneCapabilityManifest] = []
    task_paths = discover_task_paths(roots, converted_task_paths)
    task_paths = [
        path
        for path in task_paths
        if not any(
            path != report.out_task_path.resolve()
            and report.scene_dir.resolve() in path.parents
            and path.parent.name == report.task_id
            for report in converted_reports
        )
    ]
    for path in task_paths:
        task = _load_task(path)
        scene_id = converted_scene_ids.get(path.resolve(), _scene_id(path, roots))
        active = [str(item) for item in task.get("delivery_active_objects", []) or []]
        active_set = set(active)
        objects = [
            _asset_capability(item, active_set)
            for item in task.get("objects", []) or []
            if isinstance(item, dict) and item.get("name")
        ]
        object_by_name = {item.name: item for item in objects}
        active_caps = [object_by_name[name] for name in active if name in object_by_name]
        basic_ready = bool(active_caps) and all(
            item.rigid_body is True and item.collision_enabled is True for item in active_caps
        )
        robot_instances = [
            _robot_instance_capability(item, path)
            for item in task.get("robots", []) or []
            if isinstance(item, dict)
        ]
        if not robot_instances:
            raise ValueError(f"task requires at least one robot instance: {path}")
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        language = [
            str(item)
            for item in (
                task.get("name"),
                data.get("language_instruction"),
                data.get("detailed_language_instruction"),
                scene_id,
                *[item.category for item in objects],
                *active,
            )
            if item
        ]
        manifests.append(
            SceneCapabilityManifest(
                task_id=stable_task_id(path),
                scene_id=scene_id,
                source_task=str(path),
                task_class=str(task.get("task") or "unknown"),
                language=language,
                robot_instances=robot_instances,
                active_objects=active,
                objects=objects,
                container_regions=[
                    item for item in task.get("container_regions", []) or [] if isinstance(item, dict)
                ],
                existing_skills=_flatten_skills(task),
                physics_readiness="basic_ready" if basic_ready else "runtime_audit_required",
            )
        )
    return sorted(manifests, key=lambda item: (item.scene_id, item.task_id))


def write_inventory(
    manifests: list[SceneCapabilityManifest],
    output_path: Path = DEFAULT_INDEX_PATH,
) -> Path:
    payload = {
        "version": INVENTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(manifests),
        "tasks": [item.to_dict() for item in manifests],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def read_inventory(path: Path = DEFAULT_INDEX_PATH) -> list[SceneCapabilityManifest]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != INVENTORY_SCHEMA_VERSION:
        raise InventorySchemaError(
            f"inventory schema is not version {INVENTORY_SCHEMA_VERSION}: {path}"
        )
    return [SceneCapabilityManifest.from_dict(item) for item in payload.get("tasks", [])]


def load_or_build_inventory(
    index_path: Path = DEFAULT_INDEX_PATH,
    scene_roots: Iterable[Path] | None = None,
) -> list[SceneCapabilityManifest]:
    if index_path.is_file():
        try:
            return read_inventory(index_path)
        except InventorySchemaError:
            pass
    manifests = build_inventory(scene_roots)
    write_inventory(manifests, index_path)
    return manifests
