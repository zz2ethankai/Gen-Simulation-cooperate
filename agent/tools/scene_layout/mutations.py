"""Compile typed scene mutations into immutable derived SimBox documents."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import yaml

from .models import SceneSpec, SceneSpecError, SupportGraph


class SceneMutationError(ValueError):
    """A typed mutation cannot be applied without an ambiguous scene edit."""


@dataclass(frozen=True)
class MoveEntityOnSupport:
    entity: str
    support: str
    delta_xy_m: tuple[float, float]
    kind: str = "move_entity_on_support"


@dataclass(frozen=True)
class RotateEntityOnSupport:
    entity: str
    yaw_offset_deg: float
    kind: str = "rotate_entity_on_support"


@dataclass(frozen=True)
class SetSupportHeight:
    support: str
    world_z_m: float
    kind: str = "set_support_height"


@dataclass(frozen=True)
class SetRobotPlacement:
    instance_name: str
    support: str
    delta_xy_m: tuple[float, float]
    yaw_offset_deg: float
    kind: str = "set_robot_placement"


SceneMutation: TypeAlias = (
    MoveEntityOnSupport | RotateEntityOnSupport | SetSupportHeight | SetRobotPlacement
)


@dataclass(frozen=True)
class DerivedScene:
    task_path: Path
    arena_path: Path
    mutation_path: Path
    scene_revision: str
    scene_spec: SceneSpec


class SceneLayoutCompiler:
    """Apply an explicit mutation vocabulary without exposing YAML paths."""

    def compile(
        self,
        source_task_path: Path,
        source_arena_path: Path,
        mutations: list[SceneMutation],
        output_dir: Path,
    ) -> DerivedScene:
        source_task_path = source_task_path.resolve()
        source_arena_path = source_arena_path.resolve()
        output_dir = output_dir.resolve()
        task_path = output_dir / "simbox_task.yaml"
        arena_path = output_dir / "simbox_arena.yaml"
        mutation_path = output_dir / "scene_mutations.json"
        if task_path in {source_task_path, source_arena_path} or arena_path in {
            source_task_path,
            source_arena_path,
        }:
            raise SceneMutationError("derived scene paths must differ from both source documents")
        if output_dir.exists():
            raise SceneMutationError(f"derived scene revision already exists: {output_dir}")

        source_task = _load_yaml(source_task_path)
        source_arena = _load_yaml(source_arena_path)
        source_spec = SceneSpec.from_documents(
            source_task,
            source_arena,
            source_task=source_task_path,
            source_arena=source_arena_path,
        )
        task_document = copy.deepcopy(source_task)
        arena_document = copy.deepcopy(source_arena)
        task = _task(task_document)
        for mutation in mutations:
            self._apply(task, arena_document, mutation)
        SupportGraph.from_documents(task_document, arena_document)

        serialized = [_mutation_dict(mutation) for mutation in mutations]
        revision_payload = _revision_payload(source_spec, mutations)
        scene_revision = scene_revision_for(source_spec, mutations)
        task.setdefault("metadata", {})["agent_scene_layout"] = {
            "scene_revision": scene_revision,
            "source_task": str(source_task_path),
            "source_arena": str(source_arena_path),
            "source_task_hash": source_spec.source_task_hash,
            "source_arena_hash": source_spec.source_arena_hash,
            "mutations": serialized,
        }
        task["arena_file"] = str(arena_path)
        _write_revision(
            output_dir,
            task_document,
            arena_document,
            {**revision_payload, "scene_revision": scene_revision},
        )
        derived_spec = SceneSpec.from_documents(
            task_document,
            arena_document,
            source_task=task_path,
            source_arena=arena_path,
        )
        return DerivedScene(task_path, arena_path, mutation_path, scene_revision, derived_spec)

    def _apply(
        self,
        task: dict[str, Any],
        arena: dict[str, Any],
        mutation: SceneMutation,
    ) -> None:
        if isinstance(mutation, MoveEntityOnSupport):
            region = _region(task, mutation.entity)
            container_updates = _translated_container_regions(
                task,
                mutation.entity,
                mutation.delta_xy_m,
            )
            _shift_fixed_region(
                region,
                support=mutation.support,
                delta_xy_m=mutation.delta_xy_m,
            )
            _apply_container_region_updates(container_updates)
            return
        if isinstance(mutation, RotateEntityOnSupport):
            region = _region(task, mutation.entity)
            yaw = float(mutation.yaw_offset_deg)
            container_updates = _rotated_container_regions(
                task,
                mutation.entity,
                region,
                yaw,
            )
            random_config = _random_config(region)
            random_config["yaw_rotation"] = [yaw, yaw]
            region["yaw_range"] = [yaw, yaw]
            _apply_container_region_updates(container_updates)
            return
        if isinstance(mutation, SetSupportHeight):
            _set_support_height(task, arena, mutation.support, float(mutation.world_z_m))
            return
        if isinstance(mutation, SetRobotPlacement):
            region = _region(task, mutation.instance_name)
            _shift_fixed_region(
                region,
                support=mutation.support,
                delta_xy_m=mutation.delta_xy_m,
            )
            yaw = float(mutation.yaw_offset_deg)
            _random_config(region)["yaw_rotation"] = [yaw, yaw]
            region["yaw_range"] = [yaw, yaw]
            for source_region in task.get("source_regions") or []:
                if not isinstance(source_region, dict):
                    continue
                if source_region.get("robot_base") != mutation.instance_name:
                    continue
                dx, dy = (float(value) for value in mutation.delta_xy_m)
                center = source_region.get("center")
                if not isinstance(center, list) or len(center) < 2:
                    raise SceneMutationError(
                        f"source region for {mutation.instance_name!r} has no center"
                    )
                x, y = float(center[0]) + dx, float(center[1]) + dy
                source_region.update(
                    {
                        "B": mutation.support,
                        "center": [x, y],
                        "center_xyz": [x, 0.0, y],
                        "yaw_range": [yaw, yaw],
                    }
                )
            return
        raise SceneMutationError(f"unsupported scene mutation: {type(mutation).__name__}")


def _mutation_dict(mutation: SceneMutation) -> dict[str, Any]:
    value = asdict(mutation)
    value["kind"] = mutation.kind
    return value


def _revision_payload(
    scene_spec: SceneSpec,
    mutations: list[SceneMutation] | tuple[SceneMutation, ...],
) -> dict[str, Any]:
    return {
        "source_task_hash": scene_spec.source_task_hash,
        "source_arena_hash": scene_spec.source_arena_hash,
        "mutations": [_mutation_dict(mutation) for mutation in mutations],
    }


def scene_revision_for(
    scene_spec: SceneSpec,
    mutations: list[SceneMutation] | tuple[SceneMutation, ...],
) -> str:
    payload = _revision_payload(scene_spec, mutations)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SceneMutationError(f"cannot load source YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SceneMutationError(f"source YAML must be a mapping: {path}")
    return value


def _task(document: Mapping[str, Any]) -> dict[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise SceneMutationError("SimBox source must contain exactly one tasks[0] mapping")
    return tasks[0]


def _regions(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = task.get("regions")
    if not isinstance(values, list):
        raise SceneMutationError("tasks[0].regions must be a list")
    if not all(isinstance(value, dict) for value in values):
        raise SceneMutationError("every tasks[0].regions entry must be a mapping")
    return values


def _region(task: Mapping[str, Any], entity: str) -> dict[str, Any]:
    matches = [
        region
        for region in _regions(task)
        if str(region.get("object") or region.get("A") or "") == entity
    ]
    if len(matches) != 1:
        raise SceneMutationError(f"entity {entity!r} must own exactly one task region, found {len(matches)}")
    return matches[0]


def _random_config(region: dict[str, Any]) -> dict[str, Any]:
    value = region.setdefault("random_config", {})
    if not isinstance(value, dict):
        raise SceneMutationError(f"region for {region.get('object')} has invalid random_config")
    return value


def _container_regions_for(
    task: Mapping[str, Any],
    entity: str,
) -> tuple[dict[str, Any], ...]:
    values = task.get("container_regions")
    if values is None:
        return ()
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise SceneMutationError("tasks[0].container_regions must be a list of mappings")
    return tuple(
        value
        for value in values
        if str(value.get("object") or value.get("target") or "") == entity
    )


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SceneMutationError(f"{label} must contain exactly two numbers")
    try:
        pair = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise SceneMutationError(f"{label} must contain exactly two numbers") from exc
    if not all(math.isfinite(item) for item in pair):
        raise SceneMutationError(f"{label} must contain finite numbers")
    return pair


ContainerRegionUpdate: TypeAlias = tuple[
    dict[str, Any],
    tuple[float, float],
    tuple[float, float] | None,
]


def _translated_container_regions(
    task: Mapping[str, Any],
    entity: str,
    delta_xy_m: tuple[float, float],
) -> tuple[ContainerRegionUpdate, ...]:
    dx, dy = _finite_pair(delta_xy_m, "delta_xy_m")
    updates = []
    for index, region in enumerate(_container_regions_for(task, entity)):
        center = _finite_pair(
            region.get("center"),
            f"container region {index} for {entity!r}.center",
        )
        updates.append((region, (center[0] + dx, center[1] + dy), None))
    return tuple(updates)


def _rotated_container_regions(
    task: Mapping[str, Any],
    entity: str,
    entity_region: Mapping[str, Any],
    new_runtime_yaw_deg: float,
) -> tuple[ContainerRegionUpdate, ...]:
    regions = _container_regions_for(task, entity)
    if not regions:
        return ()
    entity_center = _finite_pair(
        entity_region.get("center"),
        f"region for {entity!r}.center",
    )
    current_runtime_yaw = _fixed_runtime_yaw(entity_region, entity)
    delta_turns = _quarter_turn(
        new_runtime_yaw_deg - current_runtime_yaw,
        f"container-region yaw delta for {entity!r}",
    )
    angle = math.radians(delta_turns * 90.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    updates = []
    for index, region in enumerate(regions):
        unsupported_orientation = {
            key
            for key in ("yaw", "yaw_deg", "euler", "orientation", "rotation")
            if key in region
        }
        if unsupported_orientation:
            raise SceneMutationError(
                f"container region {index} for {entity!r} has unsupported orientation fields "
                f"{sorted(unsupported_orientation)}"
            )
        center = _finite_pair(
            region.get("center"),
            f"container region {index} for {entity!r}.center",
        )
        inner_size = _finite_pair(
            region.get("inner_size"),
            f"container region {index} for {entity!r}.inner_size",
        )
        if inner_size[0] <= 0.0 or inner_size[1] <= 0.0:
            raise SceneMutationError(
                f"container region {index} for {entity!r}.inner_size must be positive"
            )
        relative_x = center[0] - entity_center[0]
        relative_y = center[1] - entity_center[1]
        rotated_center = (
            entity_center[0] + relative_x * cosine - relative_y * sine,
            entity_center[1] + relative_x * sine + relative_y * cosine,
        )
        rotated_size = (
            (inner_size[1], inner_size[0])
            if delta_turns % 2
            else inner_size
        )
        updates.append((region, rotated_center, rotated_size))
    return tuple(updates)


def _apply_container_region_updates(
    updates: tuple[ContainerRegionUpdate, ...],
) -> None:
    for region, center, inner_size in updates:
        region["center"] = [center[0], center[1]]
        if inner_size is not None:
            region["inner_size"] = [inner_size[0], inner_size[1]]


def _fixed_runtime_yaw(region: Mapping[str, Any], entity: str) -> float:
    random_config = region.get("random_config")
    yaw_range = random_config.get("yaw_rotation") if isinstance(random_config, Mapping) else None
    if not isinstance(yaw_range, (list, tuple)) or len(yaw_range) != 2:
        raise SceneMutationError(
            f"container entity {entity!r} requires a fixed runtime yaw before rotation"
        )
    try:
        low, high = float(yaw_range[0]), float(yaw_range[1])
    except (TypeError, ValueError) as exc:
        raise SceneMutationError(
            f"container entity {entity!r} runtime yaw must be numeric"
        ) from exc
    if not all(math.isfinite(value) for value in (low, high)) or not math.isclose(
        low,
        high,
        abs_tol=1e-9,
    ):
        raise SceneMutationError(
            f"container entity {entity!r} requires a fixed runtime yaw before rotation"
        )
    return low


def _quarter_turn(yaw_deg: float, label: str) -> int:
    if not math.isfinite(yaw_deg):
        raise SceneMutationError(f"{label} must be finite")
    turns = round(yaw_deg / 90.0)
    if not math.isclose(yaw_deg, turns * 90.0, abs_tol=1e-9):
        raise SceneMutationError(
            f"{label} cannot be represented by the axis-aligned container-region schema"
        )
    return turns


def _shift_fixed_region(
    region: dict[str, Any],
    *,
    support: str,
    delta_xy_m: tuple[float, float],
) -> None:
    if not support:
        raise SceneMutationError("support name cannot be empty")
    if len(delta_xy_m) != 2:
        raise SceneMutationError("delta_xy_m must contain two numbers")
    current_support = str(
        region.get("target") or region.get("B") or region.get("parent_fixture") or ""
    )
    if current_support != support:
        raise SceneMutationError(
            f"typed XY mutation cannot change support from {current_support!r} to {support!r}"
        )
    dx, dy = (float(value) for value in delta_xy_m)
    random_config = _random_config(region)
    pos_range = random_config.get("pos_range")
    if not (
        isinstance(pos_range, list)
        and len(pos_range) == 2
        and all(isinstance(value, list) and len(value) >= 2 for value in pos_range)
    ):
        raise SceneMutationError(f"region for {region.get('object')} has no XY pos_range")
    midpoint = [
        (float(pos_range[0][index]) + float(pos_range[1][index])) / 2.0
        for index in range(2)
    ]
    z_midpoint = sum(
        float(value[2]) if len(value) > 2 else 0.0 for value in pos_range
    ) / 2.0
    fixed_position = [midpoint[0] + dx, midpoint[1] + dy, z_midpoint]
    random_config["pos_range"] = [fixed_position.copy(), fixed_position.copy()]
    center = region.get("center")
    if isinstance(center, list) and len(center) >= 2:
        center[0] = float(center[0]) + dx
        center[1] = float(center[1]) + dy
    runtime = region.get("runtime_placement")
    if isinstance(runtime, dict):
        offset = runtime.get("offset_xy")
        if isinstance(offset, list) and len(offset) >= 2:
            offset[0] = float(offset[0]) + dx
            offset[1] = float(offset[1]) + dy


def _fixture(arena: Mapping[str, Any], name: str) -> dict[str, Any]:
    fixtures = arena.get("fixtures")
    if not isinstance(fixtures, list):
        raise SceneMutationError("arena.fixtures must be a list")
    matches = [value for value in fixtures if isinstance(value, dict) and value.get("name") == name]
    if len(matches) != 1:
        raise SceneMutationError(f"support {name!r} must identify exactly one fixture, found {len(matches)}")
    return matches[0]


def _current_support_height(fixture: Mapping[str, Any], support: str) -> float:
    if fixture.get("support_surface_z") is not None:
        return float(fixture["support_surface_z"])
    translation = fixture.get("translation")
    if isinstance(translation, list) and len(translation) >= 3:
        return float(translation[2])
    raise SceneMutationError(f"support {support!r} has no absolute height")


def _shift_fixture(fixture: dict[str, Any], delta: float) -> None:
    translation = fixture.get("translation")
    if isinstance(translation, list) and len(translation) >= 3:
        translation[2] = float(translation[2]) + delta
    if fixture.get("support_surface_z") is not None:
        fixture["support_surface_z"] = float(fixture["support_surface_z"]) + delta


def _shift_region_support_height(region: dict[str, Any], delta: float, label: str) -> None:
    random_config = _random_config(region)
    height_fields = []
    if region.get("support_surface_z") is not None:
        region["support_surface_z"] = float(region["support_surface_z"]) + delta
        height_fields.append("support_surface_z")
    if random_config.get("support_surface_z") is not None:
        random_config["support_surface_z"] = float(random_config["support_surface_z"]) + delta
        height_fields.append("random_config.support_surface_z")
    runtime = region.get("runtime_placement")
    if isinstance(runtime, dict) and runtime.get("support_surface_z") is not None:
        runtime["support_surface_z"] = float(runtime["support_surface_z"]) + delta
        height_fields.append("runtime_placement.support_surface_z")
    if not height_fields:
        raise SceneMutationError(
            f"{label} is linked to the moved support but has no explicit support height"
        )


def _fixture_descendants(arena: Mapping[str, Any], support: str) -> tuple[dict[str, Any], ...]:
    fixtures = arena.get("fixtures")
    if not isinstance(fixtures, list):
        raise SceneMutationError("arena.fixtures must be a list")
    descendants: list[dict[str, Any]] = []
    frontier = [support]
    seen = {support}
    while frontier:
        parent = frontier.pop(0)
        for fixture in fixtures:
            if not isinstance(fixture, dict) or not fixture.get("name"):
                raise SceneMutationError("every arena fixture must be a named mapping")
            if fixture.get("parent_fixture") != parent:
                continue
            name = str(fixture["name"])
            if name in seen:
                raise SceneMutationError(f"support fixture cycle contains {name}")
            seen.add(name)
            descendants.append(fixture)
            frontier.append(name)
    return tuple(descendants)


def _set_support_height(
    task: dict[str, Any],
    arena: dict[str, Any],
    support: str,
    world_z_m: float,
) -> None:
    fixture = _fixture(arena, support)
    current = _current_support_height(fixture, support)
    delta = world_z_m - current
    descendants = _fixture_descendants(arena, support)
    affected_supports = {support, *(str(item["name"]) for item in descendants)}
    _shift_fixture(fixture, delta)
    fixture["support_surface_z"] = world_z_m
    for child in descendants:
        _shift_fixture(child, delta)
    affected_entities = set()
    for region in _regions(task):
        region_support = region.get("target") or region.get("B") or region.get("parent_fixture")
        if region_support in affected_supports:
            entity = str(region.get("object") or region.get("A") or "")
            if entity:
                affected_entities.add(entity)
            _shift_region_support_height(
                region,
                delta,
                f"task region for {region.get('object') or region.get('A')}",
            )
    _shift_container_region_heights(task, affected_entities, delta)
    source_regions = task.get("source_regions") or []
    if not isinstance(source_regions, list):
        raise SceneMutationError("tasks[0].source_regions must be a list")
    for region in source_regions:
        if not isinstance(region, dict):
            raise SceneMutationError("every tasks[0].source_regions entry must be a mapping")
        region_support = region.get("target") or region.get("B") or region.get("parent_fixture")
        if region_support in affected_supports:
            _shift_region_support_height(
                region,
                delta,
                f"source region for {region.get('robot_base') or region.get('A')}",
            )
    arena_regions = arena.get("regions") or []
    if not isinstance(arena_regions, list):
        raise SceneMutationError("arena.regions must be a list")
    for region in arena_regions:
        if not isinstance(region, dict):
            raise SceneMutationError("every arena.regions entry must be a mapping")
        region_support = region.get("target") or region.get("B") or region.get("parent_fixture")
        if region_support in affected_supports:
            _shift_region_support_height(
                region,
                delta,
                f"arena region for {region.get('object') or region.get('A')}",
            )


def _shift_container_region_heights(
    task: Mapping[str, Any],
    affected_entities: set[str],
    delta: float,
) -> None:
    for entity in sorted(affected_entities):
        for index, region in enumerate(_container_regions_for(task, entity)):
            value = region.get("interior_support_z")
            try:
                height = float(value)
            except (TypeError, ValueError) as exc:
                raise SceneMutationError(
                    f"container region {index} for {entity!r} has no numeric interior_support_z"
                ) from exc
            if not math.isfinite(height):
                raise SceneMutationError(
                    f"container region {index} for {entity!r}.interior_support_z must be finite"
                )
            region["interior_support_z"] = height + delta


def _write_revision(
    output_dir: Path,
    task_document: dict[str, Any],
    arena_document: dict[str, Any],
    mutation_document: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (staging / "simbox_task.yaml").write_text(
            yaml.safe_dump(task_document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (staging / "simbox_arena.yaml").write_text(
            yaml.safe_dump(arena_document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (staging / "scene_mutations.json").write_text(
            json.dumps(mutation_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise SceneMutationError(f"cannot write derived scene revision: {exc}") from exc
