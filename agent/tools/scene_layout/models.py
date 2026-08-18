"""Canonical scene and support contracts built from SimBox documents."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class SceneSpecError(ValueError):
    """A source document cannot be normalized without losing semantics."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneSpecError(f"{label} must be a mapping")
    return copy.deepcopy(dict(value))


def _first_task(document: Mapping[str, Any]) -> dict[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise SceneSpecError("SimBox task document must contain exactly one tasks[0] mapping")
    return copy.deepcopy(dict(tasks[0]))


def _number(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SceneSpecError(f"{label} must be numeric") from exc


def _optional_vec(value: Any, size: int, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < size:
        raise SceneSpecError(f"{label} must contain at least {size} numbers")
    return tuple(_number(value[index], f"{label}[{index}]") for index in range(size))


def _optional_range_xy(
    value: Any,
    label: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SceneSpecError(f"{label} must contain exactly two endpoints")
    endpoints = tuple(_optional_vec(endpoint, 2, f"{label}[{index}]") for index, endpoint in enumerate(value))
    assert all(endpoint is not None for endpoint in endpoints)
    low = (
        min(endpoints[0][0], endpoints[1][0]),
        min(endpoints[0][1], endpoints[1][1]),
    )
    high = (
        max(endpoints[0][0], endpoints[1][0]),
        max(endpoints[0][1], endpoints[1][1]),
    )
    return low, high


def _entity(value: Mapping[str, Any]) -> str:
    return str(value.get("object") or value.get("robot_base") or value.get("A") or "").strip()


def _entity_aliases(task: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for label in ("objects", "robots"):
        values = task.get(label) or []
        if not isinstance(values, list):
            raise SceneSpecError(f"tasks[0].{label} must be a list")
        for index, item in enumerate(values):
            if not isinstance(item, Mapping) or not item.get("name"):
                raise SceneSpecError(f"tasks[0].{label}[{index}] must have a name")
            canonical = str(item["name"])
            for alias in (canonical, item.get("source_name")):
                if not alias:
                    continue
                alias = str(alias)
                existing = aliases.get(alias)
                if existing is not None and existing != canonical:
                    raise SceneSpecError(
                        f"entity alias {alias!r} identifies both {existing!r} and {canonical!r}"
                    )
                aliases[alias] = canonical
    return aliases


def _canonical_entity(value: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    entity = _entity(value)
    return aliases.get(entity, entity)


def _fixture_aliases(arena_document: Mapping[str, Any]) -> dict[str, str]:
    fixtures = arena_document.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise SceneSpecError("arena.fixtures must be a list")
    aliases: dict[str, str] = {}
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, Mapping) or not fixture.get("name"):
            raise SceneSpecError(f"arena.fixtures[{index}] must have a name")
        canonical = str(fixture["name"])
        for alias in (canonical, fixture.get("source_name")):
            if not alias:
                continue
            alias = str(alias)
            existing = aliases.get(alias)
            if existing is not None and existing != canonical:
                raise SceneSpecError(
                    f"fixture alias {alias!r} identifies both {existing!r} and {canonical!r}"
                )
            aliases[alias] = canonical
    return aliases


def _semantic_support(
    value: Mapping[str, Any],
    support_aliases: Mapping[str, str] | None = None,
) -> str:
    values = {
        str(value[key]).strip()
        for key in ("parent_fixture", "support_target_fixture")
        if value.get(key)
    }
    if len(values) > 1:
        raise SceneSpecError(f"region has conflicting semantic supports {sorted(values)!r}")
    support = next(iter(values), "")
    return (support_aliases or {}).get(support, support)


def _merge_source_region_geometry(
    region: Mapping[str, Any],
    source_region: Mapping[str, Any] | None,
    *,
    index: int,
    support_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(region))
    if source_region is None:
        return merged

    task_support = _semantic_support(region, support_aliases)
    source_support = _semantic_support(source_region, support_aliases)
    if task_support and source_support and task_support != source_support:
        raise SceneSpecError(
            f"regions[{index}] and its source region disagree on semantic support "
            f"{task_support!r} != {source_support!r}"
        )
    task_runtime_support = str(
        region.get("target") or region.get("B") or region.get("support_collision_plane") or ""
    ).strip()
    source_runtime_support = str(
        source_region.get("target")
        or source_region.get("B")
        or source_region.get("support_collision_plane")
        or ""
    ).strip()
    if support_aliases is not None:
        task_runtime_support = support_aliases.get(task_runtime_support, task_runtime_support)
        source_runtime_support = support_aliases.get(source_runtime_support, source_runtime_support)
    if (
        not task_support
        and not source_support
        and task_runtime_support
        and source_runtime_support
        and task_runtime_support != source_runtime_support
    ):
        raise SceneSpecError(
            f"regions[{index}] has conflicting semantic supports "
            f"{task_runtime_support!r} and {source_runtime_support!r}"
        )
    task_size = _optional_vec(region.get("size"), 2, f"regions[{index}].size")
    source_size = _optional_vec(
        source_region.get("size"),
        2,
        f"source region for regions[{index}].size",
    )
    if task_size is not None and source_size is not None and task_size != source_size:
        raise SceneSpecError(
            f"regions[{index}] and its source region disagree on entity footprint"
        )

    for key in (
        "center",
        "size",
        "runtime_placement",
        "support_surface_z",
        "parent_fixture",
        "support_target_fixture",
    ):
        if merged.get(key) is None and source_region.get(key) is not None:
            merged[key] = copy.deepcopy(source_region[key])
    task_random = merged.get("random_config")
    source_random = source_region.get("random_config")
    if task_random is None and isinstance(source_random, Mapping):
        merged["random_config"] = copy.deepcopy(dict(source_random))
    elif isinstance(task_random, Mapping) and isinstance(source_random, Mapping):
        random_config = copy.deepcopy(dict(task_random))
        for key in ("pos_range", "yaw_rotation", "support_surface_z"):
            if random_config.get(key) is None and source_random.get(key) is not None:
                random_config[key] = copy.deepcopy(source_random[key])
        merged["random_config"] = random_config
    return merged


@dataclass(frozen=True)
class RegionSpec:
    entity: str
    support: str
    runtime_support: str
    name: str | None = None
    center_xy: tuple[float, float] | None = None
    size_xy: tuple[float, float] | None = None
    position_range_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
    yaw_range_deg: tuple[float, float] | None = None
    authored_yaw_deg: float = 0.0
    runtime_placement: dict[str, Any] | None = None
    support_surface_z: float | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        index: int,
        *,
        authored_yaw_deg: float = 0.0,
    ) -> "RegionSpec":
        entity = _entity(value)
        semantic_values = {
            str(value[key]).strip()
            for key in ("parent_fixture", "support_target_fixture")
            if value.get(key)
        }
        runtime_values = {
            str(value[key]).strip()
            for key in ("target", "B", "support_collision_plane")
            if value.get(key)
        }
        if len(semantic_values) > 1:
            raise SceneSpecError(
                f"regions[{index}] has conflicting semantic supports {sorted(semantic_values)!r}"
            )
        if len(runtime_values) > 1:
            raise SceneSpecError(
                f"regions[{index}] has conflicting runtime supports {sorted(runtime_values)!r}"
            )
        semantic_support = next(iter(semantic_values), "")
        runtime_support = next(iter(runtime_values), "")
        support = semantic_support or runtime_support
        runtime_support = runtime_support or support
        if not entity or not support or not runtime_support:
            raise SceneSpecError(f"regions[{index}] must identify both entity and support")
        support_z = value.get("support_surface_z")
        if support_z is None and isinstance(value.get("random_config"), Mapping):
            support_z = value["random_config"].get("support_surface_z")
        random_config = value.get("random_config")
        position_range = None
        yaw_range = None
        if isinstance(random_config, Mapping):
            position_range = _optional_range_xy(
                random_config.get("pos_range"),
                f"regions[{index}].random_config.pos_range",
            )
            yaw_range = _optional_vec(
                random_config.get("yaw_rotation"),
                2,
                f"regions[{index}].random_config.yaw_rotation",
            )
        runtime = value.get("runtime_placement")
        return cls(
            entity=entity,
            support=support,
            runtime_support=runtime_support,
            name=str(value["name"]) if value.get("name") else None,
            center_xy=_optional_vec(value.get("center"), 2, f"regions[{index}].center"),
            size_xy=_optional_vec(value.get("size"), 2, f"regions[{index}].size"),
            position_range_xy=position_range,
            yaw_range_deg=yaw_range,
            authored_yaw_deg=float(authored_yaw_deg),
            runtime_placement=_mapping(runtime, f"regions[{index}].runtime_placement") if runtime is not None else None,
            support_surface_z=_number(support_z, f"regions[{index}].support_surface_z") if support_z is not None else None,
        )


@dataclass(frozen=True)
class SupportRelation:
    entity: str
    support: str
    source: str


@dataclass(frozen=True)
class SupportGraph:
    supports: tuple[str, ...]
    relations: tuple[SupportRelation, ...]
    runtime_relations: tuple[SupportRelation, ...] = ()

    @classmethod
    def from_documents(
        cls,
        task_document: Mapping[str, Any],
        arena_document: Mapping[str, Any],
    ) -> "SupportGraph":
        task = _first_task(task_document)
        aliases = _entity_aliases(task)
        support_aliases = _fixture_aliases(arena_document)
        fixtures = arena_document.get("fixtures") or []
        if not isinstance(fixtures, list):
            raise SceneSpecError("arena.fixtures must be a list")
        supports: set[str] = set()
        relations: list[SupportRelation] = []
        runtime_relations: list[SupportRelation] = []
        source_regions = task.get("source_regions") or []
        if not isinstance(source_regions, list):
            raise SceneSpecError("tasks[0].source_regions must be a list")
        source_by_entity: dict[str, Mapping[str, Any]] = {}
        for index, source_region in enumerate(source_regions):
            if not isinstance(source_region, Mapping):
                raise SceneSpecError(f"tasks[0].source_regions[{index}] must be a mapping")
            entity = _canonical_entity(source_region, aliases)
            if not entity:
                continue
            if entity in source_by_entity:
                raise SceneSpecError(f"entity {entity!r} owns multiple source regions")
            source_by_entity[entity] = source_region
        for index, fixture in enumerate(fixtures):
            if not isinstance(fixture, Mapping) or not fixture.get("name"):
                raise SceneSpecError(f"arena.fixtures[{index}] must have a name")
            name = str(fixture["name"])
            supports.add(name)
            parent = str(fixture.get("parent_fixture") or "").strip()
            if parent:
                relations.append(SupportRelation(name, parent, "arena.fixture"))
        for index, region in enumerate(task.get("regions") or []):
            if not isinstance(region, Mapping):
                raise SceneSpecError(f"tasks[0].regions[{index}] must be a mapping")
            entity = _canonical_entity(region, aliases)
            merged = _merge_source_region_geometry(
                region,
                    source_by_entity.get(entity),
                    index=index,
                    support_aliases=support_aliases,
            )
            merged["object"] = entity
            normalized = RegionSpec.from_mapping(merged, index)
            relations.append(SupportRelation(normalized.entity, normalized.support, "task.region"))
            runtime_relations.append(
                SupportRelation(
                    normalized.entity,
                    normalized.runtime_support,
                    "task.region.runtime",
                )
            )
        task_entities = {
            _canonical_entity(region, aliases)
            for region in task.get("regions") or []
            if isinstance(region, Mapping)
        }
        for index, region in enumerate(source_regions):
            entity = _canonical_entity(region, aliases)
            if entity and entity not in task_entities:
                source_region = dict(region)
                source_region["object"] = entity
                normalized = RegionSpec.from_mapping(source_region, index)
                relations.append(
                    SupportRelation(entity, normalized.support, "task.source_region")
                )
        for index, region in enumerate(arena_document.get("regions") or []):
            if not isinstance(region, Mapping):
                raise SceneSpecError(f"arena.regions[{index}] must be a mapping")
            normalized = RegionSpec.from_mapping(region, index)
            relations.append(SupportRelation(normalized.entity, normalized.support, "arena.region"))
        graph = cls(
            tuple(sorted(supports)),
            tuple(relations),
            tuple(runtime_relations),
        )
        graph._validate()
        return graph

    def _validate(self) -> None:
        self._validate_unique_parents(self.relations, "semantic supports")
        self._validate_unique_parents(self.runtime_relations, "runtime supports")
        parents: dict[str, str] = {}
        for relation in self.relations:
            parents[relation.entity] = relation.support
        for start in parents:
            visited: set[str] = set()
            current = start
            while current in parents:
                if current in visited:
                    raise SceneSpecError(f"support relation cycle contains {current}")
                visited.add(current)
                current = parents[current]

    @staticmethod
    def _validate_unique_parents(
        relations: tuple[SupportRelation, ...],
        label: str,
    ) -> None:
        parents: dict[str, str] = {}
        for relation in relations:
            existing = parents.get(relation.entity)
            if existing is not None and existing != relation.support:
                raise SceneSpecError(
                    f"{relation.entity} has conflicting {label} {existing!r} and {relation.support!r}"
                )
            parents[relation.entity] = relation.support

    def entities_on(self, support: str) -> tuple[str, ...]:
        return tuple(
            sorted({relation.entity for relation in self.relations if relation.support == support})
        )

    def runtime_targets_for(self, entity: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    relation.support
                    for relation in self.runtime_relations
                    if relation.entity == entity
                }
            )
        )


@dataclass(frozen=True)
class SceneSpec:
    source_task: str
    source_arena: str
    source_task_hash: str
    source_arena_hash: str
    task_name: str
    objects: tuple[str, ...]
    robots: tuple[str, ...]
    fixtures: tuple[str, ...]
    regions: tuple[RegionSpec, ...]
    support_graph: SupportGraph
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_documents(
        cls,
        task_document: Mapping[str, Any],
        arena_document: Mapping[str, Any],
        *,
        source_task: Path,
        source_arena: Path,
    ) -> "SceneSpec":
        task = _first_task(task_document)
        aliases = _entity_aliases(task)
        support_aliases = _fixture_aliases(arena_document)

        def names(values: Any, label: str) -> tuple[str, ...]:
            if values is None:
                return ()
            if not isinstance(values, list):
                raise SceneSpecError(f"{label} must be a list")
            result = []
            for index, item in enumerate(values):
                if not isinstance(item, Mapping) or not item.get("name"):
                    raise SceneSpecError(f"{label}[{index}] must have a name")
                result.append(str(item["name"]))
            return tuple(result)

        region_values = task.get("regions") or []
        if not isinstance(region_values, list):
            raise SceneSpecError("tasks[0].regions must be a list")
        source_regions = task.get("source_regions") or []
        if not isinstance(source_regions, list):
            raise SceneSpecError("tasks[0].source_regions must be a list")
        source_by_entity: dict[str, Mapping[str, Any]] = {}
        for index, source_region in enumerate(source_regions):
            if not isinstance(source_region, Mapping):
                raise SceneSpecError(f"tasks[0].source_regions[{index}] must be a mapping")
            entity = _canonical_entity(source_region, aliases)
            if not entity:
                continue
            if entity in source_by_entity:
                raise SceneSpecError(f"entity {entity!r} owns multiple source regions")
            source_by_entity[entity] = source_region
        authored_yaws = {
            str(item["name"]): float((item.get("euler") or [0.0, 0.0, 0.0])[2])
            for item in task.get("objects") or []
            if isinstance(item, Mapping)
            and item.get("name")
            and isinstance(item.get("euler") or [0.0, 0.0, 0.0], (list, tuple))
            and len(item.get("euler") or [0.0, 0.0, 0.0]) >= 3
        }
        normalized_regions = []
        for index, region in enumerate(region_values):
            if not isinstance(region, Mapping):
                raise SceneSpecError(f"tasks[0].regions[{index}] must be a mapping")
            entity = _canonical_entity(region, aliases)
            merged = _merge_source_region_geometry(
                region,
                source_by_entity.get(entity),
                index=index,
                support_aliases=support_aliases,
            )
            merged["object"] = entity
            normalized_regions.append(
                RegionSpec.from_mapping(
                    merged,
                    index,
                    authored_yaw_deg=authored_yaws.get(entity, 0.0),
                )
            )
        regions = tuple(normalized_regions)
        source_task = source_task.resolve()
        source_arena = source_arena.resolve()
        return cls(
            source_task=str(source_task),
            source_arena=str(source_arena),
            source_task_hash=_file_hash(source_task),
            source_arena_hash=_file_hash(source_arena),
            task_name=str(task.get("name") or task.get("task_id") or source_task.stem),
            objects=names(task.get("objects"), "tasks[0].objects"),
            robots=names(task.get("robots"), "tasks[0].robots"),
            fixtures=names(arena_document.get("fixtures"), "arena.fixtures"),
            regions=regions,
            support_graph=SupportGraph.from_documents(task_document, arena_document),
            metadata=copy.deepcopy(task.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def load_scene_spec(source_task: Path, repo_root: Path | None = None) -> SceneSpec:
    source_task = source_task.resolve()
    task_document = _load_yaml(source_task)
    task = _first_task(task_document)
    arena_value = task.get("arena_file")
    if not isinstance(arena_value, str) or not arena_value:
        raise SceneSpecError("tasks[0].arena_file is required")
    arena_path = Path(arena_value)
    if not arena_path.is_absolute():
        roots = [source_task.parent]
        if repo_root is not None:
            roots.insert(0, repo_root.resolve())
        arena_path = next((root / arena_path for root in roots if (root / arena_path).is_file()), roots[0] / arena_path)
    arena_document = _load_yaml(arena_path)
    return SceneSpec.from_documents(
        task_document,
        arena_document,
        source_task=source_task,
        source_arena=arena_path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SceneSpecError(f"cannot read YAML document {path}: {exc}") from exc
    return _mapping(value, str(path))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
