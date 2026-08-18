"""Measured scene-layout populations and deterministic static validation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from agent.tools.feedback import RepairAction, classify_failure

from .evolution import CandidateAggregate, CandidateGenome
from .models import RegionSpec, SceneSpec, SceneSpecError
from .mutations import (
    MoveEntityOnSupport,
    RotateEntityOnSupport,
    SceneLayoutCompiler,
    SceneMutationError,
    scene_revision_for,
)


class SceneLayoutBlocked(ValueError):
    """The requested layout change lacks the measurements needed for a safe edit."""

    action = "block"

    def __init__(
        self,
        failure_code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{failure_code}: {message}")
        self.failure_code = failure_code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "failure_code": self.failure_code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class SupportSurfaceMeasurement:
    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    yaw_deg: float
    center_source: str
    size_source: str


@dataclass(frozen=True)
class PlacementMeasurement:
    entity: str
    support: str
    runtime_support: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    position_range_xy: tuple[tuple[float, float], tuple[float, float]]
    runtime_yaw_deg: float
    authored_yaw_deg: float
    runtime_offset_xy: tuple[float, float] | None = None

    @property
    def actual_yaw_deg(self) -> float:
        return self.authored_yaw_deg + self.runtime_yaw_deg

    @property
    def jitter_half_xy(self) -> tuple[float, float]:
        low, high = self.position_range_xy
        return (high[0] - low[0]) / 2.0, (high[1] - low[1]) / 2.0


@dataclass(frozen=True)
class StaticLayoutValidation:
    candidate_id: str
    generation: int
    scene_revision: str
    hard_constraints: dict[str, bool]
    failure_code: str | None
    message: str
    derived_task_path: str | None = None
    derived_arena_path: str | None = None
    mutation_path: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @property
    def hard_ok(self) -> bool:
        return bool(self.hard_constraints) and all(self.hard_constraints.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SceneLayoutPlanner:
    """Build layout genomes only from source-declared, cross-checked geometry."""

    population_size = 8

    def __init__(
        self,
        source_task_path: Path,
        source_arena_path: Path,
        manipulated_entity: str,
        failure_code: str,
        profile_id: str,
        *,
        protected_entities: Iterable[str] = (),
    ) -> None:
        self.source_task_path = source_task_path.resolve()
        self.source_arena_path = source_arena_path.resolve()
        self.manipulated_entity = manipulated_entity
        self.failure_code = failure_code
        self.profile_id = profile_id
        self.protected_entities = frozenset(
            str(entity).strip() for entity in protected_entities if str(entity).strip()
        )
        repair = classify_failure(failure_code, "workspace")
        layout_authorized = repair.action in {
            RepairAction.MUTATE_LAYOUT,
            RepairAction.NEXT_CANDIDATE,
        } or bool(repair.allowed_scene_mutations)
        if not layout_authorized:
            raise SceneLayoutBlocked(
                "LAYOUT_FAILURE_NOT_MUTABLE",
                f"failure {failure_code!r} does not authorize a scene-layout mutation",
                {"failure_code": failure_code},
            )
        task_document = _load_yaml(self.source_task_path)
        arena_document = _load_yaml(self.source_arena_path)
        try:
            self.scene_spec = SceneSpec.from_documents(
                task_document,
                arena_document,
                source_task=self.source_task_path,
                source_arena=self.source_arena_path,
            )
        except SceneSpecError as exc:
            raise SceneLayoutBlocked("LAYOUT_SCHEMA_INVALID", str(exc)) from exc
        regions = [
            region for region in self.scene_spec.regions if region.entity == manipulated_entity
        ]
        if len(regions) != 1:
            raise SceneLayoutBlocked(
                "LAYOUT_ENTITY_REGION_INVALID",
                f"entity {manipulated_entity!r} must own exactly one runtime region",
                {"region_count": len(regions)},
            )
        self.region = regions[0]
        self.placement = _placement_measurement(self.region)
        peer_regions = [
            region
            for region in self.scene_spec.regions
            if region.entity != manipulated_entity
            and region.support == self.region.support
            and region.entity in self.scene_spec.objects
        ]
        try:
            self.peers = tuple(_placement_measurement(region) for region in peer_regions)
        except SceneLayoutBlocked as exc:
            raise SceneLayoutBlocked(
                exc.failure_code,
                "every object on the manipulated entity's support must be measurable "
                "before overlap validation",
                exc.details,
            ) from exc
        self.placements = (self.placement, *self.peers)
        self.placement_by_entity = {
            placement.entity: placement for placement in self.placements
        }
        runtime_supports = {placement.runtime_support for placement in self.placements}
        self.support_by_runtime = {
            runtime_support: _support_measurement(
                arena_document,
                self.scene_spec.regions,
                runtime_support,
            )
            for runtime_support in sorted(runtime_supports)
        }
        self.support = self.support_by_runtime[self.placement.runtime_support]
        for placement in self.placements:
            self._validate_runtime_anchor(placement)
        overlapping_peers = sorted(
            self.peers,
            key=lambda peer: (
                not _bounds_overlap(
                    _occupancy_bounds(self.placement),
                    _occupancy_bounds(peer),
                ),
                peer.entity,
            ),
        )
        self.movable_placements = tuple(
            placement
            for placement in (self.placement, *overlapping_peers)
            if placement.entity not in self.protected_entities
        )
        if not self.movable_placements:
            raise SceneLayoutBlocked(
                "LAYOUT_NO_MUTABLE_ENTITY",
                "every measured object on the support is protected",
                {"protected_entities": sorted(self.protected_entities)},
            )

    def initial_population(self) -> tuple[CandidateGenome, ...]:
        movable = self.movable_placements[: self.population_size]
        population = tuple(
            self._genome(
                index,
                0,
                placement,
                *self._initial_poses(placement)[index // len(movable)],
            )
            for index in range(self.population_size)
            for placement in (movable[index % len(movable)],)
        )
        self._validate_population(population, 0)
        return population

    def evolve(
        self,
        ranked: tuple[CandidateAggregate, ...],
        generation: int,
        population_size: int,
    ) -> Iterable[CandidateGenome]:
        if generation not in range(1, 5) or population_size != self.population_size:
            raise ValueError("SceneLayout v1 evolution requires generations 1-4 and K=8")
        if not ranked:
            raise ValueError("ranked parent candidates are required")
        primary = ranked[0].genome
        entity, primary_position, runtime_yaw = self._genome_pose(primary)
        placement = self.placement_by_entity[entity]
        support = self.support_by_runtime[placement.runtime_support]
        if len(ranked) > 1:
            secondary_entity, secondary_position, _ = self._genome_pose(
                ranked[1].genome
            )
            if secondary_entity == entity:
                anchor = (
                    (primary_position[0] + secondary_position[0]) / 2.0,
                    (primary_position[1] + secondary_position[1]) / 2.0,
                )
            else:
                anchor = primary_position
        else:
            anchor = primary_position
        bounds = self._feasible_bounds(
            placement,
            support,
            placement.authored_yaw_deg + runtime_yaw,
        )
        positions = _perimeter_positions(
            bounds,
            anchor_xy=_world_to_local(anchor, support),
            span_fraction=1.0 / (generation + 1.0),
        )
        population = tuple(
            self._genome(
                index,
                generation,
                placement,
                position,
                runtime_yaw,
                parent_id=primary.candidate_id,
            )
            for index, position in enumerate(positions)
        )
        self._validate_population(population, generation)
        return population

    def validate_candidate(
        self,
        candidate: CandidateGenome,
        output_dir: Path,
    ) -> StaticLayoutValidation:
        base_constraints = {
            "schema": False,
            "support": False,
            "containment": False,
            "no_overlap": False,
        }
        try:
            candidate_entity = self._candidate_entity(candidate)
        except SceneLayoutBlocked as exc:
            return StaticLayoutValidation(
                candidate.candidate_id,
                candidate.generation,
                candidate.scene_revision,
                base_constraints,
                exc.failure_code,
                exc.message,
            )
        if candidate.profile_id != self.profile_id:
            return StaticLayoutValidation(
                candidate.candidate_id,
                candidate.generation,
                candidate.scene_revision,
                base_constraints,
                "LAYOUT_PROFILE_MISMATCH",
                "candidate profile does not match the measured layout planner",
            )
        if candidate.scene_revision != scene_revision_for(
            self.scene_spec, candidate.mutations
        ):
            return StaticLayoutValidation(
                candidate.candidate_id,
                candidate.generation,
                candidate.scene_revision,
                base_constraints,
                "LAYOUT_REVISION_MISMATCH",
                "candidate scene revision does not match its mutation payload",
            )
        try:
            derived = SceneLayoutCompiler().compile(
                self.source_task_path,
                self.source_arena_path,
                list(candidate.mutations),
                output_dir,
            )
        except (SceneMutationError, SceneSpecError) as exc:
            return StaticLayoutValidation(
                candidate.candidate_id,
                candidate.generation,
                candidate.scene_revision,
                base_constraints,
                "LAYOUT_COMPILE_FAILED",
                str(exc),
            )

        constraints = dict(base_constraints)
        constraints["schema"] = True
        try:
            measured = SceneLayoutPlanner(
                derived.task_path,
                derived.arena_path,
                self.manipulated_entity,
                self.failure_code,
                self.profile_id,
                protected_entities=self.protected_entities,
            )
            source_supports = {
                placement.entity: (placement.support, placement.runtime_support)
                for placement in self.placements
            }
            derived_supports = {
                placement.entity: (placement.support, placement.runtime_support)
                for placement in measured.placements
            }
            constraints["support"] = (
                source_supports == derived_supports
                and candidate_entity in measured.placement_by_entity
            )
            constraints["containment"] = all(
                measured._contains(placement) for placement in measured.placements
            )
            constraints["no_overlap"] = measured._has_no_overlap()
            failure_code, message = _constraint_failure(constraints)
        except SceneLayoutBlocked as exc:
            failure_code = exc.failure_code
            message = exc.message

        result = StaticLayoutValidation(
            candidate.candidate_id,
            candidate.generation,
            derived.scene_revision,
            constraints,
            failure_code,
            message,
            str(derived.task_path),
            str(derived.arena_path),
            str(derived.mutation_path),
            (
                str(derived.task_path),
                str(derived.arena_path),
                str(derived.mutation_path),
                str(output_dir / "static_validation.json"),
            ),
        )
        (output_dir / "static_validation.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    def _genome(
        self,
        index: int,
        generation: int,
        placement: PlacementMeasurement,
        local_position: tuple[float, float],
        runtime_yaw_deg: float,
        *,
        parent_id: str | None = None,
    ) -> CandidateGenome:
        support = self.support_by_runtime[placement.runtime_support]
        world_position = _local_to_world(local_position, support)
        delta = (
            _stable_float(world_position[0] - placement.center_xy[0]),
            _stable_float(world_position[1] - placement.center_xy[1]),
        )
        mutations: list[MoveEntityOnSupport | RotateEntityOnSupport] = [
            MoveEntityOnSupport(placement.entity, placement.runtime_support, delta)
        ]
        if not math.isclose(
            runtime_yaw_deg,
            placement.runtime_yaw_deg,
            abs_tol=1e-9,
        ):
            mutations.append(
                RotateEntityOnSupport(
                    placement.entity,
                    _stable_float(runtime_yaw_deg),
                )
            )
        mutation_tuple = tuple(mutations)
        revision = scene_revision_for(self.scene_spec, mutation_tuple)
        candidate_id = (
            f"layout_g{generation}_{index:02d}_"
            f"{_candidate_signature(revision, self.profile_id, self.failure_code)[:10]}"
        )
        return CandidateGenome(
            candidate_id,
            generation,
            revision,
            self.profile_id,
            parent_id=parent_id,
            mutations=mutation_tuple,
        )

    def _genome_pose(
        self,
        candidate: CandidateGenome,
    ) -> tuple[str, tuple[float, float], float]:
        entity = self._candidate_entity(candidate)
        placement = self.placement_by_entity[entity]
        move = [
            mutation
            for mutation in candidate.mutations
            if isinstance(mutation, MoveEntityOnSupport)
            and mutation.entity == entity
        ]
        if len(move) != 1:
            raise ValueError("layout candidate must contain exactly one entity move")
        runtime_yaw = placement.runtime_yaw_deg
        rotations = [
            mutation
            for mutation in candidate.mutations
            if isinstance(mutation, RotateEntityOnSupport)
            and mutation.entity == entity
        ]
        if len(rotations) > 1:
            raise ValueError("layout candidate cannot contain multiple entity rotations")
        if rotations:
            runtime_yaw = rotations[0].yaw_offset_deg
        return entity, (
            placement.center_xy[0] + move[0].delta_xy_m[0],
            placement.center_xy[1] + move[0].delta_xy_m[1],
        ), runtime_yaw

    def _candidate_entity(self, candidate: CandidateGenome) -> str:
        moves = [
            mutation
            for mutation in candidate.mutations
            if isinstance(mutation, MoveEntityOnSupport)
        ]
        rotations = [
            mutation
            for mutation in candidate.mutations
            if isinstance(mutation, RotateEntityOnSupport)
        ]
        if (
            len(moves) != 1
            or len(rotations) > 1
            or len(candidate.mutations) != len(moves) + len(rotations)
            or any(rotation.entity != moves[0].entity for rotation in rotations)
        ):
            raise SceneLayoutBlocked(
                "LAYOUT_MUTATION_NOT_ALLOWED",
                "a layout genome must move exactly one measured entity and may rotate only that entity",
            )
        entity = moves[0].entity
        if entity in self.protected_entities:
            raise SceneLayoutBlocked(
                "LAYOUT_PROTECTED_ENTITY",
                f"entity {entity!r} is protected from scene-layout mutation",
                {"protected_entities": sorted(self.protected_entities)},
            )
        placement = self.placement_by_entity.get(entity)
        if placement is None or placement not in self.movable_placements:
            raise SceneLayoutBlocked(
                "LAYOUT_ENTITY_NOT_MOVABLE",
                f"entity {entity!r} is not a measured movable object on the task support",
            )
        if moves[0].support != placement.runtime_support:
            raise SceneLayoutBlocked(
                "LAYOUT_SUPPORT_CHANGED",
                f"entity {entity!r} cannot move from runtime support {placement.runtime_support!r}",
            )
        return entity

    def _initial_poses(
        self,
        placement: PlacementMeasurement,
    ) -> tuple[tuple[tuple[float, float], float], ...]:
        support = self.support_by_runtime[placement.runtime_support]
        current_positions = _perimeter_positions(
            self._feasible_bounds(
                placement,
                support,
                placement.actual_yaw_deg,
            ),
        )
        orthogonal_runtime_yaw = placement.runtime_yaw_deg + 90.0
        try:
            orthogonal_positions = _perimeter_positions(
                self._feasible_bounds(
                    placement,
                    support,
                    placement.authored_yaw_deg + orthogonal_runtime_yaw,
                ),
            )
        except SceneLayoutBlocked:
            orthogonal_positions = current_positions
            orthogonal_runtime_yaw = placement.runtime_yaw_deg
        return (
            (current_positions[1], placement.runtime_yaw_deg),
            (orthogonal_positions[0], orthogonal_runtime_yaw),
            (current_positions[6], placement.runtime_yaw_deg),
            (orthogonal_positions[2], orthogonal_runtime_yaw),
            (current_positions[3], placement.runtime_yaw_deg),
            (orthogonal_positions[5], orthogonal_runtime_yaw),
            (current_positions[4], placement.runtime_yaw_deg),
            (orthogonal_positions[7], orthogonal_runtime_yaw),
        )

    def _validate_population(
        self,
        population: tuple[CandidateGenome, ...],
        generation: int,
    ) -> None:
        if len(population) != self.population_size:
            raise SceneLayoutBlocked(
                "LAYOUT_POPULATION_INCOMPLETE",
                f"generation {generation} did not produce exactly K=8 candidates",
            )
        revisions = {candidate.scene_revision for candidate in population}
        identifiers = {candidate.candidate_id for candidate in population}
        if len(revisions) != self.population_size or len(identifiers) != self.population_size:
            raise SceneLayoutBlocked(
                "LAYOUT_POPULATION_DUPLICATE",
                f"generation {generation} contains duplicate mutation signatures",
            )

    def _validate_runtime_anchor(self, placement: PlacementMeasurement) -> None:
        support = self.support_by_runtime[placement.runtime_support]
        low, high = placement.position_range_xy
        midpoint = ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0)
        expected_center = (
            support.center_xy[0] + midpoint[0],
            support.center_xy[1] + midpoint[1],
        )
        if placement.runtime_offset_xy is not None and not _close_xy(
            placement.runtime_offset_xy, midpoint
        ):
            raise SceneLayoutBlocked(
                "LAYOUT_RUNTIME_OFFSET_CONFLICT",
                f"runtime_placement.offset_xy and pos_range disagree for {placement.entity!r}",
                {
                    "runtime_offset_xy": placement.runtime_offset_xy,
                    "position_range_midpoint_xy": midpoint,
                },
            )
        if not _close_xy(expected_center, placement.center_xy):
            raise SceneLayoutBlocked(
                "LAYOUT_RUNTIME_POSITION_CONFLICT",
                f"region center and runtime pos_range disagree for {placement.entity!r}",
                {
                    "declared_center_xy": placement.center_xy,
                    "runtime_center_xy": expected_center,
                    "runtime_support": placement.runtime_support,
                },
            )

    def _feasible_bounds(
        self,
        placement: PlacementMeasurement,
        support: SupportSurfaceMeasurement,
        actual_yaw_deg: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        width, depth = placement.size_xy
        relative_yaw = math.radians(actual_yaw_deg - support.yaw_deg)
        object_half_x = (
            abs(math.cos(relative_yaw)) * width + abs(math.sin(relative_yaw)) * depth
        ) / 2.0
        object_half_y = (
            abs(math.sin(relative_yaw)) * width + abs(math.cos(relative_yaw)) * depth
        ) / 2.0
        jitter_x, jitter_y = placement.jitter_half_xy
        support_yaw = math.radians(support.yaw_deg)
        jitter_local_x = abs(math.cos(support_yaw)) * jitter_x + abs(
            math.sin(support_yaw)
        ) * jitter_y
        jitter_local_y = abs(math.sin(support_yaw)) * jitter_x + abs(
            math.cos(support_yaw)
        ) * jitter_y
        available_half_x = support.size_xy[0] / 2.0 - object_half_x - jitter_local_x
        available_half_y = support.size_xy[1] / 2.0 - object_half_y - jitter_local_y
        if available_half_x <= 0.0 or available_half_y <= 0.0:
            raise SceneLayoutBlocked(
                "LAYOUT_SEARCH_SPACE_DEGENERATE",
                f"entity {placement.entity!r} has no measurable in-support XY search area",
                {
                    "support_size_xy": support.size_xy,
                    "entity_size_xy": placement.size_xy,
                    "runtime_jitter_half_xy": placement.jitter_half_xy,
                    "actual_yaw_deg": actual_yaw_deg,
                },
            )
        return (
            (-available_half_x, available_half_x),
            (-available_half_y, available_half_y),
        )

    def _contains(self, placement: PlacementMeasurement) -> bool:
        support = self.support_by_runtime[placement.runtime_support]
        local = _world_to_local(placement.center_xy, support)
        (min_x, max_x), (min_y, max_y) = self._feasible_bounds(
            placement,
            support,
            placement.actual_yaw_deg,
        )
        return (
            min_x - 1e-9 <= local[0] <= max_x + 1e-9
            and min_y - 1e-9 <= local[1] <= max_y + 1e-9
        )

    def _has_no_overlap(self) -> bool:
        return all(
            not _bounds_overlap(
                _occupancy_bounds(first),
                _occupancy_bounds(second),
            )
            for first, second in itertools.combinations(self.placements, 2)
        )



def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SceneLayoutBlocked("LAYOUT_SOURCE_UNREADABLE", f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SceneLayoutBlocked("LAYOUT_SCHEMA_INVALID", f"YAML root must be a mapping: {path}")
    return value


def _placement_measurement(region: RegionSpec) -> PlacementMeasurement:
    missing = [
        name
        for name, value in (
            ("center", region.center_xy),
            ("size", region.size_xy),
            ("random_config.pos_range", region.position_range_xy),
            ("random_config.yaw_rotation", region.yaw_range_deg),
        )
        if value is None
    ]
    if missing:
        raise SceneLayoutBlocked(
            "LAYOUT_MEASUREMENT_MISSING",
            f"entity {region.entity!r} lacks required layout measurements",
            {"entity": region.entity, "missing": missing},
        )
    assert region.center_xy is not None
    assert region.size_xy is not None
    assert region.position_range_xy is not None
    assert region.yaw_range_deg is not None
    if region.size_xy[0] <= 0.0 or region.size_xy[1] <= 0.0:
        raise SceneLayoutBlocked(
            "LAYOUT_MEASUREMENT_INVALID",
            f"entity {region.entity!r} footprint must be positive",
            {"size_xy": region.size_xy},
        )
    if not math.isclose(
        region.yaw_range_deg[0],
        region.yaw_range_deg[1],
        abs_tol=1e-9,
    ):
        raise SceneLayoutBlocked(
            "LAYOUT_RUNTIME_YAW_NONDETERMINISTIC",
            f"entity {region.entity!r} must have a fixed runtime yaw for static validation",
            {"yaw_range_deg": region.yaw_range_deg},
        )
    runtime_offset = None
    if region.runtime_placement is not None:
        frame = region.runtime_placement.get("frame")
        if frame != "parent_world_xy_offset":
            raise SceneLayoutBlocked(
                "LAYOUT_RUNTIME_FRAME_UNSUPPORTED",
                f"entity {region.entity!r} uses unsupported runtime placement frame {frame!r}",
            )
        runtime_offset = _xy(
            region.runtime_placement.get("offset_xy"),
            f"entity {region.entity!r}.runtime_placement.offset_xy",
        )
    return PlacementMeasurement(
        region.entity,
        region.support,
        region.runtime_support,
        region.center_xy,
        region.size_xy,
        region.position_range_xy,
        region.yaw_range_deg[0],
        region.authored_yaw_deg,
        runtime_offset,
    )


def _support_measurement(
    arena_document: Mapping[str, Any],
    regions: tuple[RegionSpec, ...],
    runtime_support: str,
) -> SupportSurfaceMeasurement:
    fixtures = arena_document.get("fixtures")
    if not isinstance(fixtures, list):
        raise SceneLayoutBlocked("LAYOUT_SCHEMA_INVALID", "arena.fixtures must be a list")
    matches = [
        fixture
        for fixture in fixtures
        if isinstance(fixture, Mapping) and str(fixture.get("name")) == runtime_support
    ]
    if len(matches) != 1:
        raise SceneLayoutBlocked(
            "LAYOUT_SUPPORT_MISSING",
            f"runtime support {runtime_support!r} must identify exactly one fixture",
            {"fixture_count": len(matches)},
        )
    fixture = matches[0]
    size, size_source = _support_size(fixture)
    yaw = _yaw(fixture.get("euler"), f"support {runtime_support!r}.euler")
    translation = fixture.get("translation")
    target_class = str(fixture.get("target_class") or "")
    if target_class == "PlaneObject":
        center = _xy(translation, f"support {runtime_support!r}.translation")
        center_source = "arena.translation"
    else:
        anchors = []
        for region in regions:
            if region.runtime_support != runtime_support:
                continue
            if region.center_xy is None or region.position_range_xy is None:
                continue
            low, high = region.position_range_xy
            anchors.append(
                (
                    region.entity,
                    (
                        region.center_xy[0] - (low[0] + high[0]) / 2.0,
                        region.center_xy[1] - (low[1] + high[1]) / 2.0,
                    ),
                )
            )
        if not anchors:
            raise SceneLayoutBlocked(
                "LAYOUT_SUPPORT_CENTER_MISSING",
                f"runtime support {runtime_support!r} has no auditable bbox-center anchor",
            )
        center = anchors[0][1]
        conflicts = [
            {"entity": entity, "inferred_center_xy": inferred}
            for entity, inferred in anchors[1:]
            if not _close_xy(center, inferred)
        ]
        if conflicts:
            raise SceneLayoutBlocked(
                "LAYOUT_SUPPORT_CENTER_CONFLICT",
                f"regions disagree on runtime support {runtime_support!r} bbox center",
                {
                    "reference": {"entity": anchors[0][0], "inferred_center_xy": center},
                    "conflicts": conflicts,
                },
            )
        center_source = "region.center-minus-pos-range-midpoint"
    return SupportSurfaceMeasurement(
        runtime_support,
        center,
        size,
        yaw,
        center_source,
        size_source,
    )


def _support_size(fixture: Mapping[str, Any]) -> tuple[tuple[float, float], str]:
    inline = fixture.get("size")
    if isinstance(inline, (list, tuple)) and len(inline) >= 2:
        size = float(inline[0]), float(inline[1])
        source = "arena.size"
    else:
        measured = fixture.get("asset_world_extents")
        if isinstance(measured, (list, tuple)) and len(measured) >= 2:
            size = float(measured[0]), float(measured[1])
            source = "arena.asset_world_extents"
        else:
            source_extents = fixture.get("asset_source_extents")
            scale = fixture.get("scale")
            if not (
                isinstance(source_extents, (list, tuple))
                and len(source_extents) >= 2
                and isinstance(scale, (list, tuple))
                and len(scale) >= 2
            ):
                raise SceneLayoutBlocked(
                    "LAYOUT_SUPPORT_SIZE_MISSING",
                    f"runtime support {fixture.get('name')!r} has no measured XY extent",
                )
            size = (
                float(source_extents[0]) * abs(float(scale[0])),
                float(source_extents[1]) * abs(float(scale[1])),
            )
            source = "arena.asset_source_extents-times-scale"
    if size[0] <= 0.0 or size[1] <= 0.0:
        raise SceneLayoutBlocked(
            "LAYOUT_SUPPORT_SIZE_INVALID",
            f"runtime support {fixture.get('name')!r} XY extent must be positive",
            {"size_xy": size},
        )
    return size, source


def _xy(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise SceneLayoutBlocked("LAYOUT_MEASUREMENT_MISSING", f"{label} must contain XY")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise SceneLayoutBlocked("LAYOUT_MEASUREMENT_INVALID", f"{label} must be numeric") from exc


def _yaw(value: Any, label: str) -> float:
    if value is None:
        return 0.0
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise SceneLayoutBlocked("LAYOUT_MEASUREMENT_INVALID", f"{label} must contain XYZ")
    try:
        return float(value[2])
    except (TypeError, ValueError) as exc:
        raise SceneLayoutBlocked("LAYOUT_MEASUREMENT_INVALID", f"{label}[2] must be numeric") from exc


def _perimeter_positions(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    anchor_xy: tuple[float, float] | None = None,
    span_fraction: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    x_values = _axis_values(bounds[0], None if anchor_xy is None else anchor_xy[0], span_fraction)
    y_values = _axis_values(bounds[1], None if anchor_xy is None else anchor_xy[1], span_fraction)
    low_x, mid_x, high_x = x_values
    low_y, mid_y, high_y = y_values
    return (
        (low_x, low_y),
        (low_x, mid_y),
        (low_x, high_y),
        (mid_x, low_y),
        (mid_x, high_y),
        (high_x, low_y),
        (high_x, mid_y),
        (high_x, high_y),
    )


def _axis_values(
    bounds: tuple[float, float],
    anchor: float | None,
    span_fraction: float,
) -> tuple[float, float, float]:
    minimum, maximum = bounds
    full_span = maximum - minimum
    if full_span <= 0.0 or not 0.0 < span_fraction <= 1.0:
        raise SceneLayoutBlocked(
            "LAYOUT_SEARCH_SPACE_DEGENERATE",
            "layout search interval must have a positive measurable span",
        )
    if anchor is None:
        low = minimum + full_span / 3.0
        high = maximum - full_span / 3.0
    else:
        window_span = full_span * span_fraction
        center = min(max(anchor, minimum), maximum)
        low = min(max(center - window_span / 2.0, minimum), maximum - window_span)
        high = low + window_span
    middle = (low + high) / 2.0
    if math.isclose(low, middle, abs_tol=1e-12) or math.isclose(
        middle, high, abs_tol=1e-12
    ):
        raise SceneLayoutBlocked(
            "LAYOUT_SEARCH_SPACE_DEGENERATE",
            "layout search interval cannot produce three distinct measured positions",
        )
    return tuple(_stable_float(value) for value in (low, middle, high))


def _world_to_local(
    world_xy: tuple[float, float],
    support: SupportSurfaceMeasurement,
) -> tuple[float, float]:
    dx = world_xy[0] - support.center_xy[0]
    dy = world_xy[1] - support.center_xy[1]
    yaw = math.radians(support.yaw_deg)
    return (
        dx * math.cos(yaw) + dy * math.sin(yaw),
        -dx * math.sin(yaw) + dy * math.cos(yaw),
    )


def _local_to_world(
    local_xy: tuple[float, float],
    support: SupportSurfaceMeasurement,
) -> tuple[float, float]:
    yaw = math.radians(support.yaw_deg)
    return (
        support.center_xy[0] + local_xy[0] * math.cos(yaw) - local_xy[1] * math.sin(yaw),
        support.center_xy[1] + local_xy[0] * math.sin(yaw) + local_xy[1] * math.cos(yaw),
    )


def _occupancy_bounds(
    placement: PlacementMeasurement,
) -> tuple[float, float, float, float]:
    yaw = math.radians(placement.actual_yaw_deg)
    width, depth = placement.size_xy
    half_x = (abs(math.cos(yaw)) * width + abs(math.sin(yaw)) * depth) / 2.0
    half_y = (abs(math.sin(yaw)) * width + abs(math.cos(yaw)) * depth) / 2.0
    jitter_x, jitter_y = placement.jitter_half_xy
    return (
        placement.center_xy[0] - half_x - jitter_x,
        placement.center_xy[1] - half_y - jitter_y,
        placement.center_xy[0] + half_x + jitter_x,
        placement.center_xy[1] + half_y + jitter_y,
    )


def _bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _constraint_failure(constraints: Mapping[str, bool]) -> tuple[str | None, str]:
    failures = {
        "schema": ("LAYOUT_SCHEMA_INVALID", "derived layout schema validation failed"),
        "support": ("LAYOUT_SUPPORT_CHANGED", "typed mutation changed the entity support"),
        "containment": (
            "LAYOUT_REGION_OUT_OF_SUPPORT",
            "entity placement envelope is not contained by its measured support footprint",
        ),
        "no_overlap": (
            "LAYOUT_REGION_OVERLAP",
            "entity placement envelope overlaps another region on the same support",
        ),
    }
    for name, passed in constraints.items():
        if not passed:
            return failures[name]
    return None, "static schema, support, containment, and overlap checks passed"


def _candidate_signature(revision: str, profile_id: str, failure_code: str) -> str:
    return hashlib.sha256(
        f"{revision}\0{profile_id}\0{failure_code}".encode("utf-8")
    ).hexdigest()


def _close_xy(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return math.isclose(first[0], second[0], abs_tol=1e-6) and math.isclose(
        first[1], second[1], abs_tol=1e-6
    )


def _stable_float(value: float) -> float:
    return float(f"{value:.12g}")
