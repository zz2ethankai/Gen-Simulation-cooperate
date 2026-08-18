"""Typed scene layout inspection, mutation, and search."""

from .evolution import (
    CandidateAggregate,
    CandidateEvaluation,
    CandidateGenome,
    EvolutionSearch,
    SearchConfig,
)
from .models import RegionSpec, SceneSpec, SupportGraph, SupportRelation
from .mutations import (
    MoveEntityOnSupport,
    RotateEntityOnSupport,
    SceneLayoutCompiler,
    SceneMutation,
    SceneMutationError,
    SetRobotPlacement,
    SetSupportHeight,
    scene_revision_for,
)
from .planner import (
    PlacementMeasurement,
    SceneLayoutBlocked,
    SceneLayoutPlanner,
    StaticLayoutValidation,
    SupportSurfaceMeasurement,
)
from .runtime_search import SceneLayoutSearchResult, run_scene_layout_search

__all__ = [
    "CandidateAggregate",
    "CandidateEvaluation",
    "CandidateGenome",
    "EvolutionSearch",
    "MoveEntityOnSupport",
    "PlacementMeasurement",
    "RegionSpec",
    "RotateEntityOnSupport",
    "SceneLayoutCompiler",
    "SceneLayoutBlocked",
    "SceneLayoutPlanner",
    "SceneLayoutSearchResult",
    "SceneMutation",
    "SceneMutationError",
    "SceneSpec",
    "SearchConfig",
    "SetRobotPlacement",
    "SetSupportHeight",
    "StaticLayoutValidation",
    "SupportGraph",
    "SupportRelation",
    "SupportSurfaceMeasurement",
    "scene_revision_for",
    "run_scene_layout_search",
]
