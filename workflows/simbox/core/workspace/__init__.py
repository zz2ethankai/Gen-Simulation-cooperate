"""Offline workspace candidate generation.

This package must stay importable without Isaac Sim or CuRobo.  Runtime grasp
validation lives under ``core.planning`` and consumes the manifest emitted here.
"""

from .models import (
    CuroboCandidateResult,
    DEFAULT_ROBOT_PROFILES,
    GeometryCandidate,
    PickAttemptResult,
    RobotCollisionLayer,
    RobotProfile,
    SamplingConfig,
    WorkspaceManifest,
    WorkspacePlanningError,
)
from .planner import (
    apply_candidate_to_document,
    apply_tabletop_candidate_to_document,
    audit_assets,
    build_manifest,
    dump_json,
    dump_yaml,
    generate_manifest_file,
    load_yaml,
)
from .task_compiler import compile_existing_pose_probe_task, compile_pick_task, compile_probe_task

__all__ = [
    "CuroboCandidateResult",
    "DEFAULT_ROBOT_PROFILES",
    "GeometryCandidate",
    "PickAttemptResult",
    "RobotCollisionLayer",
    "RobotProfile",
    "SamplingConfig",
    "WorkspaceManifest",
    "WorkspacePlanningError",
    "apply_candidate_to_document",
    "apply_tabletop_candidate_to_document",
    "audit_assets",
    "build_manifest",
    "compile_pick_task",
    "compile_probe_task",
    "compile_existing_pose_probe_task",
    "dump_json",
    "dump_yaml",
    "generate_manifest_file",
    "load_yaml",
]
