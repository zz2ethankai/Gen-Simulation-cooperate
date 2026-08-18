"""Offline workspace candidate generation.

This package must stay importable without Isaac Sim or CuRobo.  Runtime grasp
validation lives under ``core.planning`` and consumes the manifest emitted here.
"""

from ..robots.profile import (
    PlacementFamily,
    RobotModelProfile,
    RobotProfileError,
    load_robot_profile,
    load_robot_profile_for_task,
)

from .models import (
    CuroboCandidateResult,
    GeometryCandidate,
    PickAttemptResult,
    SamplingConfig,
    WorkspaceManifest,
    WorkspacePlanningError,
)
from .planner import (
    apply_candidate_to_document,
    apply_support_mounted_candidate_to_document,
    audit_assets,
    build_manifest,
    dump_json,
    dump_yaml,
    generate_manifest_file,
    load_yaml,
)
from .task_compiler import (
    compile_existing_pose_probe_task,
    compile_pick_place_probe_task,
    compile_pick_task,
    compile_probe_task,
)

__all__ = [
    "CuroboCandidateResult",
    "GeometryCandidate",
    "PickAttemptResult",
    "PlacementFamily",
    "RobotModelProfile",
    "RobotProfileError",
    "SamplingConfig",
    "WorkspaceManifest",
    "WorkspacePlanningError",
    "apply_candidate_to_document",
    "apply_support_mounted_candidate_to_document",
    "audit_assets",
    "build_manifest",
    "compile_pick_task",
    "compile_pick_place_probe_task",
    "compile_probe_task",
    "compile_existing_pose_probe_task",
    "dump_json",
    "dump_yaml",
    "generate_manifest_file",
    "load_yaml",
    "load_robot_profile",
    "load_robot_profile_for_task",
]
