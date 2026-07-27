"""Stable public facade for the offline workspace planner.

Implementation is intentionally split under :mod:`core.workspace`; callers
should import this facade rather than depending on internal module layout.
"""

from core.workspace import (  # noqa: F401
    CuroboCandidateResult,
    DEFAULT_ROBOT_PROFILES,
    GeometryCandidate,
    PickAttemptResult,
    RobotProfile,
    SamplingConfig,
    WorkspaceManifest,
    WorkspacePlanningError,
    apply_candidate_to_document,
    audit_assets,
    build_manifest,
    compile_pick_task,
    compile_probe_task,
    compile_existing_pose_probe_task,
    dump_json,
    dump_yaml,
    generate_manifest_file,
    load_yaml,
)

__all__ = [
    "CuroboCandidateResult",
    "DEFAULT_ROBOT_PROFILES",
    "GeometryCandidate",
    "PickAttemptResult",
    "RobotProfile",
    "SamplingConfig",
    "WorkspaceManifest",
    "WorkspacePlanningError",
    "apply_candidate_to_document",
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
