"""Stable public facade for the offline workspace planner.

Implementation is intentionally split under :mod:`core.workspace`; callers
should import this facade rather than depending on internal module layout.
"""

from ..workspace import (  # noqa: F401
    CuroboCandidateResult,
    GeometryCandidate,
    PickAttemptResult,
    SamplingConfig,
    WorkspaceManifest,
    WorkspacePlanningError,
    apply_candidate_to_document,
    audit_assets,
    build_manifest,
    compile_pick_task,
    compile_pick_place_probe_task,
    compile_probe_task,
    compile_existing_pose_probe_task,
    dump_json,
    dump_yaml,
    generate_manifest_file,
    load_yaml,
)
from ..robots.profile import (  # noqa: F401
    PlacementFamily,
    RobotModelProfile,
    RobotProfileError,
    load_robot_profile,
    load_robot_profile_for_task,
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
