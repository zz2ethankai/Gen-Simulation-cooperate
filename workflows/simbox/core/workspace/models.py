"""Data contracts shared by the offline workspace planner and runtime runner."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from ..robots.profile import RobotCollisionLayer


class WorkspacePlanningError(ValueError):
    """A deterministic workspace-input or geometry failure."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class SamplingConfig:
    planner: str = "target_annulus_v1"
    min_radius_m: float = 0.45
    max_radius_m: float = 1.05
    candidate_count: int = 96
    preferred_radius_m: float = 0.75
    sequence: str = "golden_angle"
    radial_count: int | None = None
    angular_count: int | None = None
    yaw_policy: str = "face_target"
    yaw_offsets_deg: tuple[float, ...] = (0.0,)

    def validate(self) -> None:
        if self.planner not in {"target_annulus_v1", "target_annulus_v2"}:
            raise WorkspacePlanningError("INVALID_PLANNER", f"unsupported workspace planner: {self.planner}")
        expected_sequence = "golden_angle" if self.planner == "target_annulus_v1" else "polar_grid"
        if self.sequence != expected_sequence:
            raise WorkspacePlanningError(
                "INVALID_SAMPLING_SEQUENCE",
                f"planner {self.planner} requires sequence {expected_sequence}, got {self.sequence}",
            )
        if not 0.0 < self.min_radius_m < self.max_radius_m:
            raise WorkspacePlanningError(
                "INVALID_RADIUS_RANGE",
                f"expected 0 < min_radius_m < max_radius_m, got {self.min_radius_m}, {self.max_radius_m}",
            )
        if self.candidate_count <= 0:
            raise WorkspacePlanningError("INVALID_CANDIDATE_COUNT", "candidate_count must be positive")
        if self.yaw_policy not in {"face_target", "align_required_arm"}:
            raise WorkspacePlanningError(
                "INVALID_YAW_POLICY", f"unsupported yaw policy: {self.yaw_policy}"
            )
        if not self.yaw_offsets_deg or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in self.yaw_offsets_deg
        ):
            raise WorkspacePlanningError(
                "INVALID_YAW_OFFSETS", "yaw_offsets_deg must contain finite values"
            )
        if len(set(float(value) for value in self.yaw_offsets_deg)) != len(
            self.yaw_offsets_deg
        ):
            raise WorkspacePlanningError(
                "INVALID_YAW_OFFSETS", "yaw_offsets_deg must be unique"
            )
        if self.planner == "target_annulus_v1" and tuple(self.yaw_offsets_deg) != (0.0,):
            raise WorkspacePlanningError(
                "INVALID_YAW_OFFSETS", "target_annulus_v1 supports only yaw_offsets_deg=[0]"
            )
        if self.planner == "target_annulus_v2":
            if not self.radial_count or not self.angular_count:
                raise WorkspacePlanningError(
                    "INVALID_POLAR_GRID",
                    "target_annulus_v2 requires positive radial_count and angular_count",
                )
            if self.radial_count < 2 or self.angular_count < 4:
                raise WorkspacePlanningError(
                    "INVALID_POLAR_GRID",
                    "polar grid requires radial_count >= 2 and angular_count >= 4",
                )
            expected_count = (
                self.radial_count * self.angular_count * len(self.yaw_offsets_deg)
            )
            if self.candidate_count != expected_count:
                raise WorkspacePlanningError(
                    "INVALID_CANDIDATE_COUNT",
                    "target_annulus_v2 candidate_count must equal radial_count * "
                    "angular_count * len(yaw_offsets_deg)",
                )
        if not self.min_radius_m <= self.preferred_radius_m <= self.max_radius_m:
            raise WorkspacePlanningError(
                "INVALID_PREFERRED_RADIUS",
                "preferred_radius_m must be inside the sampling annulus",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryCandidate:
    candidate_id: str
    world_xy: tuple[float, float]
    yaw_deg: float
    radius_m: float
    angle_deg: float
    yaw_offset_deg: float = 0.0
    collision_free: bool = False
    inside_floor: bool = False
    geometry_feasible: bool = False
    obstacle: str | None = None
    rejection_code: str | None = None
    mount_support: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["world_xy"] = list(self.world_xy)
        return value


@dataclass
class CuroboCandidateResult:
    candidate_id: str
    gpu: str
    return_code: int | None
    timed_out: bool
    results_complete: bool
    terminated_after_results: bool
    arms: dict[str, dict[str, Any]]
    feasible: bool
    selected_arm: str | None
    joint_success_count: int
    selected_grasp_score: float | None
    log: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PickAttemptResult:
    candidate_id: str
    arm: str
    seed: int
    gpu: str
    return_code: int | None
    timed_out: bool
    task_is_successful: bool
    event_success: bool
    episode_dir: str | None
    episode_name_valid: bool
    meta_info_created: bool
    lmdb_created: bool
    success: bool
    log: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkspaceManifest:
    source_task: str
    task_name: str
    target: dict[str, Any]
    support: dict[str, Any]
    sampling: dict[str, Any]
    robot: dict[str, Any]
    geometry_candidates: list[dict[str, Any]]
    required_arm: str | None = None
    version: int = 4
    curobo_results: list[dict[str, Any]] = field(default_factory=list)
    pick_attempts: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    status: str = "geometry_ready"
    failure_code: str | None = None
    asset_audit: list[dict[str, Any]] = field(default_factory=list)
    fixture_audit: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
