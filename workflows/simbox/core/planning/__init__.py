"""Runtime motion-planning services shared by Skills and validation probes.

Heavy CuRobo-backed services remain lazily imported so the schema and state
contracts can be tested with ordinary USD Python outside Isaac Sim.
"""

from .domain_types import (
    AttachmentResult,
    AttachmentSpec,
    AttachmentState,
    BatchPlanResult,
    BatchPosePlanRequest,
    CollisionMode,
    CollisionOptions,
    CollisionPolicy,
    CommandStatus,
    CspacePlanRequest,
    JointTrajectory,
    PlanResult,
    PlannerKind,
    PlannerStatus,
    PlannerStatusSnapshot,
    PlanningProfile,
    PosePlanRequest,
)
from .planner_runtime import (
    PlannerCallError,
    PlannerDestroyedError,
    PlannerFactoryError,
    PlannerRuntime,
    PlannerRuntimeError,
    StaleSceneError,
)
from .planner_runtime import NativeCollisionOptions, map_collision_policy
from .attachment_runtime import AttachmentRuntime, AttachmentRuntimeError

__all__ = [
    "AttachmentResult",
    "AttachmentRuntime",
    "AttachmentRuntimeError",
    "AttachmentSpec",
    "AttachmentState",
    "BatchPlanResult",
    "BatchPosePlanRequest",
    "CollisionMode",
    "CollisionOptions",
    "CollisionPolicy",
    "CommandStatus",
    "CspacePlanRequest",
    "JointTrajectory",
    "PlanResult",
    "PlannerCallError",
    "PlannerDestroyedError",
    "PlannerFactoryError",
    "PlannerKind",
    "PlanningProfile",
    "PosePlanRequest",
    "PlannerRuntime",
    "PlannerRuntimeError",
    "PlannerStatus",
    "PlannerStatusSnapshot",
    "NativeCollisionOptions",
    "map_collision_policy",
    "StaleSceneError",
]


def __getattr__(name):
    if name in {"MotionPhase", "MotionPhaseCommand"}:
        # motion_command retains its NumPy dependency. Keep the domain/runtime
        # imports usable in a plain Python process by loading it only when its
        # symbols are requested.
        from .motion_command import MotionPhase, MotionPhaseCommand

        return {"MotionPhase": MotionPhase, "MotionPhaseCommand": MotionPhaseCommand}[name]
    raise AttributeError(name)
