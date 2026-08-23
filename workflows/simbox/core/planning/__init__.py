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
    CommandType,
    CspacePlanRequest,
    GripperCommand,
    HoldCommand,
    JointCommand,
    JointTrajectory,
    PlanResult,
    PlannerCommand,
    PlannerKind,
    PlannerOperation,
    PlannerStatus,
    PlannerStatusSnapshot,
    PlanningProfile,
    PoseCommand,
    PosePlanRequest,
    SceneCommand,
    SceneRevision,
    SceneUpdate,
)
from .planner_runtime import (
    PlannerCallError,
    PlannerDestroyedError,
    PlannerFactoryError,
    PlannerRuntime,
    PlannerRuntimeError,
    StaleSceneError,
)
from .scene_runtime import (
    SceneFanoutError,
    SceneRuntime,
    SceneRuntimeError,
    SceneSubscription,
)
from .native_scene_adapter import NativeSceneAdapter, NativeSceneAdapterError, SceneFanoutAdapter
from .native_planner_adapter import (
    NativeCollisionOptions,
    NativeCollisionPolicyError,
    NativePlannerAdapter,
    NativePlannerAdapterError,
    map_collision_policy,
)
from .attachment_runtime import (
    AttachmentRollbackError,
    AttachmentRuntime,
    AttachmentRuntimeError,
    AttachmentSnapshot,
    AttachmentTransaction,
)

__all__ = [
    "AttachmentResult",
    "AttachmentRollbackError",
    "AttachmentRuntime",
    "AttachmentRuntimeError",
    "AttachmentSnapshot",
    "AttachmentSpec",
    "AttachmentState",
    "AttachmentTransaction",
    "CollisionMode",
    "CollisionOptions",
    "CollisionPolicy",
    "CommandStatus",
    "CommandType",
    "CspacePlanRequest",
    "BatchPlanResult",
    "BatchPosePlanRequest",
    "GripperCommand",
    "HoldCommand",
    "JointCommand",
    "JointTrajectory",
    "PlanResult",
    "PlannerCallError",
    "PlannerCommand",
    "PlannerDestroyedError",
    "PlannerFactoryError",
    "PlannerKind",
    "PlannerOperation",
    "PlanningProfile",
    "PoseCommand",
    "PosePlanRequest",
    "SceneCommand",
    "PlannerRuntime",
    "PlannerRuntimeError",
    "PlannerStatus",
    "PlannerStatusSnapshot",
    "NativeSceneAdapter",
    "NativeSceneAdapterError",
    "SceneFanoutAdapter",
    "NativeCollisionOptions",
    "NativeCollisionPolicyError",
    "NativePlannerAdapter",
    "NativePlannerAdapterError",
    "map_collision_policy",
    "SceneFanoutError",
    "SceneRevision",
    "SceneRuntime",
    "SceneRuntimeError",
    "SceneSubscription",
    "SceneUpdate",
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
