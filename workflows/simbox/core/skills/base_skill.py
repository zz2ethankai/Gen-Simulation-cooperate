import logging
import re
from abc import ABC
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from core.planning.motion_command import MotionPhase, MotionPhaseCommand

if TYPE_CHECKING:
    from core.controllers.pick_planning import PickPlanningPort
    from core.controllers.placement_planning import PlacementPlanningPort
    from core.controllers.skill_runtime import SkillRuntimePort

SKILL_DICT = {}
LOGGER = logging.getLogger("de_logger")


class SkillExecutorPort(Protocol):
    """Typed execution surface available to a manipulation Skill.

    ``SkillRuntimePort`` implements this protocol.  Keeping the protocol
    separate makes it possible for host-side tests to provide a tiny executor
    without constructing a simulator controller.
    """

    def execute(self, command: Any) -> Any: ...

    def execution_status(self, command: Any = None) -> Any: ...


@dataclass(frozen=True)
class SkillBindings:
    """The narrow ports a Skill may consume.

    This is deliberately not a controller/context object.  It carries only
    the immutable runtime view and the operation-specific planning ports.  A
    controller façade is never stored here (or on a Skill).
    """

    skill_runtime: "SkillRuntimePort"
    pick_planning: "PickPlanningPort | None" = None
    placement_planning: "PlacementPlanningPort | None" = None
    executor: SkillExecutorPort | None = None


def register_skill(target_class):
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    # key = target_class.__name__
    # assert key not in SKILL_DICT
    SKILL_DICT[key] = target_class
    return target_class


class BaseSkill(ABC):
    def __init__(self, bindings: SkillBindings | None = None):
        self.plan_flag = False
        self._target_visualizer = None
        self._target_visualization_context = {}
        self._target_visualization_handle = None
        self.skill_runtime = None
        self.pick_planning = None
        self.placement_planning = None
        self.executor = None
        if bindings is not None:
            self.bind_skill_runtime(
                bindings.skill_runtime,
                pick_planning=bindings.pick_planning,
                placement_planning=bindings.placement_planning,
                executor=bindings.executor,
            )

    def bind_skill_runtime(
        self,
        skill_runtime: "SkillRuntimePort",
        *,
        pick_planning: "PickPlanningPort | None" = None,
        placement_planning: "PlacementPlanningPort | None" = None,
        executor: SkillExecutorPort | None = None,
    ):
        """Bind explicit typed ports without retaining a controller façade."""

        self.skill_runtime = skill_runtime
        self.pick_planning = pick_planning
        self.placement_planning = placement_planning
        self.executor = executor if executor is not None else skill_runtime
        return skill_runtime

    def _require_skill_runtime(self):
        runtime = self.skill_runtime
        if runtime is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a composed SkillRuntimePort"
            )
        return runtime

    def _require_pick_planning(self):
        planning = self.pick_planning
        if planning is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a composed PickPlanningPort"
            )
        return planning

    def _require_placement_planning(self):
        planning = self.placement_planning
        if planning is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a composed PlacementPlanningPort"
            )
        return planning

    def execute(self, command):
        """Execute a typed command through the narrow runtime port."""

        return self._require_skill_runtime().execute(command)

    def bind_target_visualizer(self, visualizer, **context):
        """Bind optional observational target rendering without changing Skill APIs."""
        self._target_visualizer = visualizer
        self._target_visualization_context = dict(context)

    def publish_target_intent(self, descriptor: dict):
        if self._target_visualizer is None:
            return None
        try:
            self._target_visualization_handle = self._target_visualizer.record_target(
                self, descriptor
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to publish skill=%s",
                self.__class__.__name__,
            )
        return self._target_visualization_handle

    def complete_target_intent(self, success: bool):
        if self._target_visualizer is None:
            return
        try:
            self._target_visualizer.finish_target(
                self._target_visualization_handle, bool(success)
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to complete skill=%s",
                self.__class__.__name__,
            )

    def abort_target_intent(self, reason: str):
        if self._target_visualizer is None:
            return
        try:
            self._target_visualizer.finish_target(
                self._target_visualization_handle,
                False,
                reason=str(reason),
            )
        except Exception:
            LOGGER.exception(
                "[SkillTargetDebug] failed to abort skill=%s",
                self.__class__.__name__,
            )

    def is_ready(self):
        return True

    def is_done(self):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError

    def update(self):
        pass

    def is_feasible(self):
        return True

    def is_record(self):
        return True

    # ------------------------------------------------------------------
    # Typed command helpers
    # ------------------------------------------------------------------
    # Skills used to return positional tuples whose third item was reflected
    # into a controller method.  Keep the small amount of construction and
    # completion plumbing here so every Skill has the same typed boundary.
    @staticmethod
    def pose_command(
        phase,
        position,
        orientation,
        *,
        gripper_action=None,
        active_object=None,
        support_object=None,
        **kwargs,
    ):
        return MotionPhaseCommand(
            phase=phase,
            target_position=np.asarray(position, dtype=float),
            target_orientation=np.asarray(orientation, dtype=float),
            gripper_action=gripper_action,
            active_object=active_object,
            support_object=support_object,
            **kwargs,
        )

    @staticmethod
    def joint_command(
        joints,
        *,
        gripper_action=None,
        phase=MotionPhase.CARRY_HOME,
        direct=False,
        **kwargs,
    ):
        params = dict(kwargs.pop("params", {}) or {})
        if direct:
            if not params.get("passthrough", False):
                raise ValueError(
                    "direct joint commands require explicit passthrough/hold metadata"
                )
            params["direct_joint_action"] = np.asarray(joints, dtype=float).copy()
            return MotionPhaseCommand(
                phase=phase,
                gripper_action=gripper_action,
                params=params,
                **kwargs,
            )
        return MotionPhaseCommand(
            phase=phase,
            gripper_action=gripper_action,
            joint_target=np.asarray(joints, dtype=float).copy(),
            params=params,
            **kwargs,
        )

    def measured_hold_command(self, *, gripper_action=None, **kwargs):
        """Return a typed measured-state hold used by passthrough Skills.

        The command takes the arm joints from the articulation and contains no
        planner/world operation.  This is the one measured-state hold boundary
        allowed to carry an explicit direct payload; ordinary arm targets use
        ``joint_target`` and native c-space planning.
        """

        state_getter = getattr(self.robot, "get_joints_state", None)
        if callable(state_getter):
            state = state_getter()
            positions = getattr(state, "positions", state)
        else:
            positions = self.robot.get_joint_positions()
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float)
        runtime = self._require_skill_runtime()
        indices = np.asarray(runtime.arm_indices, dtype=int)
        if indices.size:
            positions = positions[indices]
        phase = kwargs.pop("phase", MotionPhase.CARRY_HOME)
        kwargs.setdefault("collision_policy", "passthrough")
        return self.joint_command(
            positions,
            gripper_action=gripper_action,
            phase=phase,
            replan_allowed=False,
            params={"passthrough": True},
            direct=True,
            **kwargs,
        )

    def command_complete(self, command):
        """Read completion from the typed controller status boundary."""

        if not isinstance(command, MotionPhaseCommand):
            raise TypeError(f"{self.__class__.__name__} emits typed motion commands only")
        runtime = self._require_skill_runtime()
        status = runtime.execution_status(command)
        if isinstance(status, dict):
            return bool(status.get("complete", False))
        if hasattr(status, "complete"):
            return bool(status.complete)
        direct = command.params.get("direct_joint_action")
        if direct is not None:
            state_getter = getattr(self.robot, "get_joints_state", None)
            if callable(state_getter):
                positions = getattr(state_getter(), "positions", None)
                if hasattr(positions, "detach"):
                    positions = positions.detach().cpu().numpy()
                indices = np.asarray(runtime.arm_indices, dtype=int)
                if positions is not None and indices.size:
                    return bool(np.linalg.norm(np.asarray(positions)[indices] - direct) < 5e-3)
        if command.target_position is not None:
            position, orientation = runtime.ee_pose()
            position_ok = np.linalg.norm(np.asarray(position) - command.target_position) <= command.translation_tolerance
            orientation_ok = 2 * np.arccos(
                np.clip(abs(np.dot(np.asarray(orientation), command.target_orientation)), 0.0, 1.0)
            ) <= command.orientation_tolerance
            return bool(position_ok and orientation_ok)
        return False

    @staticmethod
    def command_status_debug(status):
        """Serialize the public typed command status for Skill diagnostics."""

        if status is None:
            return None
        if isinstance(status, dict):
            return dict(status)
        enum_status = getattr(getattr(status, "status", None), "value", None)
        if not hasattr(status, "phase"):
            return {
                "status": enum_status
                or getattr(status, "value", status),
            }
        return {
            "status": enum_status,
            "phase": status.phase,
            "complete": bool(status.complete),
            "plan_active": bool(status.plan_active),
            "plan_failed": bool(status.plan_failed),
            "tracking_failed": bool(getattr(status, "tracking_failed", False)),
            "plan_steps_remaining": int(status.plan_steps_remaining),
            "reason": status.reason,
        }

    def pop_completed_command(self):
        if self.manip_list and self.command_complete(self.manip_list[0]):
            self.manip_list.pop(0)
        return not self.manip_list
