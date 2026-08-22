import logging
import re
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from core.planning.direct_command import dummy_forward_params
from core.planning.motion_command import MotionPhase, MotionPhaseCommand

if TYPE_CHECKING:
    from core.controllers.curobo.skill_runtime import SkillRuntimePort

SKILL_DICT = {}
LOGGER = logging.getLogger("de_logger")


def register_skill(target_class):
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    # key = target_class.__name__
    # assert key not in SKILL_DICT
    SKILL_DICT[key] = target_class
    return target_class


class BaseSkill(ABC):
    def __init__(self):
        self.plan_flag = False
        self._target_visualizer = None
        self._target_visualization_context = {}
        self._target_visualization_handle = None
        self.skill_runtime = None

    def bind_skill_runtime(self, skill_runtime: "SkillRuntimePort"):
        """Bind the single generic runtime contract used by this Skill."""

        self.skill_runtime = skill_runtime
        return skill_runtime

    def _require_skill_runtime(self):
        runtime = self.skill_runtime
        if runtime is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a composed SkillRuntimePort"
            )
        return runtime

    def execute(self, command):
        """Execute a typed command through the narrow runtime port."""

        return self._require_skill_runtime().execute(command)

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        """Send a direct joint action through the controller compatibility API."""

        return self._require_skill_runtime().dummy_forward(
            arm_action, gripper_state, *args, **kwargs
        )

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
        """Read completion from either the typed or direct command boundary."""

        legacy_params = dummy_forward_params(command)
        if legacy_params is not None:
            runtime = self._require_skill_runtime()
            target = np.asarray(legacy_params["arm_action"], dtype=float).reshape(-1)
            state_getter = getattr(self.robot, "get_joints_state", None)
            if not callable(state_getter):
                return False
            positions = getattr(state_getter(), "positions", None)
            if hasattr(positions, "detach"):
                positions = positions.detach().cpu().numpy()
            if positions is None:
                return False
            current = np.asarray(positions, dtype=float)[runtime.arm_indices]
            tolerance = float(legacy_params.get("t_eps", 5e-3))
            return bool(np.linalg.norm(current - target) < tolerance)

        if not isinstance(command, MotionPhaseCommand):
            raise TypeError(
                f"{self.__class__.__name__} emits MotionPhaseCommand or dummy_forward commands"
            )
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
