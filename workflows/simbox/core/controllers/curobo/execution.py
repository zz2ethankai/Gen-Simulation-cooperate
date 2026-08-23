"""Typed command execution and articulation feedback operations."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from core.controllers.curobo.phase_execution import ExecutionStatus
from core.controllers.curobo.trajectory import execution_trajectory_tensor
from core.planning.domain_types import CollisionPolicy, CommandStatus
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.controllers.curobo.components import ComponentState
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.xforms import get_world_pose

LOGGER = logging.getLogger("de_logger")

# Heavy simulator/planner types are injected by the runtime at call time.  The
# placeholders also let narrow component tests bind light fakes without
# importing the complete Isaac/CuRobo stack.
JointState = None
ArticulationAction = None


class ControllerExecution(ComponentState):
    def _begin_phase_command(self, command: MotionPhaseCommand) -> bool:
        """Reset execution bookkeeping when a new typed command starts."""

        if command is self._active_phase_command:
            return False
        self._active_phase_command = command
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_bookkeeping_done = False
        self._phase_dwell_count = 0
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._phase_completion_logged = False
        self._last_command_name = command.phase.value
        self._phase_base_position, self._phase_base_orientation = self.get_armbase_pose()
        LOGGER.info(
            "[PhaseDebug] start robot=%s arm=%s phase=%s object=%s support=%s target=%s",
            self.name,
            self.lr_name,
            command.phase.value,
            command.active_object,
            command.support_object,
            None
            if command.target_position is None
            else np.asarray(command.target_position, dtype=float).round(5).tolist(),
        )
        return True

    def _install_preplanned_phase_path(self, command: MotionPhaseCommand):
        """Install a named trajectory already validated by Pick/Place."""

        preplanned_path = command.params.get("preplanned_joint_path")
        if preplanned_path is None:
            return False
        trajectory = self.runtime._command_path(preplanned_path)
        self.runtime._install_command_plan(
            trajectory,
            target_position=command.target_position,
            target_orientation=command.target_orientation,
            phase_name=command.phase.value,
            cached=True,
        )
        return True

    def _apply_gripper_action(self, action: Any) -> None:
        """Apply one of the finite gripper actions carried by a phase command."""

        if action in (None, ""):
            return
        handlers = {
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }
        try:
            handler = handlers[action]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unsupported typed gripper action: {action!r}") from exc
        handler()

    def _apply_gripper_state(self, state: float | None) -> None:
        """Apply the numeric gripper contract carried by direct commands."""

        if state is None:
            return
        if float(state) == 1.0:
            self.open_gripper()
        elif float(state) == -1.0:
            self.close_gripper()
        else:  # MotionPhaseCommand validates this; keep the execution guard too.
            raise ValueError("direct gripper_state must be exactly 1.0 or -1.0")

    def dummy_forward(self, arm_action, gripper_state):
        """Return one direct articulation action without invoking the planner.

        This is the execution boundary used by interpolation Skills.  The
        caller owns interpolation and completion; this method only applies the
        requested arm vector and gripper state.
        """

        arm_action = np.asarray(arm_action, dtype=float).copy()
        clear = getattr(self.phase_executor, "clear", None)
        if callable(clear):
            clear()
        self._active_phase_command = None
        self._phase_bookkeeping_done = True
        self._phase_plan_started = False
        self._phase_plan_finished = True
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._last_command_name = "dummy_forward"
        self._last_arm_action = arm_action.copy()
        self._last_commanded_arm_position = arm_action.copy()
        self.num_last_cmd += 1

        gripper_state = float(gripper_state)
        if gripper_state == 1.0:
            self.open_gripper()
        elif gripper_state == -1.0:
            self.close_gripper()
        else:
            raise NotImplementedError(
                "dummy_forward gripper_state must be exactly 1.0 or -1.0"
            )
        return self._make_action(arm_action, self.get_gripper_action())

    def forward_phase_command(self, command: MotionPhaseCommand):
        """Execute one structured motion phase.

        The normal phases are planner-backed and use the exact collision
        scene manager.  A typed ``joint_target`` follows the same phase
        executor through a native c-space request.  The separate
        ``dummy_forward`` method is the explicit execution-only boundary for
        Skills such as Home that own a direct interpolation; this typed phase
        method still reserves ``direct_joint_action`` for measured-state
        passthrough holds.
        """

        first_step = self._begin_phase_command(command)
        direct_joint_action = command.direct_joint_action
        if direct_joint_action is not None:
            if command.collision_policy != CollisionPolicy.PASSTHROUGH:
                raise ValueError(
                    "direct_joint_action is reserved for CollisionPolicy.PASSTHROUGH"
                )
            if first_step:
                self._phase_bookkeeping_done = True
                self._phase_plan_finished = True
            self._apply_gripper_action(command.gripper_action)
            self._apply_gripper_state(command.gripper_state)
            self._last_command_name = command.phase.value
            self._last_arm_action = np.asarray(direct_joint_action, dtype=float).copy()
            self._last_commanded_arm_position = self._last_arm_action.copy()
            return self._make_action(
                self._last_arm_action,
                self.get_gripper_action(),
            )

        if command.joint_target is not None:
            return self._forward_joint_target(command, first_step)

        manager = self.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPhaseCommand requires CollisionSceneManager")
        robot, arm = self.name, self.lr_name
        if first_step:
            if command.phase == MotionPhase.SYNC_WORLD:
                manager.sync_dynamic_poses(self._step_idx, interval_steps=1, force=True)
                if command.active_object:
                    manager.begin_target_transit(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TRANSIT_PREGRASP:
                manager.begin_target_transit(command.active_object, robot, arm)
                if command.params.get("preplanned_joint_path") is not None:
                    self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.TERMINAL_GRASP_APPROACH:
                manager.begin_target_approach(command.active_object, robot, arm)
                preplanned_path = command.params.get("preplanned_joint_path")
                if preplanned_path is not None:
                    self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.TRANSIT_PREPLACE:
                manager.assert_attached_owner(command.active_object, robot, arm)
                self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.ATTACH:
                verify_contact = command.params.get("verify_grasp_contact")
                if not callable(verify_contact) or not bool(verify_contact()):
                    raise RuntimeError(
                        "ATTACH requires a verified target-finger contact from GRIPPER_CLOSE"
                    )
                manager.attach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.CARRY_HOME:
                manager.assert_attached_owner(command.active_object, robot, arm)
                preplanned_path = command.params.get("preplanned_joint_path")
                if preplanned_path is None:
                    raise RuntimeError("CARRY_HOME requires a preplanned joint path")
                self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.TERMINAL_PLACE_DESCENT:
                manager.begin_placement_descent(
                    command.active_object, command.support_object, robot, arm
                )
                if command.params.get("preplanned_joint_path") is not None:
                    cached_plan = self.runtime._command_path(
                        command.params["preplanned_joint_path"]
                    )
                    if command.params.get("continuous_descent", False):
                        cached_valid = self._validate_continuous_place_plan(
                            command, cached_plan
                        )
                    else:
                        cached_valid = True
                    if cached_valid:
                        self._install_preplanned_phase_path(command)
                    else:
                        # A cached path can be invalidated by a safety hold or
                        # a small attached-object slip between evaluation and
                        # execution.  Fall back to a fresh single-plan query
                        # from the measured state instead of failing the phase
                        # on stale candidate data.
                        command.params.pop("preplanned_joint_path", None)
                        LOGGER.warning(
                            "[PhaseDebug] cached-place-plan rejected robot=%s arm=%s; "
                            "falling back to native-v2 replanning",
                            self.name,
                            self.lr_name,
                        )
            elif command.phase == MotionPhase.DETACH_AND_SETTLE:
                manager.detach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TERMINAL_RETREAT:
                manager.begin_terminal_retreat(command.active_object, robot, arm)
            elif command.phase == MotionPhase.RESTORE_WORLD:
                manager.restore_world(command.active_object)
                self._phase_bookkeeping_done = True

        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            return self.hold_action()

        self._apply_gripper_action(command.gripper_action)

        if command.is_bookkeeping or command.phase in {
            MotionPhase.GRIPPER_CLOSE,
            MotionPhase.GRIPPER_OPEN,
        }:
            self._phase_dwell_count += 1
            position, orientation = self.get_ee_pose()
            return self.ee_forward(position, orientation, skip_plan=True)
        plan_validator = None
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("continuous_descent", False)
        ):
            plan_validator = lambda plan: self._validate_continuous_place_plan(
                command, plan
            )
        return self.ee_forward(
            command.target_position,
            command.target_orientation,
            # Planning-target identity and physical completion are different
            # contracts. Passing completion tolerance here can suppress a
            # needed plan while the measured EE is still outside completion,
            # causing an infinite hold. Keep target-change detection tight;
            # ``is_phase_command_complete`` applies physical tolerance.
            eps=command.planning_epsilon,
            plan_validator=plan_validator,
        )

    def _forward_joint_target(self, command: MotionPhaseCommand, first_step: bool):
        """Plan and execute one typed joint target through native c-space."""

        if first_step:
            manager = self.collision_scene_manager
            if command.active_object is not None:
                if manager is None:
                    raise RuntimeError(
                        "attached joint command requires CollisionSceneManager"
                    )
                manager.assert_attached_owner(
                    command.active_object, self.name, self.lr_name
                )

            cached_path = command.params.get("preplanned_joint_path")
            if cached_path is not None:
                trajectory = self.runtime._command_path(cached_path)
                self.runtime._install_command_plan(
                    trajectory,
                    target_position=command.target_position,
                    target_orientation=command.target_orientation,
                    phase_name=command.phase.value,
                    cached=True,
                )
            else:
                request_metadata = command.planning_request_metadata
                result = self.runtime.plan_cspace(
                    command.joint_target,
                    context=command.phase.value,
                    request_metadata=request_metadata,
                )
                self.runtime._log_plan_result(
                    command.phase.value,
                    result,
                    target=np.asarray(command.joint_target, dtype=float),
                )
                if self.runtime._result_success(result):
                    raw_plan = self.runtime._result_path(result)
                    trajectory = self.runtime._command_path(raw_plan)
                    if trajectory is not None:
                        self.runtime._install_command_plan(
                            trajectory,
                            target_position=command.target_position,
                            target_orientation=command.target_orientation,
                            phase_name=command.phase.value,
                            cached=False,
                        )
                        self.num_plan_failed = 0
                    else:
                        self._phase_plan_failed = True
                else:
                    self._phase_plan_failed = True

                if self._phase_plan_failed:
                    self.num_plan_failed += 1
                    LOGGER.warning(
                        "[PlanDebug] cspace plan failed robot=%s arm=%s command=%s num_plan_failed=%d",
                        self.name,
                        self.lr_name,
                        command.phase.value,
                        self.num_plan_failed,
                    )

        self._apply_gripper_action(command.gripper_action)
        return self._forward_installed_joint_path()

    def _trajectory_tensor(self, trajectory, *, context="controller execution trajectory"):
        """Materialize a public trajectory at the Isaac action boundary."""

        target_names = getattr(self, "raw_js_names", None)
        if target_names:
            return execution_trajectory_tensor(
                trajectory,
                self.tensor_args,
                target_joint_names=target_names,
                context=context,
            )
        return execution_trajectory_tensor(
            trajectory,
            self.tensor_args,
            context=context,
        )

    @staticmethod
    def _host_array(value):
        """Convert a boundary tensor/list value to one host NumPy array."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)

    def _forward_installed_joint_path(self):
        """Consume the currently installed typed c-space trajectory."""

        sim_js = self.robot.get_joints_state()
        current_trajectory = self.phase_executor.current
        current_index = self.phase_executor.index
        if (
            current_trajectory is not None
            and current_index < len(current_trajectory)
            and self._step_idx % 1 == 0
        ):
            positions, _joint_names = self._trajectory_tensor(current_trajectory)
            arm_action = self._host_array(positions[current_index]).copy()
            self._last_arm_action = arm_action
            next_state = self.phase_executor.advance(self.ds_ratio)
            if next_state is None:
                self.phase_executor.clear()
                self._phase_plan_finished = True
        else:
            self.num_last_cmd += 1
            if self._last_arm_action is None:
                arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
            else:
                arm_action = np.asarray(self._last_arm_action, dtype=float).copy()

        self._step_idx += 1
        self._last_commanded_arm_position = np.asarray(arm_action, dtype=float).copy()
        return self._make_action(arm_action, self.get_gripper_action())

    def _validate_continuous_place_plan(self, command: MotionPhaseCommand, plan) -> bool:
        """Reject a fast descent that is indirect or advances too far per frame."""

        start_position, _ = self.get_ee_pose()
        plan_position, plan_joint_names = self._trajectory_tensor(
            plan,
            context="continuous place trajectory",
        )
        if len(plan_position):
            batch_fk = getattr(self.runtime, "_compute_cartesian_fk_batch", None)
            if not callable(batch_fk):
                raise RuntimeError(
                    "continuous-place validation requires the formal batched FK API"
                )
            positions_from_batch = batch_fk(
                plan_position, joint_names=list(plan_joint_names)
            )
            positions = np.asarray(positions_from_batch, dtype=float)
        else:
            positions = np.asarray([], dtype=float)
        if not len(positions) or not np.all(np.isfinite(positions)):
            LOGGER.warning(
                "[PhaseDebug] continuous-place-plan-invalid robot=%s arm=%s reason=non_finite_path",
                self.name,
                self.lr_name,
            )
            return False

        direct_vector = np.asarray(command.target_position, dtype=float) - start_position
        direct_length = float(np.linalg.norm(direct_vector))
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        if direct_length <= 1e-9:
            path_length_ratio = 1.0 if path_length <= 1e-9 else float("inf")
            max_deviation = path_length
        else:
            direction = direct_vector / direct_length
            relative = positions - start_position
            projection = np.clip(relative @ direction, 0.0, direct_length)
            closest = start_position + projection[:, None] * direct_vector / direct_length
            path_length_ratio = path_length / direct_length
            max_deviation = float(
                np.max(np.linalg.norm(positions - closest, axis=1))
            )

        executed = positions[:: self.ds_ratio]
        executed_with_start = np.concatenate(
            [np.asarray(start_position, dtype=float).reshape(1, 3), executed], axis=0
        )
        max_step = float(
            np.max(np.linalg.norm(np.diff(executed_with_start, axis=0), axis=1))
        )
        max_allowed_step = float(command.params["max_cartesian_step_m"])
        max_allowed_ratio = float(command.params["max_path_length_ratio"])
        max_allowed_deviation = float(command.params["max_path_deviation_m"])
        valid_limits = (
            np.isfinite(max_allowed_step)
            and max_allowed_step > 0.0
            and np.isfinite(max_allowed_ratio)
            and max_allowed_ratio >= 1.0
            and np.isfinite(max_allowed_deviation)
            and max_allowed_deviation >= 0.0
        )
        valid = bool(
            valid_limits
            and max_step <= max_allowed_step + 1e-6
            and path_length_ratio <= max_allowed_ratio + 1e-6
            and max_deviation <= max_allowed_deviation + 1e-6
        )
        log = LOGGER.info if valid else LOGGER.warning
        log(
            "[PhaseDebug] continuous-place-plan robot=%s arm=%s valid=%s waypoints=%d "
            "stride=%d max_step=%.6f/%.6f path_ratio=%.4f/%.4f "
            "max_deviation=%.6f/%.6f",
            self.name,
            self.lr_name,
            valid,
            len(positions),
            self.ds_ratio,
            max_step,
            max_allowed_step,
            path_length_ratio,
            max_allowed_ratio,
            max_deviation,
            max_allowed_deviation,
        )
        return valid

    def complete_terminal_place_on_contact(self, command: MotionPhaseCommand) -> None:
        """Cancel the remaining descent without resetting phase completion state."""

        if command is not self._active_phase_command:
            return
        self.phase_executor.clear()
        self._phase_plan_finished = True
        self._last_arm_action = None
        if not command.params.get("contact_stop_logged", False):
            LOGGER.info(
                "[PhaseDebug] contact-stop robot=%s arm=%s phase=%s",
                self.name,
                self.lr_name,
                command.phase.value,
            )
            command.params["contact_stop_logged"] = True

    def is_phase_command_complete(self, command: MotionPhaseCommand) -> bool:
        if command is not self._active_phase_command:
            return False
        if command.joint_target is not None:
            return bool(self._phase_plan_finished and self.phase_executor.current is None)
        if command.direct_joint_action is not None:
            return bool(self._phase_plan_finished)
        if command.is_bookkeeping:
            if (
                command.phase == MotionPhase.DETACH_AND_SETTLE
                and self._phase_dwell_count >= max(1, command.dwell_steps)
                and not command.params.get("settle_finalized", False)
            ):
                self.collision_scene_manager.finalize_detach_target(
                    command.active_object, self.name, self.lr_name
                )
                command.params["settle_finalized"] = True
            return self._phase_bookkeeping_done and self._phase_dwell_count >= max(1, command.dwell_steps)
        if command.phase in {MotionPhase.GRIPPER_CLOSE, MotionPhase.GRIPPER_OPEN}:
            return self._phase_dwell_count >= max(1, command.dwell_steps)
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            self.phase_executor.clear()
            self._phase_plan_finished = True
            return True
        position, orientation = self.get_ee_pose()
        position_error = float(np.linalg.norm(position - command.target_position))
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(abs(np.dot(orientation, command.target_orientation)), 0.0, 1.0)
            )
        )
        inside = (
            position_error <= command.translation_tolerance
            and orientation_error <= command.orientation_tolerance
        )
        if inside and self.phase_executor.current is None:
            self._phase_plan_finished = True
            if not self._phase_completion_logged:
                LOGGER.info(
                    "[PhaseDebug] complete robot=%s arm=%s phase=%s position_error=%.6f orientation_error=%.6f",
                    self.name,
                    self.lr_name,
                    command.phase.value,
                    position_error,
                    orientation_error,
                )
                self._phase_completion_logged = True
            return True
        if self._phase_plan_finished and not inside:
            self._phase_tracking_failed = True
            if not self._phase_completion_logged:
                LOGGER.warning(
                    "[PhaseDebug] tracking-failed robot=%s arm=%s phase=%s position_error=%.6f orientation_error=%.6f",
                    self.name,
                    self.lr_name,
                    command.phase.value,
                    position_error,
                    orientation_error,
                )
                self._phase_completion_logged = True
        return False

    def execution_status(self, command=None) -> ExecutionStatus:
        """Return detailed execution state for diagnostics and safety logic."""

        active = command is None or command is self._active_phase_command
        current = command if command is not None else self._active_phase_command
        executor_status = self.phase_executor.status()
        phase = current.phase.value if current is not None else None
        complete = bool(
            active
            and command is not None
            and self.is_phase_command_complete(command)
        )
        plan_failed = bool(self._phase_plan_failed)
        tracking_failed = bool(self._phase_tracking_failed)
        scene_failed = bool(
            getattr(getattr(self, "setup", None), "_world_cleanup_failed", False)
        )
        reason = None
        if scene_failed:
            reason = "scene_failed"
            status = CommandStatus.SCENE_FAILED
        elif plan_failed:
            reason = "plan_failed"
            status = CommandStatus.PLAN_FAILED
        elif tracking_failed:
            reason = "tracking_failed"
            status = CommandStatus.TRACKING_FAILED
        elif complete:
            status = CommandStatus.COMPLETED
        elif current is not None or executor_status.plan_active:
            status = CommandStatus.ACTIVE
        else:
            status = CommandStatus.IDLE
        return ExecutionStatus(
            status=status,
            phase=phase,
            complete=complete,
            plan_active=executor_status.plan_active,
            plan_failed=plan_failed,
            tracking_failed=tracking_failed,
            plan_steps_remaining=int(executor_status.plan_steps_remaining),
            reason=reason,
            plan_id=getattr(current, "plan_id", None),
            replan_allowed=bool(getattr(current, "replan_allowed", True)),
            active=bool(self._active_phase_command is not None and active),
            last_commanded_arm_position=(
                None
                if self._last_commanded_arm_position is None
                else np.asarray(self._last_commanded_arm_position, dtype=float).copy()
            ),
            phase_base_position=(
                None
                if self._phase_base_position is None
                else np.asarray(self._phase_base_position, dtype=float).copy()
            ),
            phase_base_orientation=(
                None
                if self._phase_base_orientation is None
                else np.asarray(self._phase_base_orientation, dtype=float).copy()
            ),
        )

    def clear_plan_and_hold(self) -> None:
        """Stop consuming the old plan; the next action holds measured joints."""

        self.phase_executor.clear()
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._active_phase_command = None
        position, orientation = self.get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(position)
        self._ee_ori = self.tensor_args.to_device(orientation)

    def _make_action(self, arm_action, gripper_action):
        """Build the common simulator action payload for every hold path."""

        arm_action = np.asarray(arm_action, dtype=float).copy()
        gripper_action = np.asarray(gripper_action, dtype=float).copy()
        joint_indices = np.concatenate([self.arm_indices, self.gripper_indices])
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": joint_indices,
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }

    def hold_action(self):
        """Return an articulation target equal to the measured current joints."""

        sim_js = self.robot.get_joints_state()
        arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
        return self._make_action(arm_action, self.get_gripper_action())

    def observe_hold(self):
        """Hold measured arm and gripper joints without changing controller state."""

        sim_js = self.robot.get_joints_state()
        positions = sim_js.positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float)
        joint_indices = np.concatenate([self.arm_indices, self.gripper_indices])
        joint_positions = positions[joint_indices].copy()
        arm_count = len(self.arm_indices)
        return self._make_action(
            joint_positions[:arm_count], joint_positions[arm_count:]
        )

    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        gripper_action=None,
        plan_validator=None,
    ):
        action_type = ArticulationAction
        if action_type is None:
            from isaacsim.core.utils.types import ArticulationAction as action_type

        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        plan_flag = torch.logical_or(
            torch.norm(self._ee_trans - ee_trans) > eps,
            torch.norm(self._ee_ori - ee_ori) > eps,
        )
        if not skip_plan:
            new_plan_created = False
            if plan_flag:
                self.phase_executor.clear()
                self._step_idx = 0
                self.num_last_cmd = 0
                self._last_arm_action = None
                self._phase_plan_started = True
                active_command = getattr(self, "_active_phase_command", None)
                if isinstance(active_command, MotionPhaseCommand):
                    result = self.runtime.plan_pose(
                        ee_trans,
                        ee_ori,
                        start_state=self.runtime.arm_joint_state(sim_js).unsqueeze(0),
                        request_metadata=active_command.planning_request_metadata,
                    )
                else:
                    result = self.runtime.plan_pose(
                        ee_trans,
                        ee_ori,
                        start_state=self.runtime.arm_joint_state(sim_js).unsqueeze(0),
                    )
                self.runtime._log_plan_result("ee_forward", result, target=ee_trans.detach().cpu().numpy())
                if self.runtime._result_success(result):
                    raw_plan = self.runtime._result_path(result)
                    trajectory = self.runtime._command_path(raw_plan)
                    if trajectory is not None:
                        self.runtime._install_command_plan(
                            trajectory,
                            target_position=ee_trans,
                            target_orientation=ee_ori,
                            phase_name=self._last_command_name,
                            cached=False,
                        )
                        getattr(self.runtime.setup, "_write_curobo_plan_debug")(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=raw_plan,
                            ordered_trajectory=self.phase_executor.current,
                            branch="single",
                            selected_path_index=0,
                            selected_path_source="native result interpolated_trajectory",
                        )
                        self.num_plan_failed = 0
                        new_plan_created = True
                if not new_plan_created:
                    print("Plan did not converge to a solution.")
                    self._phase_plan_failed = True
                    self.num_plan_failed += 1
                    LOGGER.warning(
                        "[PlanDebug] plan failed robot=%s arm=%s command=%s num_plan_failed=%d",
                        self.name,
                        self.lr_name,
                        self._last_command_name,
                        self.num_plan_failed,
                    )
            if (
                new_plan_created
                and self.phase_executor.current is not None
                and plan_validator is not None
                and not bool(plan_validator(self.phase_executor.current))
            ):
                self.phase_executor.clear()
                self._last_arm_action = None
                self._phase_plan_failed = True
                self.num_plan_failed += 1
            current_trajectory = self.phase_executor.current
            current_index = self.phase_executor.index
            if (
                current_trajectory is not None
                and current_index < len(current_trajectory)
                and self._step_idx % 1 == 0
            ):
                positions, _joint_names = self._trajectory_tensor(current_trajectory)
                arm_action = self._host_array(positions[current_index])
                self._last_arm_action = np.asarray(arm_action, dtype=float).copy()
                art_action = action_type(
                    arm_action,
                    np.zeros_like(arm_action),
                    joint_indices=self.idx_list,
                )
                next_state = self.phase_executor.advance(self.ds_ratio)
                if next_state is None:
                    LOGGER.info(
                        "[PhaseDebug] plan-consumed robot=%s arm=%s phase=%s waypoints=%d stride=%d",
                        self.name,
                        self.lr_name,
                        self._last_command_name,
                        len(current_trajectory),
                        self.ds_ratio,
                    )
                    self.phase_executor.clear()
                    self._phase_plan_finished = True
            else:
                self.num_last_cmd += 1
                if self._last_arm_action is None:
                    arm_action = sim_js.positions[self.arm_indices]
                else:
                    arm_action = self._last_arm_action
                art_action = action_type(joint_positions=arm_action)
        else:
            arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float).copy()
            self._last_arm_action = arm_action.copy()
            art_action = action_type(joint_positions=arm_action)
            # Bookkeeping/update commands deliberately skip motion planning,
            # but Skills still need a finite completion signal.
            self.num_last_cmd += 1
        self._step_idx += 1
        arm_action = art_action.joint_positions
        self._last_commanded_arm_position = np.asarray(arm_action, dtype=float).copy()
        if gripper_action is None:
            gripper_action = self.get_gripper_action()
        else:
            gripper_action = np.asarray(gripper_action, dtype=float)
        self._action = self._make_action(arm_action, gripper_action)
        return self._action

    def get_gripper_action(self):
        spec = self.arm_spec
        sign = -1.0 if spec is not None and spec.gripper_invert else 1.0
        scale = 1.0 if spec is None else float(spec.gripper_scale)
        clip_max = 0.04 if spec is None else float(spec.gripper_clip_max)
        return np.clip(
            sign * self._gripper_state * self._gripper_joint_position * scale,
            0.0,
            clip_max,
        )

    def get_ee_pose(self):
        sim_js = self.robot.get_joints_state()
        q_state = self.runtime.arm_joint_state(sim_js)
        return self.runtime.compute_fk(
            q_state.position,
            joint_names=self.runtime._planner_joint_names(),
        )

    def get_armbase_pose(self):
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.transformations import (
            get_relative_transform,
            pose_from_tf_matrix,
        )

        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task_root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)

    def get_pick_armbase_transform(self):
        """Return the current world transform of this controller's arm base."""

        armbase_tf_getter = getattr(self.robot, "get_armbase_world_transform", None)
        if callable(armbase_tf_getter):
            return armbase_tf_getter()

        reference_prim_path = str(getattr(self, "reference_prim_path", "")).strip()
        if reference_prim_path:
            reference_prim = get_prim_at_path(reference_prim_path)
            if reference_prim.IsValid():
                try:
                    reference_t, reference_q = get_world_pose(reference_prim_path)
                    world_armbase = tf_matrix_from_pose(reference_t, reference_q)
                    if hasattr(self.robot, "get_mobile_base_pose"):
                        try:
                            mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
                            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
                            self._pick_cached_mobile_to_armbase_tf = (
                                np.linalg.inv(world_mobile) @ world_armbase
                            )
                        except Exception:
                            pass
                    return world_armbase
                except Exception:
                    pass

        configured_translation = np.asarray(
            self._pick_configured_mobile_to_armbase_translation, dtype=float
        )
        if configured_translation.shape == (3,):
            if hasattr(self.robot, "get_mobile_base_pose"):
                mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
            else:
                mobile_base_t, mobile_base_q = self.robot.get_world_pose()
            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
            mobile_to_armbase = tf_matrix_from_pose(
                configured_translation,
                self._pick_configured_mobile_to_armbase_orientation,
            )
            self._pick_cached_mobile_to_armbase_tf = mobile_to_armbase
            return world_mobile @ mobile_to_armbase

        reference_prim = get_prim_at_path(self.reference_prim_path)
        task_prim = get_prim_at_path(self.task_root_prim_path)
        raw_task_armbase = get_relative_transform(reference_prim, task_prim)
        mobile_base_prim_path = str(self._pick_mobile_base_prim_path or "").strip()
        if not mobile_base_prim_path:
            return raw_task_armbase

        mobile_base_prim = get_prim_at_path(mobile_base_prim_path)
        if not mobile_base_prim.IsValid():
            return raw_task_armbase
        task_mobile = get_relative_transform(mobile_base_prim, task_prim)
        if self._pick_cached_mobile_to_armbase_tf is None:
            self._pick_cached_mobile_to_armbase_tf = (
                np.linalg.inv(task_mobile) @ raw_task_armbase
            )
        return task_mobile @ self._pick_cached_mobile_to_armbase_tf

    def close_gripper(self):
        self._gripper_state = -1.0

    def open_gripper(self):
        self._gripper_state = 1.0
