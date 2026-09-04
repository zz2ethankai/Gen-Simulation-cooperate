from __future__ import annotations
import logging
from typing import Any
import numpy as np
import torch
from core.controllers.curobo.phase_execution import ExecutionStatus
from core.controllers.curobo.trajectory import execution_trajectory_tensor
from core.planning.domain_types import CollisionOptions, CollisionPolicy, CommandStatus
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.controllers.curobo.components import (
    MutableExecutionState,
)
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
LOGGER = logging.getLogger("de_logger")
class ControllerExecution:
    @property
    def ds_ratio(self) -> int:
        return int(self.state.ds_ratio)
    @ds_ratio.setter
    def ds_ratio(self, value: int) -> None:
        self.state.ds_ratio = max(1, int(value))
    def __init__(
        self,
        *,
        name: str = "unknown",
        lr_name: str | None = None,
        robot: Any = None,
        arm_spec: Any = None,
        tensor_args: Any = None,
        raw_js_names: Any = None,
        arm_indices: Any = None,
        gripper_indices: Any = None,
        phase_executor: Any = None,
        runtime: Any = None,
        setup: Any = None,
        robot_base_path: str | None = None,
        robot_ee_path: str | None = None,
        task_root_prim_path: str | None = None,
        reference_prim_path: str | None = None,
        execution_state: MutableExecutionState | None = None,
    ) -> None:
        self.name = name
        self.lr_name = lr_name
        self.robot = robot
        self.arm_spec = arm_spec
        self.tensor_args = tensor_args
        self.raw_js_names = tuple(str(name) for name in (raw_js_names or ()))
        self.arm_indices = np.asarray(
            [] if arm_indices is None else arm_indices, dtype=np.int64
        )
        self.gripper_indices = np.asarray(
            [] if gripper_indices is None else gripper_indices, dtype=np.int64
        )
        self.phase_executor = phase_executor
        self.runtime = runtime
        self.setup = setup
        self.robot_base_path = robot_base_path
        self.robot_ee_path = robot_ee_path
        self.task_root_prim_path = task_root_prim_path
        self.reference_prim_path = reference_prim_path
        self.state = execution_state or MutableExecutionState()
        if self.state.gripper_joint_position is None:
            self.state.gripper_joint_position = np.asarray(
                getattr(self.arm_spec, "gripper_home", (1.0,)),
                dtype=float,
            ).reshape(-1)
        self._action = None
        self._legacy_excluded_obstacles: tuple[str, ...] = ()
    def _begin_phase_command(self, command: MotionPhaseCommand) -> bool:
        if command is self.state.active_phase_command:
            return False
        self.state.active_phase_command = command
        self.state.phase_plan_finished = False
        self.state.phase_bookkeeping_done = False
        self.state.phase_dwell_count = 0
        self.state.phase_tracking_failed = False
        self.state.phase_plan_failed = False
        self.state.last_command_name = command.phase.value
        self.state.phase_base_position, self.state.phase_base_orientation = self.get_armbase_pose()
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
    def _apply_gripper_action(self, action: Any) -> None:
        if action in (None, ""):
            return
        if action == "open_gripper":
            self.open_gripper()
        elif action == "close_gripper":
            self.close_gripper()
        else:
            raise ValueError(f"unsupported typed gripper action: {action!r}")
    def _apply_gripper_state(self, state: float | None) -> None:
        if state is None:
            return
        if float(state) == 1.0:
            self.open_gripper()
        elif float(state) == -1.0:
            self.close_gripper()
        else:
            raise ValueError("direct gripper_state must be exactly 1.0 or -1.0")

    def _legacy_update_specific(self, ignore_substring) -> None:
        """Translate the old substring world filter to exact native paths."""

        manager = getattr(
            getattr(self.runtime, "robot_port", None),
            "collision_scene_manager",
            None,
        )
        if manager is None:
            self._legacy_excluded_obstacles = ()
            return
        if isinstance(ignore_substring, (str, bytes)):
            terms = (str(ignore_substring),)
        else:
            terms = tuple(
                str(value)
                for value in (ignore_substring or ())
                if value is not None and str(value)
            )
        native_paths = set(self.runtime.obstacle_names())
        self._legacy_excluded_obstacles = tuple(
            dict.fromkeys(
                str(path)
                for path in getattr(manager, "collision_prim_paths", ())
                if str(path) in native_paths
                and any(term in str(path) for term in terms)
            )
        )

    def forward_legacy_command(self, command: MotionPhaseCommand):
        """Execute the old art-skill lane through the current typed boundary."""

        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("legacy execution accepts MotionPhaseCommand only")
        if command.is_direct:
            return self.forward_phase_command(command)

        first_step = command is not self.state.active_phase_command
        if first_step and self.state.active_phase_command is not None:
            self.clear_plan_and_hold()
        first_step = self._begin_phase_command(command)
        if first_step:
            self.phase_executor.clear()
            self.state.last_arm_action = None
            if command.metadata.get("legacy_operation") == "update_specific":
                self._legacy_update_specific(
                    command.metadata.get("legacy_ignore_substring", ())
                )
                self.state.phase_bookkeeping_done = True

        self._apply_gripper_action(command.gripper_action)
        if command.joint_target is not None:
            raise ValueError("legacy art execution accepts pose commands only")
        if command.is_bookkeeping or command.phase in {
            MotionPhase.GRIPPER_CLOSE,
            MotionPhase.GRIPPER_OPEN,
        }:
            self.state.phase_dwell_count += 1
            position, orientation = self.get_ee_pose()
            return self.ee_forward(
                position,
                orientation,
                skip_plan=True,
                use_phase_command=False,
            )

        position = command.target_position
        orientation = command.target_orientation
        skip_plan = position is None or orientation is None
        if skip_plan:
            position, orientation = self.get_ee_pose()
        collision_options = CollisionOptions(
            policy=CollisionPolicy.WORLD_TRANSIT,
            excluded_obstacles=self._legacy_excluded_obstacles,
        )
        return self.ee_forward(
            position,
            orientation,
            eps=command.planning_epsilon,
            skip_plan=skip_plan,
            use_phase_command=False,
            collision_options=collision_options,
        )
    def forward_phase_command(self, command: MotionPhaseCommand):
        first_step = self._begin_phase_command(command)
        direct_joint_action = command.direct_joint_action
        if direct_joint_action is not None:
            if command.collision_policy != CollisionPolicy.PASSTHROUGH:
                raise ValueError(
                    "direct_joint_action is reserved for CollisionPolicy.PASSTHROUGH"
                )
            if first_step:
                self.state.phase_bookkeeping_done = True
                self.state.phase_plan_finished = True
            self._apply_gripper_action(command.gripper_action)
            self._apply_gripper_state(command.gripper_state)
            self.state.last_command_name = command.phase.value
            self.state.last_arm_action = np.asarray(direct_joint_action, dtype=float).copy()
            self.state.last_commanded_arm_position = self.state.last_arm_action.copy()
            return self._make_action(
                self.state.last_arm_action,
                self.get_gripper_action(),
            )
        if command.joint_target is not None:
            return self._forward_joint_target(command, first_step)
        if first_step:
            cached_path = self.runtime.prepare_phase(command)
            if command.is_bookkeeping:
                self.state.phase_bookkeeping_done = True
            if cached_path is not None:
                self.runtime._install_command_plan(
                    self.runtime._command_path(cached_path),
                    target_position=command.target_position,
                    target_orientation=command.target_orientation,
                    phase_name=command.phase.value,
                    cached=True,
                )
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            return self.hold_action()
        self._apply_gripper_action(command.gripper_action)
        if command.params.get("hold_position", False):
            self.phase_executor.clear()
            self.state.phase_plan_finished = True
            return self.hold_action()
        if command.is_bookkeeping or command.phase in {
            MotionPhase.GRIPPER_CLOSE,
            MotionPhase.GRIPPER_OPEN,
        }:
            self.state.phase_dwell_count += 1
            position, orientation = self.get_ee_pose()
            return self.ee_forward(position, orientation, skip_plan=True)
        return self.ee_forward(
            command.target_position,
            command.target_orientation,
            eps=command.planning_epsilon,
        )
    def _forward_joint_target(self, command: MotionPhaseCommand, first_step: bool):
        if first_step:
            if command.active_object is not None:
                self.runtime.assert_attached_owner(command.active_object)
            cached_path = command.preplanned_joint_path
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
                result = self.runtime.plan_cspace(
                    command.joint_target,
                    context=command.phase.value,
                    phase_id=command.phase_id,
                    collision_policy=command.collision_policy,
                    active_target=command.active_object,
                    support=command.support_object,
                    collision_options=command.collision_options,
                    profile=command.profile,
                    completion_policy=command.completion_policy,
                    replan_policy=command.replan_policy,
                    metadata=command.metadata,
                )
                self._install_result(result, command.target_position, command.target_orientation, command.phase.value)
        self._apply_gripper_action(command.gripper_action)
        return self._forward_installed_joint_path()
    def _install_result(self, result, position, orientation, phase_name: str) -> None:
        if not result.success:
            self.state.phase_plan_failed = True
            self.state.num_plan_failed += 1
            return
        trajectory = self.runtime._command_path(result.trajectory)
        if trajectory is None:
            self.state.phase_plan_failed = True
            self.state.num_plan_failed += 1
            return
        self.runtime._install_command_plan(
            trajectory,
            target_position=position,
            target_orientation=orientation,
            phase_name=phase_name,
            cached=False,
        )
        self.state.num_plan_failed = 0
    def _trajectory_tensor(self, trajectory, *, context="controller execution trajectory"):
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
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)
    def _forward_installed_joint_path(self):
        sim_js = self.robot.get_joints_state()
        current_trajectory = self.phase_executor.current
        current_index = self.phase_executor.index
        if (
            current_trajectory is not None
            and current_index < len(current_trajectory)
            and self.state.step_idx % 1 == 0
        ):
            positions, _joint_names = self._trajectory_tensor(current_trajectory)
            arm_action = self._host_array(positions[current_index]).copy()
            self.state.last_arm_action = arm_action
            next_state = self.phase_executor.advance(self.ds_ratio)
            if next_state is None:
                self.phase_executor.clear()
                self.state.phase_plan_finished = True
        else:
            self.state.num_last_cmd += 1
            if self.state.last_arm_action is None:
                arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
            else:
                arm_action = np.asarray(self.state.last_arm_action, dtype=float).copy()
        self.state.step_idx += 1
        self.state.last_commanded_arm_position = np.asarray(arm_action, dtype=float).copy()
        return self._make_action(arm_action, self.get_gripper_action())
    def complete_terminal_place_on_contact(self, command: MotionPhaseCommand) -> None:
        if command is not self.state.active_phase_command:
            return
        self.phase_executor.clear()
        self.state.phase_plan_finished = True
        self.state.last_arm_action = None
        if not command.params.get("contact_stop_logged", False):
            LOGGER.info(
                "[PhaseDebug] contact-stop robot=%s arm=%s phase=%s",
                self.name,
                self.lr_name,
                command.phase.value,
            )
            command.params["contact_stop_logged"] = True
    def is_phase_command_complete(self, command: MotionPhaseCommand) -> bool:
        if command is not self.state.active_phase_command:
            return False
        if command.joint_target is not None:
            return bool(self.state.phase_plan_finished and self.phase_executor.current is None)
        if command.direct_joint_action is not None:
            return bool(self.state.phase_plan_finished)
        if command.is_bookkeeping:
            if (
                command.phase == MotionPhase.DETACH_AND_SETTLE
                and self.state.phase_dwell_count >= max(1, command.dwell_steps)
                and not command.params.get("settle_finalized", False)
            ):
                self.runtime.finalize_detach_target(command.active_object)
                command.params["settle_finalized"] = True
            return self.state.phase_bookkeeping_done and self.state.phase_dwell_count >= max(1, command.dwell_steps)
        if command.phase in {MotionPhase.GRIPPER_CLOSE, MotionPhase.GRIPPER_OPEN}:
            return self.state.phase_dwell_count >= max(1, command.dwell_steps)
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            self.phase_executor.clear()
            self.state.phase_plan_finished = True
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
            self.state.phase_plan_finished = True
            return True
        if self.state.phase_plan_finished and not inside:
            self.state.phase_tracking_failed = True
        return False
    def execution_status(self, command=None) -> ExecutionStatus:
        active = command is None or command is self.state.active_phase_command
        current = command if command is not None else self.state.active_phase_command
        executor_status = self.phase_executor.status()
        phase = current.phase.value if current is not None else None
        complete = bool(
            active
            and command is not None
            and self.is_phase_command_complete(command)
        )
        plan_failed = bool(self.state.phase_plan_failed)
        tracking_failed = bool(self.state.phase_tracking_failed)
        reason = None
        if plan_failed:
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
            active=bool(self.state.active_phase_command is not None and active),
            last_commanded_arm_position=(
                None
                if self.state.last_commanded_arm_position is None
                else np.asarray(self.state.last_commanded_arm_position, dtype=float).copy()
            ),
            phase_base_position=(
                None
                if self.state.phase_base_position is None
                else np.asarray(self.state.phase_base_position, dtype=float).copy()
            ),
            phase_base_orientation=(
                None
                if self.state.phase_base_orientation is None
                else np.asarray(self.state.phase_base_orientation, dtype=float).copy()
            ),
            gripper_state=float(self.state.gripper_state),
        )
    def clear_plan_and_hold(self) -> None:
        self.phase_executor.clear()
        self.state.phase_plan_finished = False
        self.state.phase_tracking_failed = False
        self.state.phase_plan_failed = False
        self.state.active_phase_command = None
        position, orientation = self.get_ee_pose()
        self.state.ee_trans = self.tensor_args.to_device(position)
        self.state.ee_ori = self.tensor_args.to_device(orientation)
    def _make_action(self, arm_action, gripper_action):
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
        sim_js = self.robot.get_joints_state()
        arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
        return self._make_action(arm_action, self.get_gripper_action())
    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        gripper_action=None,
        *,
        use_phase_command=True,
        collision_options: CollisionOptions | None = None,
    ):
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        changed = bool(
            torch.logical_or(
                torch.norm(self.state.ee_trans - ee_trans) > eps,
                torch.norm(self.state.ee_ori - ee_ori) > eps,
            )
        )
        if not skip_plan and changed:
            self.phase_executor.clear()
            self.state.step_idx = self.state.num_last_cmd = 0
            self.state.last_arm_action = None
            command = self.state.active_phase_command
            kwargs = {"start_state": self.runtime.arm_joint_state(sim_js).unsqueeze(0)}
            if use_phase_command and isinstance(command, MotionPhaseCommand):
                kwargs["command"] = command
            else:
                kwargs["phase_id"] = self.state.last_command_name
                kwargs["collision_policy"] = CollisionPolicy.WORLD_TRANSIT
                if collision_options is not None:
                    kwargs["collision_options"] = collision_options
            result = self.runtime.plan_pose(ee_trans, ee_ori, **kwargs)
            self._install_result(
                result, ee_trans, ee_ori, self.state.last_command_name
            )
        trajectory = self.phase_executor.current if not skip_plan else None
        if trajectory is not None and self.phase_executor.index < len(trajectory):
            positions, _ = self._trajectory_tensor(trajectory)
            arm_action = self._host_array(positions[self.phase_executor.index])
            self.state.last_arm_action = arm_action.copy()
            if self.phase_executor.advance(self.ds_ratio) is None:
                self.phase_executor.clear()
                self.state.phase_plan_finished = True
        else:
            self.state.num_last_cmd += 1
            arm_action = np.asarray(
                sim_js.positions[self.arm_indices] if self.state.last_arm_action is None
                else self.state.last_arm_action,
                dtype=float,
            ).copy()
        self.state.step_idx += 1
        self.state.last_commanded_arm_position = arm_action.copy()
        return self._make_action(
            arm_action,
            self.get_gripper_action() if gripper_action is None else gripper_action,
        )
    def get_gripper_action(self):
        spec = self.arm_spec
        sign = -1.0 if spec is not None and spec.gripper_invert else 1.0
        scale = 1.0 if spec is None else float(spec.gripper_scale)
        clip_max = 0.04 if spec is None else float(spec.gripper_clip_max)
        return np.clip(
            sign * self.state.gripper_state * self.state.gripper_joint_position * scale,
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
        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task_root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)
    def get_pick_armbase_transform(self):
        return np.asarray(self.robot.get_armbase_world_transform(self.lr_name), dtype=float)
    def close_gripper(self):
        self.state.gripper_state = -1.0
    def open_gripper(self):
        self.state.gripper_state = 1.0
