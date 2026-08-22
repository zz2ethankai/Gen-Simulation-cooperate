import logging

import numpy as np
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.transformations import (
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)


LOGGER = logging.getLogger("de_logger")


# pylint: disable=unused-argument
@register_skill
class Heuristic_Skill(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg

        self.lr_hand = "right" if "right" in self.skill_runtime.robot_file else "left"
        if self.lr_hand == "left":
            self._joint_indices = self.robot.left_joint_indices
            self._joint_home = self.robot.left_joint_home
            self._joint_home = np.array(self._joint_home)
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.left_gripper_state
        else:
            self._joint_indices = self.robot.right_joint_indices
            self._joint_home = self.robot.right_joint_home
            self._joint_home = np.array(self._joint_home)
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.right_gripper_state

        ALLOWED_MODES = {"abs_qpos", "rel_qpos", "rel_ee", "home"}

        self.mode = self.skill_cfg.get("mode", "home")
        if self.mode not in ALLOWED_MODES:
            raise ValueError(
                f"Unsupported mode '{self.mode}' for JointMove. Allowed modes are: {sorted(ALLOWED_MODES)}"
            )
        self.move_steps = self.skill_cfg.get("move_steps", 50)
        self.t_eps = self.skill_cfg.get("t_eps", 0.088)

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self._goal_joints = None
        self._physics_schema_active_object = None
        self._pickcontact_view = None
        self.failure_reason = ""
        self.error_message = ""

    def replan_after_safety(self, command):
        """Rebuild a cached structured phase path after a safety hold.

        ``ExecutionSupervisor`` clears the controller's active plan so the
        next plan starts at the measured hold state.  Most structured phases
        can then plan lazily in ``ee_forward``; Physics-schema CARRY_HOME is
        deliberately different because it executes a preplanned joint path
        and must preserve the attached-object carry target.
        """

        if command.phase != MotionPhase.CARRY_HOME:
            return True
        if self._goal_joints is None:
            self.failure_reason = "CARRY_HOME_REPLAN_NO_GOAL"
            self.error_message = "Carry-home recovery has no retained goal joints."
            return False

        object_name = getattr(self, "_physics_schema_active_object", None)
        try:
            self.skill_runtime.assert_attached_owner(object_name)
            result = self.skill_runtime.plan_cspace(self._goal_joints, context="carry_home_replan")
            success = bool(result.success)
            if not success:
                self.skill_runtime.record_plan_failure()
                self.failure_reason = "CARRY_HOME_REPLAN_FAILED"
                self.error_message = (
                    "CuRobo could not rebuild the carry-home path from the "
                    "measured safety-hold state."
                )
                return False
            path = result.trajectory
            if path is None:
                self.failure_reason = "CARRY_HOME_REPLAN_EMPTY"
                self.error_message = "CuRobo returned no interpolated carry-home path."
                return False
        except Exception as exc:
            self.failure_reason = "CARRY_HOME_REPLAN_ERROR"
            self.error_message = str(exc)
            LOGGER.exception(
                "[PhaseDebug] carry-home recovery failed robot=%s arm=%s object=%s",
                self.skill_runtime.name,
                self.skill_runtime.arm_name,
                object_name,
            )
            return False

        command.params["preplanned_joint_path"] = path
        self.skill_runtime.reset_plan_failures()
        LOGGER.info(
            "[PhaseDebug] replanned robot=%s arm=%s phase=%s object=%s cached=true",
            self.skill_runtime.name,
            self.skill_runtime.arm_name,
            command.phase.value,
            object_name,
        )
        return True

    def _compute_ee_goal(self, p_base_ee_cur, q_base_ee_cur, rel_ee):
        """
        rel_ee: (4,4) transformation matrix
        """
        T_base_ee = tf_matrix_from_pose(p_base_ee_cur, q_base_ee_cur)
        if isinstance(rel_ee, (list, tuple)):
            rel_ee = np.array(rel_ee)
        T_base_ee_tgt = rel_ee @ T_base_ee
        p_base_ee_tgt, q_base_ee_tgt = pose_from_tf_matrix(T_base_ee_tgt)
        return p_base_ee_tgt, q_base_ee_tgt

    def _solve_goal_joints_via_plan(self, ee_trans_goal, ee_ori_goal):
        """
        Use the runtime port's planner to get a collision-free joint path,
        and take the last waypoint as goal arm joints.
        """
        result = self.skill_runtime.plan_pose(
            ee_trans_goal,
            ee_ori_goal,
            context="heuristic_rel_ee",
        )
        if not result.success:
            return None
        planned_path = result.trajectory
        if planned_path is None:
            return None
        positions = np.asarray(planned_path.positions, dtype=float)
        goal_arm_joints = positions[-1] if positions.ndim > 1 else positions
        return goal_arm_joints

    def _build_joint_traj(self, curr_joints, goal_joints, p_base_ee_cur, q_base_ee_cur):
        """Build typed c-space targets interpolating in joint space."""
        manip_list = []
        for k in range(self.move_steps):
            alpha = float(k + 1) / float(self.move_steps) * 1.25
            arm_action = goal_joints * alpha + curr_joints * (1.0 - alpha)
            target_position, target_orientation = self.skill_runtime.compute_fk(
                arm_action,
                joint_names=self.skill_runtime.raw_joint_names,
            )
            cmd = MotionPhaseCommand(
                MotionPhase.CARRY_HOME,
                target_position,
                target_orientation,
                gripper_action=(
                    "open_gripper" if float(self._gripper_state) >= 0.0 else "close_gripper"
                ),
                replan_allowed=False,
                joint_target=np.asarray(arm_action, dtype=float),
            )
            manip_list.append(cmd)
        return manip_list

    def simple_generate_manip_cmds(self):
        return self._physics_schema_generate_manip_cmds()

    def _physics_schema_generate_manip_cmds(self):
        """Plan an attached-object carry posture without leaving Physics mode."""

        self.manip_list = []
        self.failure_reason = ""
        self.error_message = ""
        if self.mode != "home":
            raise RuntimeError(
                f"Physics-schema heuristic adapter only supports mode='home', got {self.mode!r}"
            )
        object_name = getattr(self, "_physics_schema_active_object", None)
        if not object_name:
            raise RuntimeError("Physics-schema carry-home requires an attached object")
        self.skill_runtime.assert_attached_owner(object_name)
        if float(self._gripper_state) >= 0.0:
            raise RuntimeError(
                "Physics-schema carry-home must keep the gripper closed while the object is attached"
            )
        self._pickcontact_view = self.task.pickcontact_views[
            self.robot.name
        ][self.skill_runtime.arm_name][object_name]
        current_joints = np.asarray(
            self.robot.get_joint_positions(), dtype=float
        )[self._joint_indices]
        configured_progress = self.skill_cfg.get(
            "physics_home_progress_candidates", [1.0, 0.75, 0.5, 0.25, 0.125]
        )
        try:
            progress_candidates = sorted(
                {
                    float(progress)
                    for progress in configured_progress
                    if 0.0 < float(progress) <= 1.0
                },
                reverse=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "physics_home_progress_candidates must contain numeric values in (0, 1]"
            ) from exc
        if not progress_candidates:
            raise ValueError(
                "physics_home_progress_candidates must contain at least one value in (0, 1]"
            )

        result = None
        selected_progress = None
        for progress in progress_candidates:
            candidate = current_joints + progress * (
                self._joint_home - current_joints
            )
            candidate_result = self.skill_runtime.plan_cspace(candidate, context="carry_home")
            success = bool(candidate_result.success)
            if success:
                self._goal_joints = candidate
                result = candidate_result
                selected_progress = progress
                break
        if result is None:
            self.skill_runtime.record_plan_failure()
            self.failure_reason = "NO_COLLISION_FREE_CARRY_HOME_PLAN"
            self.error_message = (
                "Could not plan any configured carry posture with the attached object "
                f"in the Physics world; attempted_home_progress={progress_candidates}."
            )
            return
        self.skill_runtime.reset_plan_failures()
        carry_path = result.trajectory
        if carry_path is None:
            self.failure_reason = "CARRY_HOME_EMPTY_PATH"
            self.error_message = "CuRobo reported a carry-home success without a trajectory."
            return
        target_position, target_orientation = self.skill_runtime.compute_fk(
            self._goal_joints
        )
        self.manip_list = [
            MotionPhaseCommand(
                MotionPhase.CARRY_HOME,
                target_position,
                target_orientation,
                gripper_action="close_gripper",
                active_object=object_name,
                allow_target_finger_contact=True,
                completion_tolerance={
                    "position_m": float(self.skill_cfg.get("physics_t_eps", 0.01)),
                    "orientation_rad": float(
                        self.skill_cfg.get("physics_o_eps", 0.05)
                    ),
                },
                params={
                    "preplanned_joint_path": carry_path,
                    "home_progress": selected_progress,
                },
            )
        ]

    def get_contact(self, contact_threshold=0.0):
        if self._pickcontact_view is None:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=int)
        values = np.asarray(
            self._pickcontact_view.get_contact_force_matrix(), dtype=float
        )
        if not values.size:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=int)
        contact = np.atleast_1d(np.sum(np.abs(values), axis=-1).squeeze())
        indices = np.where(contact > float(contact_threshold))[0]
        return contact, indices

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        if len(self.manip_list) == 0:
            return True
        if self._goal_joints is None:
            return True
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        command = self.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("Heuristic_Skill emits MotionPhaseCommand values only")
        return self.command_complete(command)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.command_complete(self.manip_list[0]):
            self.manip_list.pop(0)
        if self.is_success(t_eps=self.t_eps):
            self.manip_list.clear()
            print("Heuristic Skill Done")
        return len(self.manip_list) == 0

    def is_success(self, t_eps=0.088):
        if self._goal_joints is None:
            print("cannot compute goal joints, skill failure")
            return False

        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        diff_trans = np.linalg.norm(curr_joints - self._goal_joints)
        success = bool(diff_trans < t_eps)
        if self._physics_schema_active_object:
            try:
                self.skill_runtime.assert_attached_owner(
                    self._physics_schema_active_object
                )
            except Exception as exc:
                self.failure_reason = "CARRY_HOME_ATTACHMENT_LOST"
                self.error_message = str(exc)
                return False
            _, indices = self.get_contact(
                float(self.skill_cfg.get("grasp_contact_threshold_n", 0.0))
            )
            if len(indices) == 0:
                self.failure_reason = "CARRY_HOME_GRASP_CONTACT_LOST"
                self.error_message = "Attached object lost finger contact during carry-home."
                return False
        return success
