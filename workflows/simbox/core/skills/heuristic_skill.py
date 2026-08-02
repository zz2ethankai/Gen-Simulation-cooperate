import numpy as np
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.transformations import (
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Heuristic_Skill(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg

        self.lr_hand = "right" if "right" in self.controller.robot_file else "left"
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
        Use controller.plan to get a collision-free joint path,
        and take the last waypoint as goal arm joints.
        """
        if self.controller.use_batch:
            raise NotImplementedError

        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.controller.plan(ee_trans_goal, ee_ori_goal, sim_js, js_names)
        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan()
            goal_arm_joints = cmd_plan[-1].position.cpu().numpy()  # replace by ik
            return goal_arm_joints
        else:
            return None

    def _build_joint_traj(self, curr_joints, goal_joints, p_base_ee_cur, q_base_ee_cur):
        """Build a list of dummy_forward commands interpolating in joint space."""
        manip_list = []
        for k in range(self.move_steps):
            alpha = float(k + 1) / float(self.move_steps) * 1.25
            arm_action = goal_joints * alpha + curr_joints * (1.0 - alpha)
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "dummy_forward",
                {
                    "arm_action": arm_action,
                    "gripper_state": self._gripper_state,
                },
            )
            manip_list.append(cmd)
        return manip_list

    def simple_generate_manip_cmds(self):
        if getattr(self.controller, "collision_world_mode", "legacy_stage_scan") == "physics_schema":
            return self._physics_schema_generate_manip_cmds()

        self.manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]

        if self.mode == "home":
            self._goal_joints = self._joint_home.copy()
        else:
            if self.mode == "abs_qpos":
                self._goal_joints = self.skill_cfg.get("value", self._joint_home)
            elif self.mode == "rel_qpos":
                self._goal_joints = self.skill_cfg.get("value", np.zeros(self._joint_home.shape))
            elif self.mode == "rel_ee":
                p_base_ee_tgt, q_base_ee_tgt = self._compute_ee_goal(
                    p_base_ee_cur, q_base_ee_cur, self.skill_cfg.get("value", np.eye(4))
                )
                self._goal_joints = self._solve_goal_joints_via_plan(p_base_ee_tgt, q_base_ee_tgt)
            else:
                raise NotImplementedError

        if self._goal_joints is None:
            self.manip_list = []
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "update_specific",
                {
                    "ignore_substring": self.controller.ignore_substring,
                    "reference_prim_path": self.controller.reference_prim_path,
                },
            )
            self.manip_list.append(cmd)
            return

        self.manip_list = self._build_joint_traj(curr_joints, self._goal_joints, p_base_ee_cur, q_base_ee_cur)

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
        manager = self.controller.collision_scene_manager
        manager.assert_attached_owner(
            object_name, self.controller.name, self.controller.lr_name
        )
        if float(self._gripper_state) >= 0.0:
            raise RuntimeError(
                "Physics-schema carry-home must keep the gripper closed while the object is attached"
            )
        self._pickcontact_view = self.task.pickcontact_views[
            self.robot.name
        ][self.controller.lr_name][object_name]
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
            candidate_result = self.controller.plan_joint_positions(candidate)
            success = bool(
                np.asarray(candidate_result.success.detach().cpu()).any()
            )
            if success:
                self._goal_joints = candidate
                result = candidate_result
                selected_progress = progress
                break
        if result is None:
            self.controller.num_plan_failed += 1
            self.failure_reason = "NO_COLLISION_FREE_CARRY_HOME_PLAN"
            self.error_message = (
                "Could not plan any configured carry posture with the attached object "
                f"in the Physics world; attempted_home_progress={progress_candidates}."
            )
            return
        self.controller.num_plan_failed = 0
        target_position, target_orientation = self.controller.forward_kinematic(
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
                    "preplanned_joint_path": result.get_interpolated_plan(),
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
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        if len(self.manip_list) == 0:
            return True
        if self._goal_joints is None:
            return True
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        target_joints = self.manip_list[0][3]["arm_action"]
        diff_trans = np.linalg.norm(curr_joints - target_joints)
        pose_flag = diff_trans < t_eps
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if isinstance(self.manip_list[0], MotionPhaseCommand):
            if self.controller.is_phase_command_complete(self.manip_list[0]):
                self.manip_list.pop(0)
            return len(self.manip_list) == 0
        if self.is_subtask_done(t_eps=self.t_eps):
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
            manager = self.controller.collision_scene_manager
            try:
                manager.assert_attached_owner(
                    self._physics_schema_active_object,
                    self.controller.name,
                    self.controller.lr_name,
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
