import logging

import numpy as np
from core.planning.config_contract import DIRECT_EXECUTION_MODE
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
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
        self.move_steps = int(self.skill_cfg.get("move_steps", 50))
        if self.move_steps <= 0:
            raise ValueError("heuristic home move_steps must be positive")
        self.t_eps = float(self.skill_cfg.get("t_eps", 0.088))

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self._goal_joints = None
        self.execution_mode = (
            DIRECT_EXECUTION_MODE if self.mode == "home" else "planned"
        )
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
        """Build typed direct joint-space interpolation commands.

        Home deliberately owns this path.  It sends one execution-only typed
        command per interpolation point; no EE target or Physics-schema
        planner request is created.  The final point is exactly the configured
        home posture.
        """

        manip_list = []
        for k in range(self.move_steps):
            alpha = float(k + 1) / float(self.move_steps)
            arm_action = goal_joints * alpha + curr_joints * (1.0 - alpha)
            cmd = self.joint_command(
                arm_action,
                gripper_state=self._gripper_state,
                phase=MotionPhase.CARRY_HOME,
                direct=True,
                replan_allowed=False,
            )
            manip_list.append(cmd)
        return manip_list

    def simple_generate_manip_cmds(self):
        """Generate the legacy joint interpolation for heuristic moves.

        ``mode: home`` is intentionally execution-only.  It must not inspect
        attachments, contacts, or call a collision-aware planner; each
        command is forwarded directly to the controller.  The other legacy
        joint modes retain their existing behavior, while ``rel_ee`` still
        uses the runtime planner only to solve its requested EE target.
        """

        self.manip_list = []
        self.failure_reason = ""
        self.error_message = ""

        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()
        curr_joints = np.asarray(
            self.robot.get_joint_positions(), dtype=float
        )[self._joint_indices]
        if self.mode == "home":
            self._goal_joints = self._joint_home.copy()
        elif self.mode == "abs_qpos":
            self._goal_joints = self.skill_cfg.get("value", self._joint_home)
        elif self.mode == "rel_qpos":
            self._goal_joints = self.skill_cfg.get(
                "value", np.zeros(self._joint_home.shape)
            )
        elif self.mode == "rel_ee":
            p_base_ee_tgt, q_base_ee_tgt = self._compute_ee_goal(
                p_base_ee_cur,
                q_base_ee_cur,
                self.skill_cfg.get("value", np.eye(4)),
            )
            self._goal_joints = self._solve_goal_joints_via_plan(
                p_base_ee_tgt, q_base_ee_tgt
            )
        else:  # pragma: no cover - constructor validates the mode
            raise NotImplementedError(self.mode)

        if self._goal_joints is None:
            self.failure_reason = "NO_HEURISTIC_GOAL"
            self.error_message = "Heuristic move did not produce a joint target."
            return self.manip_list

        self._goal_joints = np.asarray(self._goal_joints, dtype=float)
        self.manip_list = self._build_joint_traj(
            curr_joints,
            self._goal_joints,
            p_base_ee_cur,
            q_base_ee_cur,
        )
        if self.mode == "home":
            LOGGER.info(
                "[HomeDebug] direct interpolation robot=%s arm=%s steps=%d",
                self.skill_runtime.name,
                self.skill_runtime.arm_name,
                len(self.manip_list),
            )
        return self.manip_list

    def is_feasible(self, th=5):
        if self.execution_mode == DIRECT_EXECUTION_MODE:
            return True
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        if len(self.manip_list) == 0:
            return True
        if self._goal_joints is None:
            return True
        command = self.manip_list[0]
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
        return bool(diff_trans < t_eps)
