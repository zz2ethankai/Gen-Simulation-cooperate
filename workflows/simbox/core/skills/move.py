import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from scipy.spatial.transform import Rotation as R


# pylint: disable=unused-argument
@register_skill
class Move(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.move_obj = task.objects[cfg["objects"][0]]
        self.tgt_obj = task.objects[cfg["objects"][1]]

        if "invisible_object" in cfg:
            self.ivs_obj = task.objects[cfg["invisible_object"][0]]
            self.ivs_obj.set_visibility(True)
        else:
            self.ivs_obj = None
        self.skill_cfg = cfg
        self.success_threshold = cfg["success_threshold"]
        self.delta_trans = np.array(cfg.get("delta_trans", [[0, 0, 0]]))
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        _p_base_ee_cur, q_base_ee_cur = self.skill_runtime.execution.get_ee_pose()

        p_base_ee_tgt = self.getTgtTranslation()
        for delta_trans in self.delta_trans:
            manip_list.append(
                self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    p_base_ee_tgt + delta_trans,
                    q_base_ee_cur,
                    gripper_action="close_gripper",
                    completion_tolerance={"position_m": 0.005, "orientation_rad": 0.05},
                )
            )

        self.manip_list = manip_list
        self.p_base_ee_tgt = p_base_ee_tgt + self.delta_trans[-1]

    def getTgtTranslation(self):
        p_world_move_obj = self.move_obj.get_world_pose()[0]
        p_world_tgt_obj = self.tgt_obj.get_world_pose()[0]
        global_move = p_world_tgt_obj - p_world_move_obj
        _, q_world_base_cur = self.skill_runtime.execution.get_armbase_pose()
        p_base_ee_cur, _ = self.skill_runtime.execution.get_ee_pose()
        R_we = R.from_quat(q_world_base_cur, scalar_first=True).as_matrix()  # EE -> World
        R_ew = R_we.T  # World -> EE
        ee_move = R_ew @ global_move
        ee_move[2] *= 0.0

        p_base_ee_tgt = p_base_ee_cur + ee_move
        return p_base_ee_tgt

    def is_feasible(self, th=5):
        return self.skill_runtime.execution.state.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.command_complete(self.manip_list[0])

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        if self.ivs_obj is not None:
            if self.is_success():
                self.ivs_obj.set_visibility(False)
        return len(self.manip_list) == 0

    def is_success(self):
        p_base_ee_cur, _ = self.skill_runtime.execution.get_ee_pose()
        distance = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        flag = (distance < self.success_threshold) and (len(self.manip_list) == 0)

        # print("distance ee :", distance)

        p_world_move_obj = self.move_obj.get_world_pose()[0]
        p_world_tgt_obj = self.tgt_obj.get_world_pose()[0]
        distance = np.linalg.norm(p_world_tgt_obj - p_world_move_obj)
        flag = (distance < self.success_threshold) and flag

        # print("distance move :", distance)

        return flag
