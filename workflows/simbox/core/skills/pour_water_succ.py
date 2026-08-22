import numpy as np
import torch
from core.skills.base_skill import BaseSkill, register_skill
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.utils.transformation_utils import (
    get_orientation,
    perturb_orientation,
    perturb_position,
)
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from scipy.spatial.transform import Rotation as R


# pylint: disable=unused-argument
@register_skill
class Pour_Water_Succ(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg
        self.frame = self.skill_cfg.get("frame", "robot")
        if self.skill_cfg.get("translation", None):
            self.p_base_ee_tgt = np.array(self.skill_cfg.get("translation", None))
            self.p_base_ee_tgt = perturb_position(self.p_base_ee_tgt, self.skill_cfg.get("max_noise_m", 0.05))
        else:
            self.p_base_ee_tgt = None
        if self.skill_cfg.get("quaternion", None) or self.skill_cfg.get("euler", None):
            self.q_base_ee_tgt = np.array(
                get_orientation(self.skill_cfg.get("euler"), self.skill_cfg.get("quaternion"))
            )
            self.q_base_ee_tgt = perturb_orientation(self.q_base_ee_tgt, self.skill_cfg.get("max_noise_deg", 5))
        else:
            self.q_base_ee_tgt = None

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()
        if self.p_base_ee_tgt is None:
            self.p_base_ee_tgt = p_base_ee_cur
        if self.q_base_ee_tgt is None:
            self.q_base_ee_tgt = q_base_ee_cur
        gripper_state = self.skill_cfg.get("gripper_state", 1.0)
        cmd = self.measured_hold_command(
            gripper_action=(
                "close_gripper" if gripper_state < 0 else "open_gripper"
            ),
            phase=MotionPhase.CARRY_HOME,
        )
        manip_list.append(cmd)

        self.manip_list = manip_list

    # def interp(self, curr_trans, curr_ori, target_trans, target_ori, interp_num, normalize_quaternions=True):
    #     interp_trans = np.linspace(curr_trans, target_trans, interp_num, axis=0)
    #     if normalize_quaternions:
    #         curr_ori = curr_ori / np.linalg.norm(curr_ori)
    #         target_ori = target_ori / np.linalg.norm(target_ori)
    #     rotations = R.from_quat([curr_ori, target_ori], scalar_first=True)  # Current  # Target
    #     slerp = Slerp([0, 1], rotations)
    #     times = np.linspace(0, 1, interp_num)
    #     interp_ori = slerp(times).as_quat(scalar_first=True)

    #     return interp_trans, interp_ori

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, js_eps=5e-3, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        command = self.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("Pour_Water_Succ emits MotionPhaseCommand values only")
        return self.command_complete(command)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        particle_poses = np.array(self.task.particles.GetPointsAttr().Get())
        container_name = self.skill_cfg.get("container_name", "cup")
        container = self.task.objects[container_name]
        container_trans, _ = container.get_world_pose()
        if isinstance(container_trans, torch.Tensor):
            container_trans = container_trans.numpy()
        container_radius = self.skill_cfg.get("container_radius", 0.025)
        distances_xy = np.linalg.norm(particle_poses[:, :2] - container_trans[:2], axis=1)
        in_container_mask = distances_xy < container_radius
        particles_in_container = particle_poses[in_container_mask]
        fluid_flag = (len(particles_in_container) > self.skill_cfg.get("particle_num_th_min", 50)) and (
            len(particles_in_container) < self.skill_cfg.get("particle_num_th_max", 300)
        )

        succ_flag = fluid_flag
        if self.skill_cfg.get("container_up", []):
            container_up_list = self.skill_cfg.get("container_up")
            for container_name, container_up, threshold in container_up_list:
                container = self.task.objects[container_name]
                _, container_ori = container.get_world_pose()

                container_ori = R.from_quat(container_ori, scalar_first=True).as_matrix()
                up_2_idx = {"x": 0, "y": 1, "z": 2}
                container_flag = container_ori[2, up_2_idx[container_up]] > threshold
                succ_flag = succ_flag and container_flag

        print(f"[DEBUG] pour: fluid_flag {fluid_flag} | num_particles_in_container {len(particles_in_container)}")
        return succ_flag
