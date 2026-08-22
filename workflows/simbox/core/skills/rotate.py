from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from omegaconf import DictConfig, OmegaConf
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform
from isaacsim.core.utils.xforms import get_world_pose
from scipy.spatial.transform import Rotation as R
from solver.planner import KPAMPlanner, KPAMPlannerQueries, KPAMRobotFrame


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Rotate(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.stage = task.stage
        self.name = cfg["name"]
        art_obj_name = cfg["objects"][0]
        self.art_obj = task.objects[art_obj_name]
        self.cfg = cfg

        # debug start: KPAMPlanner
        self.planner_setting = OmegaConf.to_container(cfg["planner_setting"])
        self.contact_pose_index = self.planner_setting.get("contact_pose_index")
        self.success_threshold = self.planner_setting.get("success_threshold", 0.785)  # 45 degrees
        if kwargs:
            self.world = kwargs["world"]
            self.draw = kwargs["draw"]
        # debug end: KPAMPlanner
        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

        if self.cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.cfg["obj_info_path"])

    # debug start: KPAMPlanner
    def setup_kpam(self):
        robot_config = self.skill_runtime.robot_config
        robot_frame = KPAMRobotFrame.from_config(
            robot_config=robot_config,
            arm_name=self.skill_runtime.arm_name,
            base_path=self.skill_runtime.robot_base_path,
            ee_path=self.skill_runtime.robot_ee_path,
            hand_path=self.skill_runtime.robot_ee_path,
        )
        queries = KPAMPlannerQueries(
            get_joint_positions=self.robot.get_joint_positions,
            get_world_pose=get_world_pose,
            get_prim_at_path=get_prim_at_path,
            get_relative_transform=get_relative_transform,
        )
        self.planner = KPAMPlanner(
            env=self.world,
            object=self.art_obj,
            cfg_path=self.planner_setting,
            robot_name=self.robot.name,
            robot_config=robot_config,
            robot_frame=robot_frame,
            queries=queries,
            draw_points=self.draw,
            stage=self.stage,
        )
        if "additional_labels" in self.planner_setting:
            new_value = self.planner_setting["additional_labels"].get(
                self.art_obj.asset_relative_path, self.planner.modify_actuation_motion
            )
            self.planner.modify_actuation_motion = new_value

    def simple_generate_manip_cmds(self):
        if self.cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.cfg["obj_info_path"])

        self.setup_kpam()

        traj_keyframes, sample_times = self.planner.get_keypose()
        if len(traj_keyframes) == 0 and len(sample_times) == 0:
            print("No keyframes found, return empty manip_list")
            self.manip_list = []
            return
        T_world_base = get_relative_transform(
            get_prim_at_path(self.skill_runtime.robot_base_path),
            get_prim_at_path(self.task.root_prim_path),
        )
        self.traj_keyframes = traj_keyframes
        self.sample_times = sample_times
        if self.draw:
            for keypose in traj_keyframes:
                self.draw.draw_points([(T_world_base @ np.append(keypose[:3, 3], 1))[:3]], [(0, 0, 0, 1)], [7])  # black
        manip_list = []

        for i in range(len(self.traj_keyframes)):
            p_base_ee_tgt = self.traj_keyframes[i][:3, 3]
            q_base_ee_tgt = R.from_matrix(self.traj_keyframes[i][:3, :3]).as_quat(scalar_first=True)
            if i <= self.contact_pose_index - 1:
                manip_list.append(
                    self.pose_command(
                        MotionPhase.TRANSIT_PREGRASP,
                        p_base_ee_tgt,
                        q_base_ee_tgt,
                        gripper_action="open_gripper",
                        active_object=self.art_obj.name,
                    )
                )
            elif i == self.contact_pose_index:
                if "hearth" in self.art_obj.name:
                    action = "open_gripper"
                else:
                    action = "close_gripper"
                phase = MotionPhase.GRIPPER_CLOSE if action == "close_gripper" else MotionPhase.GRIPPER_OPEN
                manip_list.append(
                    self.pose_command(
                        phase,
                        p_base_ee_tgt,
                        q_base_ee_tgt,
                        gripper_action=action,
                        active_object=self.art_obj.name,
                        replan_allowed=False,
                        dwell_steps=40,
                    )
                )
            else:
                manip_list.append(
                    self.pose_command(
                        MotionPhase.TRANSIT_PREGRASP,
                        p_base_ee_tgt,
                        q_base_ee_tgt,
                        gripper_action="close_gripper",
                        active_object=self.art_obj.name,
                    )
                )

        if "hearth" in self.art_obj.name:
            manip_list.append(
                self.pose_command(
                    MotionPhase.GRIPPER_CLOSE,
                    p_base_ee_tgt,
                    q_base_ee_tgt,
                    gripper_action="close_gripper",
                    active_object=self.art_obj.name,
                    replan_allowed=False,
                    dwell_steps=40,
                )
            )
            current_arm = np.asarray(
                self.robot.get_joints_state().positions[self.skill_runtime.arm_indices],
                dtype=float,
            )
            for k in range(100):
                target = current_arm.copy()
                target[-1] -= self.success_threshold * k / 50
                cmd = MotionPhaseCommand(
                    phase=MotionPhase.CARRY_HOME,
                    joint_target=target,
                    gripper_action="close_gripper",
                )
                manip_list.append(cmd)

        self.manip_list = manip_list

    def update(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        self.art_obj._articulation_view.set_joint_position_targets(
            positions=curr_joint_p, joint_indices=self.art_obj.object_joint_index
        )

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        command = self.manip_list[0]
        return self.command_complete(command)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
            print("POP one manip cmd")
        if self.is_success():
            self.manip_list.clear()
            print("Rotate Done")
        return len(self.manip_list) == 0

    def is_success(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        distance = np.abs(curr_joint_p - self.art_obj.articulation_initial_joint_position)
        return distance >= np.abs(self.success_threshold)
