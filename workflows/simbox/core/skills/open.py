from copy import deepcopy
import json
import os
import time

import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform
from isaacsim.core.utils.xforms import get_world_pose
from scipy.spatial.transform import Rotation as R
from solver.planner import KPAMPlanner, KPAMPlannerQueries, KPAMRobotFrame


# pylint: disable=unused-argument
@register_skill
class Open(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.stage = task.stage
        self.name = cfg["name"]
        self.skill_cfg = cfg
        art_obj_name = cfg["objects"][0]
        self.art_obj = task.objects[art_obj_name]
        self.planner_setting = cfg["planner_setting"]
        self.contact_pose_index = self.planner_setting["contact_pose_index"]
        self.success_threshold = self.planner_setting["success_threshold"]
        self.success_mode = self.planner_setting.get("success_mode", "abs")
        self.update_art_joint = self.planner_setting.get("update_art_joint", False)
        if kwargs:
            self.world = kwargs["world"]
            self.draw = kwargs["draw"]
        self.manip_list = []

        lr_arm = self.skill_runtime.arm_name
        self.fingers_link_contact_view = task.artcontact_views[robot.name][lr_arm][art_obj_name + "_fingers_link"]
        self.fingers_base_contact_view = task.artcontact_views[robot.name][lr_arm][art_obj_name + "_fingers_base"]
        self.forbid_collision_contact_view = task.artcontact_views[robot.name][lr_arm][
            art_obj_name + "_forbid_collision"
        ]
        self.collision_valid = True
        self.process_valid = True
        self.output_root = str(self.skill_cfg.get("output_root", "output/local_navigation/skills"))
        self.debug_tag = f"{robot.name}_open_{art_obj_name}_{int(time.time() * 1000)}"
        self.debug_dir = os.path.join(self.output_root, self.debug_tag)
        self._plan_failure_debug_path = None

    def _json_ready(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._json_ready(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        return value

    def _write_debug_artifact(self, filename: str, payload: dict):
        os.makedirs(self.debug_dir, exist_ok=True)
        output_path = os.path.join(self.debug_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._json_ready(payload), handle, indent=2, ensure_ascii=False)
        return output_path

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

    def simple_generate_manip_cmds(self):
        if self.skill_cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.skill_cfg["obj_info_path"])

        self.setup_kpam()

        traj_keyframes, sample_times = self.planner.get_keypose()
        if len(traj_keyframes) == 0 and len(sample_times) == 0:
            print("No keyframes found, return empty manip_list")
            self._plan_failure_debug_path = self._write_debug_artifact(
                "open_plan_failure_snapshot.json",
                {
                    "skill_name": self.name,
                    "skill_id": self.skill_cfg.get("id"),
                    "object_name": self.art_obj.object_name,
                    "obj_info_path": self.skill_cfg.get("obj_info_path"),
                    "planner_setting": self.planner_setting,
                    "planner_debug_info": self.planner.debug_info,
                },
            )
            print(f"[open-debug] Wrote open planning failure snapshot: {self._plan_failure_debug_path}")
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
                self.draw.draw_points([(T_world_base @ np.append(keypose[:3, 3], 1))[:3]], [(0, 0, 0, 1)], [7])
        manip_list = []

        for i in range(len(self.traj_keyframes)):
            p_base_ee_tgt = self.traj_keyframes[i][:3, 3]
            q_base_ee_tgt = R.from_matrix(self.traj_keyframes[i][:3, :3]).as_quat(scalar_first=True)
            if i <= self.contact_pose_index:
                cmd = self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    p_base_ee_tgt,
                    q_base_ee_tgt,
                    gripper_action="open_gripper",
                    active_object=self.art_obj.name,
                )
            else:
                cmd = self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    p_base_ee_tgt,
                    q_base_ee_tgt,
                    gripper_action="close_gripper",
                    active_object=self.art_obj.name,
                )
            manip_list.append(cmd)

            if i == self.contact_pose_index:
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
        self.manip_list = manip_list

    def update(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        if self.update_art_joint:
            self.art_obj._articulation_view.set_joint_position_targets(
                positions=curr_joint_p, joint_indices=self.art_obj.object_joint_index
            )

    def get_contact(self, contact_threshold=0.0):
        contact = {}
        fingers_link_contact = np.abs(self.fingers_link_contact_view.get_contact_force_matrix()).squeeze()
        fingers_link_contact = np.sum(fingers_link_contact, axis=-1)
        fingers_link_contact_indices = np.where(fingers_link_contact > contact_threshold)[0]
        contact["fingers_link"] = {
            "fingers_link_contact": fingers_link_contact,
            "fingers_link_contact_indices": fingers_link_contact_indices,
        }

        fingers_base_contact = np.abs(self.fingers_base_contact_view.get_contact_force_matrix()).squeeze()
        fingers_base_contact = np.sum(fingers_base_contact, axis=-1)
        fingers_base_contact_indices = np.where(fingers_base_contact > contact_threshold)[0]
        contact["fingers_base"] = {
            "fingers_base_contact": fingers_base_contact,
            "fingers_base_contact_indices": fingers_base_contact_indices,
        }

        forbid_collision_contact = np.abs(self.forbid_collision_contact_view.get_contact_force_matrix()).squeeze()
        forbid_collision_contact = np.sum(forbid_collision_contact, axis=-1)
        forbid_collision_contact_indices = np.where(forbid_collision_contact > contact_threshold)[0]
        contact["forbid_collision"] = {
            "forbid_collision_contact": forbid_collision_contact,
            "forbid_collision_contact_indices": forbid_collision_contact_indices,
        }

        return contact

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.command_complete(self.manip_list[0])

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
            print("POP one manip cmd")
        if self.is_success():
            self.manip_list.clear()
            print("Open Done")
        return len(self.manip_list) == 0

    def is_success(self):
        contact = self.get_contact()

        if self.skill_cfg.get("collision_valid", True):
            self.collision_valid = (
                self.collision_valid
                and len(contact["forbid_collision"]["forbid_collision_contact_indices"]) == 0
                and len(contact["fingers_base"]["fingers_base_contact_indices"]) == 0
            )
        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.art_obj.get_joints_state().velocities)) < 5
            )

        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        init_joint_p = self.art_obj.articulation_initial_joint_position
        print(
            curr_joint_p - init_joint_p,
            "collision_valid :",
            self.collision_valid,
            "process_valid :",
            self.process_valid,
        )
        if self.success_mode == "normal":
            return (
                (curr_joint_p - init_joint_p) >= np.abs(self.success_threshold)
                and self.collision_valid
                and self.process_valid
            )
        elif self.success_mode == "abs":
            return (
                np.abs(curr_joint_p - init_joint_p) >= np.abs(self.success_threshold)
                and self.collision_valid
                and self.process_valid
            )
