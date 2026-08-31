import os
from copy import deepcopy

import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.asset_path_utils import resolve_asset_path
from core.utils.usd_geom_utils import compute_bbox
from omegaconf import DictConfig, OmegaConf
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.xforms import get_world_pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# pylint: disable=unused-argument
@register_skill
class Dexplace(BaseSkill):
    def __init__(
        self,
        robot: Robot,
        skill_runtime,
        task: BaseTask,
        cfg: DictConfig,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self._require_skill_runtime()
        self.task = task
        self.skill_cfg = cfg
        self.name = cfg["name"]
        self.pick_obj = task._task_objects[cfg["objects"][0]]
        self.place_obj = task._task_objects[cfg["objects"][1]]
        self.gripper_axis = cfg.get("gripper_axis", None)
        self.camera_axis_filter = cfg.get("camera_axis_filter", None)
        self.place_part_prim_path = cfg.get("place_part_prim_path", None)
        # Get place annotation
        object_name = self.skill_cfg["objects"][0]
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        place_range_path = usd_path.replace("Aligned_obj.usd", "place_range.yaml")
        if os.path.exists(place_range_path):
            with open(place_range_path, "r", encoding="utf-8") as f:
                place_data = OmegaConf.load(f)
                self.x_range = place_data.x_range
                self.y_range = place_data.y_range
        else:
            self.x_range = [0.4, 0.6]
            self.y_range = [0.4, 0.6]
        # Get place_prim
        if self.place_part_prim_path:
            self.place_prim_path = f"{self.place_obj.prim_path}/{self.place_part_prim_path}"
        else:
            self.place_prim_path = self.place_obj.prim_path
        self.robot_ee_path = self.skill_runtime.setup.robot_ee_path
        self.robot_base_path = self.skill_runtime.setup.robot_base_path
        if kwargs:
            self.draw = kwargs["draw"]
        self.manip_list = []

    @staticmethod
    def _get_world_pose_from_path(prim_path):
        return get_world_pose(prim_path)

    def _get_armbase_world_tf(self):
        return tf_matrix_from_pose(*self._get_world_pose_from_path(self.robot_base_path))

    def _get_ee_world_tf(self):
        return tf_matrix_from_pose(*self._get_world_pose_from_path(self.robot_ee_path))

    @staticmethod
    def _get_object_world_tf(obj):
        get_obj_world_pose = getattr(obj, "get_world_pose", None)
        if callable(get_obj_world_pose):
            return tf_matrix_from_pose(*get_obj_world_pose())
        return tf_matrix_from_pose(*obj.get_local_pose())

    def simple_generate_manip_cmds(self):
        manip_list = []
        place_traj, post_place_level = self.sample_gripper_place_traj()
        if len(place_traj) > 1:
            # Having waypoints
            for waypoint in place_traj[:-1]:
                p_base_ee_mid, q_base_ee_mid = waypoint[:3], waypoint[3:]
                manip_list.append(
                    self.pose_command(
                        MotionPhase.TRANSIT_PREPLACE,
                        p_base_ee_mid,
                        q_base_ee_mid,
                        gripper_action="close_gripper",
                        active_object=getattr(self.pick_obj, "name", None),
                        support_object=getattr(self.place_obj, "name", None),
                    )
                )

        # The last waypoint
        p_base_ee_place, q_base_ee_place = place_traj[-1][:3], place_traj[-1][3:]
        manip_list.append(
            self.pose_command(
                MotionPhase.TERMINAL_PLACE_DESCENT,
                p_base_ee_place,
                q_base_ee_place,
                gripper_action="close_gripper",
                active_object=getattr(self.pick_obj, "name", None),
                support_object=getattr(self.place_obj, "name", None),
                allow_object_support_contact=True,
            )
        )
        manip_list.append(
            self.pose_command(
                MotionPhase.GRIPPER_OPEN,
                p_base_ee_place,
                q_base_ee_place,
                gripper_action="open_gripper",
                active_object=getattr(self.pick_obj, "name", None),
                support_object=getattr(self.place_obj, "name", None),
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 10)),
            )
        )

        # Adding a pose place pose to avoid collision when combining place skill and close skill
        T_base_ee_place = tf_matrix_from_pose(p_base_ee_place, q_base_ee_place)
        # Post place
        T_base_ee_postplace = deepcopy(T_base_ee_place)
        # Retreat for a bit along gripper axis
        if "r5a" in self.skill_runtime.planner_build_config.robot_file:
            T_base_ee_postplace[0:3, 3] = T_base_ee_postplace[0:3, 3] - T_base_ee_postplace[0:3, 0] * post_place_level
        else:
            T_base_ee_postplace[0:3, 3] = T_base_ee_postplace[0:3, 3] - T_base_ee_postplace[0:3, 2] * post_place_level
        p_post, q_post = pose_from_tf_matrix(T_base_ee_postplace)
        manip_list.append(
            self.pose_command(
                MotionPhase.TERMINAL_RETREAT,
                p_post,
                q_post,
                gripper_action="open_gripper",
                active_object=getattr(self.pick_obj, "name", None),
                support_object=getattr(self.place_obj, "name", None),
            )
        )
        self.manip_list = manip_list

    def sample_gripper_place_traj(self):
        place_traj = []
        T_world_base = self._get_armbase_world_tf()
        T_world_ee = self._get_ee_world_tf()
        p_world_ee_start, q_world_ee_start = pose_from_tf_matrix(T_world_ee)
        # Getting the object pose
        T_world_obj = self._get_object_world_tf(self.pick_obj)
        # Calculate the pose of the end-effector in the object's coordinate frame
        T_obj_world = np.linalg.inv(T_world_obj)
        # Getting the relation pose and distance of ee to object (after picking, before placing)
        T_obj_ee = T_obj_world @ T_world_ee
        ee2o_distance = np.linalg.norm(T_obj_ee[0:3, 3])
        place_part_prim = get_prim_at_path(self.place_prim_path)
        bbox_place_obj = compute_bbox(place_part_prim)
        x_min, y_min, z_min = bbox_place_obj.min
        x_max, y_max, z_max = bbox_place_obj.max
        self.place_boundary = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        # Calculate the bounding box vertices
        vertices = [
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
        ]
        # Draw the bounding box vertices
        if self.draw:
            for vertex in vertices:
                self.draw.draw_points([vertex], [(0, 0, 0, 1)], [7])  # black

        # 1. Obtaining ee_ori
        initial_ee_pose = self.skill_runtime.setup.T_world_ee_init
        if isinstance(initial_ee_pose, (tuple, list)) and len(initial_ee_pose) == 2:
            initial_ee_tf = tf_matrix_from_pose(*initial_ee_pose)
        else:
            initial_ee_tf = np.asarray(initial_ee_pose, dtype=float)
        p_world_ee_init = initial_ee_tf[0:3, 3]  # getting initial ee position
        container_position = np.array(self._get_object_world_tf(self.place_obj)[:3, 3], copy=True)
        container_position[1] += 0.0
        gripper_axis = container_position - p_world_ee_init  # gripper_axis is aligned with the container direction
        gripper_axis = gripper_axis / np.linalg.norm(gripper_axis)  # Normalize the target vector
        q_world_ee = self.get_ee_ori(gripper_axis, T_world_ee, self.camera_axis_filter)
        # 2. Obtaining p_world_ee
        x = x_min + np.random.uniform(self.x_range[0], self.x_range[1]) * (x_max - x_min)
        y = y_min + np.random.uniform(self.y_range[0], self.y_range[1]) * (y_max - y_min)
        z = z_min + 0.15
        obj_place_position = np.array([x, y, z])
        if self.draw:
            self.draw.draw_points([obj_place_position.tolist()], [(1, 0, 0, 1)], [7])  # red
        p_world_ee = obj_place_position - gripper_axis * ee2o_distance
        # 3. Adding Waypoint
        # Pre place
        p_world_ee_mid = (p_world_ee_start + p_world_ee) / 2.0
        p_world_ee_mid[2] += 0.05
        slerp = Slerp([0, 1], R.from_quat([q_world_ee_start, q_world_ee]))
        q_world_ee_mid = slerp([0.5]).as_quat()[0]
        if self.draw:
            self.draw.draw_points([p_world_ee_mid.tolist()], [(0, 1, 0, 1)], [7])  # green
        place_traj.append(self.adding_waypoint(p_world_ee_mid, q_world_ee_mid, T_world_base))
        # Place
        if self.draw:
            self.draw.draw_points([p_world_ee.tolist()], [(0, 1, 0, 1)], [7])  # green
        place_traj.append(self.adding_waypoint(p_world_ee, q_world_ee, T_world_base))
        post_place_level = 0.1

        return place_traj, post_place_level

    def get_ee_ori(self, gripper_axis, T_world_ee, camera_axis_filter=None):
        gripper_x = gripper_axis
        if camera_axis_filter is not None:
            direction = camera_axis_filter[0]["direction"]
            degree = camera_axis_filter[1]["degree"]
            direction = np.array(direction) / np.linalg.norm(direction)  # Normalize the direction vector
            angle = np.radians(np.random.uniform(degree[0], degree[1]))
            gripper_z = direction - np.dot(direction, gripper_x) * gripper_x
            gripper_z = gripper_z / np.linalg.norm(gripper_z)
            rotation_axis = np.cross(gripper_z, gripper_x)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            gripper_z = R.from_rotvec(angle * rotation_axis).apply(gripper_z)

        else:
            current_z = T_world_ee[0:3, 1]
            gripper_z = current_z - np.dot(current_z, gripper_x) * gripper_x

        gripper_z = gripper_z / np.linalg.norm(gripper_z)
        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y = gripper_y / np.linalg.norm(gripper_y)
        gripper_z = np.cross(gripper_x, gripper_y)
        R_world_ee = np.column_stack((gripper_x, gripper_y, gripper_z))
        q_world_ee = R.from_matrix(R_world_ee).as_quat(scalar_first=True)
        return q_world_ee

    def adding_waypoint(self, p_world_ee, q_world_ee, T_world_base):
        """
        Adding a waypoint, also transform from wolrd frame to robot frame
        """
        T_world_ee = tf_matrix_from_pose(p_world_ee, q_world_ee)
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
        p_base_ee, q_base_ee = pose_from_tf_matrix(T_base_ee)
        waypoint = np.concatenate((p_base_ee, q_base_ee))
        return waypoint

    def is_feasible(self, th=10):
        return self.skill_runtime.execution.state.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return bool(self.skill_runtime.execution.is_phase_command_complete(self.manip_list[0]))

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        x, y, z = self._get_object_world_tf(self.pick_obj)[:3, 3]
        within_boundary = (
            self.place_boundary[0][0] <= x <= self.place_boundary[1][0]
            and self.place_boundary[0][1] <= y <= self.place_boundary[1][1]
            and self.place_boundary[0][2] <= z  # <= self.place_boundary[1][2]
        )

        print("pos :", np.array([x, y, z]))
        print("boundary :", self.place_boundary)
        print("within_boundary :", within_boundary)

        return within_boundary
