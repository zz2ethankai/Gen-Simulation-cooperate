"""PandaOmron mobile manipulator implementation."""

from copy import deepcopy
import math

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform, pose_from_tf_matrix, tf_matrix_from_pose
from pxr import UsdPhysics


# pylint: disable=line-too-long,unused-argument
@register_robot
class PandaOmron(TemplateRobot):
    """Single-arm Panda on an Omron-style wheel-speed mobile base."""

    def __init__(self, *args, **kwargs):
        self.base_cfg = {}
        self.base_steering_joint_names = []
        self.base_wheel_joint_names = []
        self.base_steering_joint_indices = []
        self.base_wheel_joint_indices = []
        self._base_initial_steering_positions = None
        self.mobile_base_prim_path = None
        self._wheel_collision_paths = []
        self._wheel_joint_paths = []
        self._wheel_physics_material_path = None
        self._mobile_to_armbase_translation = None
        self._mobile_to_armbase_orientation = None
        self._active_manipulator_joint_positions = None
        self._active_manipulator_joint_indices = None
        super().__init__(*args, **kwargs)
        self.base_cfg = deepcopy(self.cfg["base"])
        self.base_steering_joint_names = list(self.base_cfg["steering_joint_names"])
        self.base_wheel_joint_names = list(self.base_cfg["wheel_joint_names"])
        self._setup_mobile_base_interface()
        self._demote_imported_site_visuals()
        self._configure_mobile_base_wheel_drives()

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg.get("right_joint_indices", [])
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg.get("right_gripper_indices", [])
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def _setup_paths(self):
        self.fl_ee_path = f"{self.robot_prim_path}/{self.cfg['fl_ee_path']}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_base_path = ""
        self.fr_hand_path = ""
        self.fr_ee_path = ""
        self._mobile_to_armbase_translation = np.asarray(
            self.cfg["fl_base_mount_translation"],
            dtype=np.float32,
        )
        self._mobile_to_armbase_orientation = np.asarray(
            self.cfg["fl_base_mount_orientation"],
            dtype=np.float32,
        )

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = {}

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = []
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = []

    def _get_gripper_state(self, gripper_home):
        if not gripper_home:
            return 1.0
        if len(gripper_home) >= 2:
            return 1.0 if gripper_home[0] >= 0.035 and gripper_home[1] <= -0.035 else -1.0
        return 1.0 if gripper_home[0] >= 0.035 else -1.0

    def _setup_joint_velocities(self):
        arm_indices = list(self.left_joint_indices)
        if arm_indices:
            self._articulation_view.set_max_joint_velocities(
                np.asarray([500.0] * len(arm_indices), dtype=np.float32),
                joint_indices=np.asarray(arm_indices, dtype=np.int32),
            )

    def _set_initial_positions(self):
        super()._set_initial_positions()
        manip_positions = self.left_joint_home + self.left_gripper_home
        manip_indices = self.left_joint_indices + self.left_gripper_indices
        if manip_positions and manip_indices:
            positions = np.asarray(manip_positions, dtype=np.float32).reshape(1, -1)
            indices = np.asarray(manip_indices, dtype=np.int32)
            self._articulation_view.set_joint_position_targets(
                positions,
                joint_indices=indices,
            )
            self._articulation_view.set_joint_velocities(
                np.zeros_like(positions),
                joint_indices=indices,
            )
            self._set_active_manipulator_position_target(positions, indices)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._setup_manipulator_joint_indices()
        self._setup_joint_velocities()
        self._set_initial_positions()
        self._configure_manipulator_drives()
        self._setup_base_joint_indices()
        self._configure_mobile_base_wheel_drives()
        self._validate_mobile_base_joint_partition()
        self._capture_base_initial_steering_positions()

    def _setup_manipulator_joint_indices(self):
        dof_names = list(self._articulation_view.dof_names)
        arm_joint_names = list(self.cfg["left_joint_names"])
        gripper_joint_names = list(self.cfg["left_gripper_joint_names"])
        required_names = arm_joint_names + gripper_joint_names
        missing = [name for name in required_names if name not in dof_names]
        if missing:
            raise KeyError(
                "PandaOmron missing expected manipulator DOF names: "
                f"{missing}; available={dof_names}"
            )
        self.left_joint_indices = [dof_names.index(name) for name in arm_joint_names]
        self.left_gripper_indices = [dof_names.index(name) for name in gripper_joint_names]

    def _setup_base_joint_indices(self):
        dof_names = list(self._articulation_view.dof_names)
        missing = [
            name
            for name in self.base_steering_joint_names + self.base_wheel_joint_names
            if name not in dof_names
        ]
        if missing:
            raise KeyError(f"PandaOmron missing expected mobile-base DOF names: {missing}; available={dof_names}")
        self.base_steering_joint_indices = [dof_names.index(name) for name in self.base_steering_joint_names]
        self.base_wheel_joint_indices = [dof_names.index(name) for name in self.base_wheel_joint_names]

    def _validate_mobile_base_joint_partition(self):
        base_indices = set(self.base_steering_joint_indices) | set(self.base_wheel_joint_indices)
        manip_indices = set(self.left_joint_indices) | set(self.left_gripper_indices)
        overlap = sorted(base_indices & manip_indices)
        if overlap:
            dof_names = list(self._articulation_view.dof_names)
            overlap_names = [dof_names[index] if 0 <= index < len(dof_names) else str(index) for index in overlap]
            raise ValueError(
                "PandaOmron mobile base joint mapping overlaps manipulator joints: "
                f"indices={overlap}, names={overlap_names}"
            )

    def _capture_base_initial_steering_positions(self):
        if not self.base_steering_joint_indices:
            self._base_initial_steering_positions = np.zeros(0, dtype=np.float32)
            return
        joint_positions = self._articulation_view.get_joint_positions()[0]
        self._base_initial_steering_positions = np.asarray(
            joint_positions[self.base_steering_joint_indices],
            dtype=np.float32,
        ).copy()

    def _setup_mobile_base_interface(self):
        mobile_base_path = str(self.cfg["mobile_base_path"]).strip("/")
        self.mobile_base_prim_path = f"{self.robot_prim_path}/{mobile_base_path}"
        if not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            raise ValueError(f"PandaOmron mobile base prim does not exist: {self.mobile_base_prim_path}")

        wheel_links = list(self.base_cfg["wheel_link_names"])
        self._wheel_collision_paths = [
            f"{self.robot_prim_path}/robot0_base/{link_name}/collisions/{link_name}_collision"
            for link_name in wheel_links
        ]
        self._wheel_joint_paths = [
            f"{self.robot_prim_path}/robot0_base/joints/{joint_name}"
            for joint_name in self.base_wheel_joint_names
        ]
        self._wheel_physics_material_path = f"{self.robot_prim_path}/robot0_base/Looks/panda_omron_wheel_physics_material"

    def _demote_imported_site_visuals(self):
        return

    def _configure_mobile_base_wheel_drives(self):
        """Configure mobile-base drives in concrete subclasses."""

    def _configure_joint_drive(self, joint_name, drive_name, *, stiffness, damping, max_force):
        joint_path = self._resolve_joint_prim_path(joint_name)
        prim = get_prim_at_path(joint_path)
        if not prim.IsValid():
            raise ValueError(f"PandaOmron joint prim does not exist: {joint_path}")
        drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
        drive_api.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive_api.CreateStiffnessAttr().Set(float(stiffness))
        drive_api.CreateDampingAttr().Set(float(damping))
        drive_api.CreateMaxForceAttr().Set(float(max_force))
        drive_api.CreateTargetVelocityAttr().Set(0.0)

    def _resolve_joint_prim_path(self, joint_name):
        candidates = [
            f"{self.robot_prim_path}/robot0_base/joints/{joint_name}",
            f"{self.robot_prim_path}/robot0_base/panda_hand/{joint_name}",
        ]
        for joint_path in candidates:
            if get_prim_at_path(joint_path).IsValid():
                return joint_path
        raise ValueError(
            f"PandaOmron joint prim does not exist for {joint_name}; "
            f"checked={candidates}"
        )

    def _configure_manipulator_drives(self):
        arm_joint_names = list(self.cfg["left_joint_names"])
        gripper_joint_names = list(self.cfg["left_gripper_joint_names"])
        for joint_name in arm_joint_names:
            self._configure_joint_drive(
                joint_name,
                "angular",
                stiffness=self.cfg["arm_drive_stiffness"],
                damping=self.cfg["arm_drive_damping"],
                max_force=self.cfg["arm_drive_max_force"],
            )
        for joint_name in gripper_joint_names:
            self._configure_joint_drive(
                joint_name,
                "linear",
                stiffness=self.cfg["gripper_drive_stiffness"],
                damping=self.cfg["gripper_drive_damping"],
                max_force=self.cfg["gripper_drive_max_force"],
            )

    def get_base_initial_steering_positions(self):
        if self._base_initial_steering_positions is None:
            self._capture_base_initial_steering_positions()
        return np.asarray(self._base_initial_steering_positions, dtype=np.float32).copy()

    def get_base_interface(self):
        return {
            "steering_joint_names": list(self.base_steering_joint_names),
            "wheel_joint_names": list(self.base_wheel_joint_names),
            "steering_joint_indices": list(self.base_steering_joint_indices),
            "wheel_joint_indices": list(self.base_wheel_joint_indices),
            "mobile_base_prim_path": self.mobile_base_prim_path,
            "base_cfg": deepcopy(self.base_cfg),
        }

    def get_mobile_base_pose(self):
        return super().get_world_pose()

    def get_nav_base_pose(self):
        translation, orientation = self.get_mobile_base_pose()
        offset = np.asarray(
            self.base_cfg.get("wheel_center_to_base_translation", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(3)
        chassis_to_nav = tf_matrix_from_pose(offset, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        world_chassis = tf_matrix_from_pose(translation, orientation)
        world_nav = world_chassis @ chassis_to_nav
        nav_translation, nav_orientation = pose_from_tf_matrix(world_nav)
        return nav_translation.astype(np.float32), nav_orientation.astype(np.float32)

    def get_world_pose(self):
        return super().get_world_pose()

    def set_mobile_base_world_pose(self, translation, orientation):
        desired_translation = np.asarray(translation, dtype=np.float32).reshape(3)
        desired_orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(desired_translation)) or not np.all(np.isfinite(desired_orientation)):
            raise ValueError("PandaOmron mobile base pose must be finite")
        if float(np.linalg.norm(desired_orientation)) <= 1.0e-8:
            raise ValueError("PandaOmron mobile base orientation must be a non-zero quaternion")
        self.set_world_pose(position=desired_translation, orientation=desired_orientation)

    def reset_mobile_base_world_state(self, translation, orientation):
        self.set_mobile_base_world_pose(translation, orientation)
        self.set_world_velocity(np.zeros(6, dtype=np.float32))
        if getattr(self, "num_dof", 0):
            self._articulation_view.set_joint_velocities(
                np.zeros((1, int(self.num_dof)), dtype=np.float32),
            )
        self._set_initial_positions()
        self._configure_manipulator_drives()
        self._configure_mobile_base_wheel_drives()
        if self.base_wheel_joint_indices:
            wheel_joint_indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
            zero_wheel = np.zeros((1, len(self.base_wheel_joint_indices)), dtype=np.float32)
            self._articulation_view.set_joint_velocity_targets(
                zero_wheel,
                joint_indices=wheel_joint_indices,
            )
            self._articulation_view.set_joint_velocities(
                zero_wheel,
                joint_indices=wheel_joint_indices,
            )

    def apply_action(self, joint_positions, joint_indices, *args, **kwargs):
        joint_positions = np.asarray(joint_positions, dtype=np.float32).reshape(1, -1)
        joint_indices = np.asarray(joint_indices, dtype=np.int32).reshape(-1)
        if joint_positions.shape[1] != joint_indices.shape[0]:
            raise ValueError(
                "PandaOmron action position count must match joint index count: "
                f"positions={joint_positions.shape[1]}, indices={joint_indices.shape[0]}"
            )
        if not np.all(np.isfinite(joint_positions)):
            raise ValueError("PandaOmron action joint positions must be finite")

        base_indices = set(self.base_steering_joint_indices) | set(self.base_wheel_joint_indices)
        overlap = sorted(base_indices & {int(index) for index in joint_indices.tolist()})
        if overlap:
            dof_names = list(self._articulation_view.dof_names)
            overlap_names = [
                dof_names[index] if 0 <= int(index) < len(dof_names) else str(index)
                for index in overlap
            ]
            raise ValueError(
                "PandaOmron manipulator action must not target mobile-base joints: "
                f"indices={overlap}, names={overlap_names}"
            )

        self._articulation_view.set_joint_position_targets(
            joint_positions,
            joint_indices=joint_indices,
        )
        self._set_active_manipulator_position_target(joint_positions, joint_indices)

    def _set_active_manipulator_position_target(self, joint_positions, joint_indices):
        joint_positions = np.asarray(joint_positions, dtype=np.float32).reshape(1, -1)
        joint_indices = np.asarray(joint_indices, dtype=np.int32).reshape(-1)
        if joint_positions.shape[1] != joint_indices.shape[0]:
            raise ValueError(
                "Active manipulator target position count must match joint index count: "
                f"positions={joint_positions.shape[1]}, indices={joint_indices.shape[0]}"
            )
        base_indices = set(self.base_steering_joint_indices) | set(self.base_wheel_joint_indices)
        overlap = sorted(base_indices & {int(index) for index in joint_indices.tolist()})
        if overlap:
            dof_names = list(self._articulation_view.dof_names)
            overlap_names = [
                dof_names[index] if 0 <= int(index) < len(dof_names) else str(index)
                for index in overlap
            ]
            raise ValueError(
                "Active manipulator target must not include mobile-base joints: "
                f"indices={overlap}, names={overlap_names}"
            )
        self._active_manipulator_joint_positions = joint_positions.copy()
        self._active_manipulator_joint_indices = joint_indices.copy()

    def _reapply_active_manipulator_position_target(self):
        if self._active_manipulator_joint_positions is None or self._active_manipulator_joint_indices is None:
            return
        self._articulation_view.set_joint_position_targets(
            self._active_manipulator_joint_positions,
            joint_indices=self._active_manipulator_joint_indices,
        )

    def get_armbase_world_pose(self):
        mobile_translation, mobile_orientation = self.get_mobile_base_pose()
        world_mobile = tf_matrix_from_pose(mobile_translation, mobile_orientation)
        mobile_to_armbase = tf_matrix_from_pose(
            self._mobile_to_armbase_translation,
            self._mobile_to_armbase_orientation,
        )
        world_armbase = world_mobile @ mobile_to_armbase
        translation, orientation = pose_from_tf_matrix(world_armbase)
        return translation.astype(np.float32), orientation.astype(np.float32)

    def get_armbase_world_transform(self):
        translation, orientation = self.get_armbase_world_pose()
        return tf_matrix_from_pose(translation, orientation)

    def apply_base_command(self, steering_positions, wheel_velocities):
        steering_positions = np.asarray(steering_positions, dtype=np.float32).reshape(-1)
        wheel_velocities = np.asarray(wheel_velocities, dtype=np.float32).reshape(-1)
        if steering_positions.shape[0] != len(self.base_steering_joint_indices):
            raise ValueError("steering_positions size does not match steering joints")
        if wheel_velocities.shape[0] != len(self.base_wheel_joint_indices):
            raise ValueError("wheel_velocities size does not match wheel joints")
        if not np.all(np.isfinite(wheel_velocities)):
            raise ValueError("wheel_velocities must be finite")

        if self.base_steering_joint_indices:
            if not np.all(np.isfinite(steering_positions)):
                raise ValueError("steering_positions must be finite")
            self._articulation_view.set_joint_position_targets(
                steering_positions.reshape(1, -1),
                joint_indices=np.asarray(self.base_steering_joint_indices, dtype=np.int32),
            )
        self._articulation_view.set_joint_velocity_targets(
            wheel_velocities.reshape(1, -1),
            joint_indices=np.asarray(self.base_wheel_joint_indices, dtype=np.int32),
        )

    def get_base_joint_state(self):
        joint_positions = self._articulation_view.get_joint_positions()[0]
        joint_velocities = self._articulation_view.get_joint_velocities()[0]
        return {
            "steering_positions": joint_positions[self.base_steering_joint_indices].copy(),
            "wheel_positions": joint_positions[self.base_wheel_joint_indices].copy(),
            "steering_velocities": joint_velocities[self.base_steering_joint_indices].copy(),
            "wheel_velocities": joint_velocities[self.base_wheel_joint_indices].copy(),
        }

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _build_observations(self, qpos, qvel, T_base_ee_fl, T_world_base):
        from core.utils.transformation_utils import get_fk_solution, pose_to_6d

        gripper_pose = pose_to_6d(get_fk_solution(qpos[self.left_joint_indices]))
        gripper_q = np.asarray(qpos[self.left_gripper_indices], dtype=np.float32).reshape(-1)
        if gripper_q.size >= 2:
            gripper_position = np.asarray([float(gripper_q[0] - gripper_q[1])], dtype=np.float32)
        else:
            gripper_position = gripper_q * 2.0
        return {
            "states.joint.position": qpos[self.left_joint_indices],
            "states.gripper.position": gripper_position,
            "states.gripper.pose": gripper_pose,
            "qvel": qvel,
            "T_base_ee_fl": T_base_ee_fl,
            "T_world_base": T_world_base,
        }

    def get_observations(self) -> dict:
        joint_state = self.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        T_base_ee_fl = get_relative_transform(get_prim_at_path(self.fl_ee_path), get_prim_at_path(self.fl_base_path))
        obs = self._build_observations(qpos, qvel, T_base_ee_fl, self.get_armbase_world_transform())
        if not self.base_wheel_joint_indices:
            raise ValueError("PandaOmron base wheel joint indices are not initialized")

        driver = getattr(self, "_simbox_local_base_driver", None)
        if driver is None or not hasattr(driver, "get_logging_state_snapshot"):
            raise ValueError("PandaOmron requires an active local base driver for base observation logging")
        base_state = driver.get_logging_state_snapshot()

        obs["states.base.pose"] = np.asarray(base_state["pose"], dtype=np.float32)
        obs["states.base.twist_body"] = np.asarray(base_state["twist_body"], dtype=np.float32)
        obs["states.base.steering_positions"] = np.asarray(base_state["steering_positions"], dtype=np.float32)
        obs["states.base.wheel_positions"] = np.asarray(base_state["wheel_positions"], dtype=np.float32)
        obs["states.base.steering_velocities"] = np.asarray(base_state["steering_velocities"], dtype=np.float32)
        obs["states.base.wheel_velocities"] = np.asarray(base_state["wheel_velocities"], dtype=np.float32)
        return obs
