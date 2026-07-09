"""PandaOmron variant with a virtual holonomic mobile base."""

from __future__ import annotations

import math

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.panda_omron import PandaOmron
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.xforms import get_world_pose as get_prim_world_pose
from pxr import PhysxSchema, UsdPhysics


@register_robot
class PandaOmronVirtual(PandaOmron):
    """Panda on Omron geometry driven by X/Y/yaw virtual base joints."""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._reset_virtual_base_joint_state(require_ready=True)

    def _setup_mobile_base_interface(self):
        if self.base_steering_joint_names:
            raise ValueError("PandaOmronVirtual does not support steering joints")
        if len(self.base_wheel_joint_names) != 3:
            raise ValueError("PandaOmronVirtual requires exactly 3 base velocity joints")

        mobile_base_path = str(self.cfg["mobile_base_path"]).strip("/")
        self.mobile_base_prim_path = f"{self.robot_prim_path}/{mobile_base_path}"
        if not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            raise ValueError(f"PandaOmronVirtual mobile base prim does not exist: {self.mobile_base_prim_path}")

        self._wheel_collision_paths = []
        self._wheel_joint_paths = [
            f"{self.robot_prim_path}/robot0_base/joints/{joint_name}"
            for joint_name in self.base_wheel_joint_names
        ]
        self._wheel_physics_material_path = None

    def _configure_mobile_base_wheel_drives(self):
        drive_stiffness, dampings, max_forces, max_velocities = self._virtual_base_drive_parameters()

        drive_specs = (
            ("linear", dampings[0], max_forces[0], max_velocities[0]),
            ("linear", dampings[1], max_forces[1], max_velocities[1]),
            ("angular", dampings[2], max_forces[2], max_velocities[2]),
        )
        for joint_path, (drive_name, damping, max_force, max_velocity) in zip(self._wheel_joint_paths, drive_specs):
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                raise ValueError(f"PandaOmronVirtual base joint prim does not exist: {joint_path}")
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
            if not drive_api:
                raise ValueError(f"PandaOmronVirtual base joint is missing {drive_name} drive: {joint_path}")
            drive_api.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
            drive_api.CreateStiffnessAttr().Set(drive_stiffness)
            drive_api.CreateDampingAttr().Set(float(damping))
            drive_api.CreateMaxForceAttr().Set(float(max_force))
            drive_api.CreateTargetVelocityAttr().Set(0.0)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(float(max_velocity))

        self._configure_virtual_base_runtime_drives(require_ready=False)

    def _virtual_base_drive_parameters(self):
        drive_stiffness = float(self.base_cfg["virtual_drive_stiffness"])
        if not math.isclose(drive_stiffness, 0.0, abs_tol=1.0e-8):
            raise ValueError("PandaOmronVirtual velocity drives require virtual_drive_stiffness: 0.0")

        linear_drive_damping = float(self.base_cfg["virtual_linear_drive_damping"])
        linear_drive_max_force = float(self.base_cfg["virtual_linear_drive_max_force"])
        yaw_drive_damping = float(self.base_cfg["virtual_yaw_drive_damping"])
        yaw_drive_max_force = float(self.base_cfg["virtual_yaw_drive_max_force"])
        hard_limits = self.base_cfg["platform"]["nav2"]["controller_hard_limits"]
        max_velocity = np.asarray(hard_limits["max_velocity"], dtype=np.float32).reshape(-1)
        min_velocity = np.asarray(hard_limits["min_velocity"], dtype=np.float32).reshape(-1)
        if max_velocity.size != 3 or min_velocity.size != 3:
            raise ValueError("PandaOmronVirtual controller_hard_limits velocity limits must be 3-element lists")

        dampings = np.asarray(
            [linear_drive_damping, linear_drive_damping, yaw_drive_damping],
            dtype=np.float32,
        )
        max_forces = np.asarray(
            [linear_drive_max_force, linear_drive_max_force, yaw_drive_max_force],
            dtype=np.float32,
        )
        max_velocities = np.maximum(np.abs(max_velocity), np.abs(min_velocity)).astype(np.float32)
        if (
            not np.all(np.isfinite(dampings))
            or not np.all(np.isfinite(max_forces))
            or not np.all(np.isfinite(max_velocities))
            or np.any(dampings <= 0.0)
            or np.any(max_forces <= 0.0)
            or np.any(max_velocities <= 0.0)
        ):
            raise ValueError("PandaOmronVirtual velocity-drive damping, force, and velocity limits must be positive finite values")
        return drive_stiffness, dampings, max_forces, max_velocities

    def _configure_virtual_base_runtime_drives(self, *, require_ready: bool):
        if len(self.base_wheel_joint_indices) != 3:
            if require_ready:
                raise ValueError("PandaOmronVirtual base joint indices are not initialized")
            return
        articulation_view = getattr(self, "_articulation_view", None)
        if articulation_view is None or not getattr(articulation_view, "_is_initialized", False):
            if require_ready:
                raise ValueError("PandaOmronVirtual articulation view is not initialized")
            return
        if not articulation_view.is_physics_handle_valid():
            if require_ready:
                raise ValueError("PandaOmronVirtual physics handle is not valid")
            return

        _, dampings, max_forces, max_velocities = self._virtual_base_drive_parameters()
        base_joint_indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
        zeros = np.zeros((1, 3), dtype=np.float32)
        articulation_view.set_max_joint_velocities(
            max_velocities.reshape(1, 3),
            joint_indices=base_joint_indices,
        )
        articulation_view.set_max_efforts(
            max_forces.reshape(1, 3),
            joint_indices=base_joint_indices,
        )
        articulation_view.set_gains(
            kps=zeros,
            kds=dampings.reshape(1, 3),
            joint_indices=base_joint_indices,
        )
        articulation_view.set_joint_velocity_targets(
            zeros,
            joint_indices=base_joint_indices,
        )

    def _reset_virtual_base_joint_state(self, *, require_ready: bool):
        if len(self.base_wheel_joint_indices) != 3:
            if require_ready:
                raise ValueError("PandaOmronVirtual base joint indices are not initialized")
            return
        base_joint_indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
        zero = np.zeros((1, 3), dtype=np.float32)
        self._articulation_view.set_joint_positions(zero, joint_indices=base_joint_indices)
        self._articulation_view.set_joint_velocities(zero, joint_indices=base_joint_indices)
        self._configure_virtual_base_runtime_drives(require_ready=require_ready)

    def get_mobile_base_pose(self):
        if not self.mobile_base_prim_path or not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            raise ValueError("PandaOmronVirtual mobile base prim is not initialized")
        return get_prim_world_pose(self.mobile_base_prim_path)

    def get_world_pose(self):
        if not self.mobile_base_prim_path or not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            return super().get_world_pose()
        return self.get_mobile_base_pose()

    def set_mobile_base_world_pose(self, translation, orientation):
        desired_translation = np.asarray(translation, dtype=np.float32).reshape(3)
        desired_orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(desired_translation)) or not np.all(np.isfinite(desired_orientation)):
            raise ValueError("PandaOmronVirtual mobile base pose must be finite")
        if float(np.linalg.norm(desired_orientation)) <= 1.0e-8:
            raise ValueError("PandaOmronVirtual mobile base orientation must be a non-zero quaternion")
        self._reset_virtual_base_joint_state(require_ready=False)
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
        self._reset_virtual_base_joint_state(require_ready=True)
        self._configure_virtual_base_runtime_drives(require_ready=True)

    def apply_base_command(self, steering_positions, wheel_velocities, *, step_dt: float):
        steering_positions = np.asarray(steering_positions, dtype=np.float32).reshape(-1)
        body_velocity = np.asarray(wheel_velocities, dtype=np.float32).reshape(-1)
        if steering_positions.size != 0:
            raise ValueError("PandaOmronVirtual does not accept steering commands")
        if body_velocity.size != 3:
            raise ValueError("PandaOmronVirtual requires exactly three virtual base velocity commands")
        if not np.all(np.isfinite(body_velocity)):
            raise ValueError("PandaOmronVirtual base velocity commands must be finite")
        dt = float(step_dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("PandaOmronVirtual step_dt must be a finite non-negative value")

        base_joint_indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
        if base_joint_indices.size != 3:
            raise ValueError("PandaOmronVirtual base joint indices are not initialized")

        joint_positions = self._articulation_view.get_joint_positions()[0]
        virtual_positions = np.asarray(joint_positions[self.base_wheel_joint_indices], dtype=np.float32).reshape(3)
        yaw_joint = float(virtual_positions[2])
        cos_yaw = math.cos(yaw_joint)
        sin_yaw = math.sin(yaw_joint)
        vx_body = float(body_velocity[0])
        vy_body = float(body_velocity[1])
        wz_body = float(body_velocity[2])
        world_velocity = np.asarray(
            [
                cos_yaw * vx_body - sin_yaw * vy_body,
                sin_yaw * vx_body + cos_yaw * vy_body,
                wz_body,
            ],
            dtype=np.float32,
        )
        world_velocity = world_velocity.reshape(1, -1)
        self._articulation_view.set_joint_velocity_targets(
            world_velocity,
            joint_indices=base_joint_indices,
        )
        self._reapply_active_manipulator_position_target()

    @staticmethod
    def _wrap_angle(angle: float):
        return math.atan2(math.sin(angle), math.cos(angle))
