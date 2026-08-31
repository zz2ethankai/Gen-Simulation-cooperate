"""Unitree G1 articulation used by the decoupled-WBC locomotion pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from core.robots.base_robot import register_robot
from core.mobile.g1_decoupled_wbc import RobotState
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd


UNITREE_G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

UNITREE_G1_STAND_Q = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

UNITREE_G1_TORQUE_LIMITS = np.asarray(
    [
        139, 139, 88, 139, 25, 25,
        139, 139, 88, 139, 25, 25,
        88, 25, 25,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float64,
)

UNITREE_G1_WBC_KPS = np.asarray(
    [
        150, 150, 150, 200, 40, 40,
        150, 150, 150, 200, 40, 40,
        250, 250, 250,
        100, 100, 40, 40, 20, 20, 20,
        100, 100, 40, 40, 20, 20, 20,
    ],
    dtype=np.float64,
)

UNITREE_G1_WBC_KDS = np.asarray(
    [
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        5, 5, 5,
        5, 5, 2, 2, 2, 2, 2,
        5, 5, 2, 2, 2, 2, 2,
    ],
    dtype=np.float64,
)


@register_robot
class UnitreeG1(Robot):
    """29-body-DOF Unitree G1 with PhysX position-drive WBC control."""

    def __init__(self, asset_root: str, root_prim_path: str, cfg: dict, *args, **kwargs):
        self.asset_root = asset_root
        self.cfg = cfg
        self.base_cfg = deepcopy(cfg.get("base", {}))
        configured_path = Path(str(cfg["path"])).expanduser()
        if configured_path.is_absolute():
            usd_path = configured_path
        else:
            repo_path = Path(__file__).resolve().parents[4] / configured_path
            asset_path = Path(asset_root) / configured_path
            usd_path = repo_path if repo_path.is_file() else asset_path
        if not usd_path.is_file():
            raise FileNotFoundError(f"Unitree G1 USD not found: {usd_path}")
        prim_path = f"{root_prim_path}/{cfg['name']}"
        create_prim(usd_path=str(usd_path.resolve()), prim_path=prim_path)
        articulation_path = self._find_articulation_root(prim_path)
        super().__init__(articulation_path, cfg["name"], *args, **kwargs)
        self.robot_prim_path = prim_path
        self.articulation_prim_path = articulation_path
        self.body_joint_names = list(cfg.get("body_joint_names", UNITREE_G1_BODY_JOINT_NAMES))
        if self.body_joint_names != list(UNITREE_G1_BODY_JOINT_NAMES):
            raise ValueError("Unitree G1 body_joint_names must use the HUMANO 29-DOF order")
        self.body_joint_indices: list[int] = []
        self._torque_limits = np.asarray(
            cfg.get("body_torque_limits", UNITREE_G1_TORQUE_LIMITS), dtype=np.float64
        ).reshape(-1)
        if self._torque_limits.shape != (29,) or np.any(self._torque_limits <= 0.0):
            raise ValueError("Unitree G1 body_torque_limits must contain 29 positive values")
        self._drive_kps = UNITREE_G1_WBC_KPS.copy()
        self._drive_kds = UNITREE_G1_WBC_KDS.copy()
        self.set_solver_position_iteration_count(int(cfg.get("solver_position_iteration_count", 8)))
        self.set_solver_velocity_iteration_count(int(cfg.get("solver_velocity_iteration_count", 1)))
        self.set_stabilization_threshold(float(cfg.get("stabilization_threshold", 0.005)))

    @staticmethod
    def _find_articulation_root(wrapper_path: str) -> str:
        wrapper = get_prim_at_path(wrapper_path)
        if wrapper.IsValid():
            for prim in Usd.PrimRange(wrapper):
                if any("ArticulationRootAPI" in schema for schema in prim.GetAppliedSchemas()):
                    return str(prim.GetPath())
        return wrapper_path

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._articulation_view.initialize()
        runtime_names = list(self.dof_names)
        missing = [name for name in self.body_joint_names if name not in runtime_names]
        if missing:
            raise ValueError(f"Unitree G1 USD is missing body joints: {missing}")
        self.body_joint_indices = [runtime_names.index(name) for name in self.body_joint_names]
        indices = np.asarray(self.body_joint_indices, dtype=np.int32)
        zeros = np.zeros((1, 29), dtype=np.float32)
        self._articulation_view.set_gains(
            kps=self._drive_kps.astype(np.float32).reshape(1, 29),
            kds=self._drive_kds.astype(np.float32).reshape(1, 29),
            joint_indices=indices,
        )
        self._articulation_view.set_max_efforts(
            self._torque_limits.astype(np.float32).reshape(1, 29), joint_indices=indices
        )
        self._articulation_view.set_joint_positions(
            UNITREE_G1_STAND_Q.astype(np.float32).reshape(1, 29), joint_indices=indices
        )
        self._articulation_view.set_joint_velocities(zeros, joint_indices=indices)

    def get_base_interface(self):
        return {
            "steering_joint_names": [],
            "wheel_joint_names": [],
            "steering_joint_indices": [],
            "wheel_joint_indices": [],
            "base_cfg": self.base_cfg,
        }

    def get_nav_base_pose(self):
        return self.get_world_pose()

    def set_mobile_base_world_pose(self, translation, orientation):
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(orientation)):
            raise ValueError("Unitree G1 mobile base pose must be finite")
        if float(np.linalg.norm(orientation)) <= 1.0e-8:
            raise ValueError("Unitree G1 mobile base orientation must be a non-zero quaternion")
        self.set_world_pose(position=translation, orientation=orientation)

    def reset_mobile_base_world_state(self, translation, orientation):
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(orientation)):
            raise ValueError("Unitree G1 reset pose must be finite")
        if float(np.linalg.norm(orientation)) <= 1.0e-8:
            raise ValueError("Unitree G1 reset orientation must be a non-zero quaternion")
        joint_state = self.get_joints_state()
        joint_positions = np.asarray(
            joint_state.positions, dtype=np.float32
        ).reshape(1, -1).copy()
        body_indices = np.asarray(self.body_joint_indices, dtype=np.int64)
        joint_positions[:, body_indices] = UNITREE_G1_STAND_Q.astype(np.float32)
        self.set_mobile_base_world_pose(translation, orientation)
        self.set_world_velocity(np.zeros(6, dtype=np.float32))
        self._articulation_view.set_joint_positions(joint_positions)
        self._articulation_view.set_joint_velocities(np.zeros_like(joint_positions))

    def get_locomotion_state(self) -> RobotState:
        joint_state = self.get_joints_state()
        position, orientation = self.get_world_pose()
        angular_velocity_world = np.asarray(self.get_angular_velocity(), dtype=np.float64).reshape(3)
        angular_velocity_body = self._world_vector_to_body(angular_velocity_world, orientation)
        indices = np.asarray(self.body_joint_indices, dtype=np.int64)
        return RobotState(
            body_q=np.asarray(joint_state.positions, dtype=np.float64)[indices],
            body_dq=np.asarray(joint_state.velocities, dtype=np.float64)[indices],
            base_quat=np.asarray(orientation, dtype=np.float64).reshape(4),
            base_ang_vel=angular_velocity_body,
            pelvis_z=float(position[2]),
        )

    @staticmethod
    def _world_vector_to_body(vector, quaternion):
        w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return rotation.T @ np.asarray(vector, dtype=np.float64)

    def apply_locomotion_command(self, command):
        state = self.get_locomotion_state()
        efforts = (
            np.asarray(command.tau_ff, dtype=np.float64)
            + self._drive_kps * (command.q_target - state.body_q)
            + self._drive_kds * (command.dq_target - state.body_dq)
        )
        efforts = np.clip(efforts, -self._torque_limits, self._torque_limits)
        self.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(command.q_target, dtype=np.float32),
                joint_indices=np.asarray(self.body_joint_indices, dtype=np.int32),
            )
        )
        return efforts

    def get_observations(self):
        state = self.get_locomotion_state()
        position, orientation = self.get_world_pose()
        return {
            "states.body_joint.position": state.body_q,
            "states.body_joint.velocity": state.body_dq,
            "states.base.position": np.asarray(position, dtype=np.float64),
            "states.base.orientation": np.asarray(orientation, dtype=np.float64),
            "qvel": np.concatenate([state.base_ang_vel, state.body_dq]),
        }
