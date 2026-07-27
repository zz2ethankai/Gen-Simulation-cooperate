"""
Template robot class for manipulator robots with configurable parameters.
All robot implementations (FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha) inherit from this class.
"""
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from core.robots.base_robot import register_robot
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)
from scipy.interpolate import interp1d

from core.utils.joint_index_resolver import (
    JOINT_GROUP_FIELDS,
    resolve_configured_joint_groups,
    resolve_joint_names,
)


LOGGER = logging.getLogger("de_logger")


# pylint: disable=line-too-long,unused-argument
@register_robot
class TemplateRobot(Robot):
    """
    Template class for manipulator robots.

    All important parameters should be prepared in cfg before instantiation.
    The cfg is merged from: robot_config_file -> task_config_robots
    """

    def __init__(self, asset_root: str, root_prim_path: str, cfg: dict, *args, **kwargs):
        self.asset_root = asset_root
        self.cfg = cfg

        # Create prim
        path_value = cfg["path"]
        usd_path = f"{asset_root}/{path_value}"
        if not Path(usd_path).is_file():
            alt = str(Path(path_value))
            if Path(alt).is_file():
                usd_path = alt
                usd_path = str(Path(usd_path).resolve())
        prim_path = f"{root_prim_path}/{cfg['name']}"
        create_prim(usd_path=usd_path, prim_path=prim_path)
        super().__init__(prim_path, cfg["name"], *args, **kwargs)

        self.robot_prim_path = prim_path

        # Gripper parameters (from cfg, no .get())
        self.gripper_max_width = cfg["gripper_max_width"]
        self.gripper_min_width = cfg["gripper_min_width"]

        # Solver parameters
        self.set_solver_position_iteration_count(cfg["solver_position_iteration_count"])
        self.set_stabilization_threshold(cfg["stabilization_threshold"])
        self.set_solver_velocity_iteration_count(cfg["solver_velocity_iteration_count"])

        # Setup methods (subclass implements)
        self._setup_joint_indices()
        self._setup_paths()
        self._setup_gripper_keypoints()
        self._setup_collision_paths()
        self._load_extra_depth(usd_path)
        self._manipulation_base_hold_active = False
        self._manipulation_base_hold_indices = np.asarray([], dtype=np.int64)
        self._manipulation_base_hold_positions = np.asarray([], dtype=float)

    def _setup_joint_indices(self):
        """Setup joint indices. Override in subclass."""
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg.get("right_joint_indices", [])
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg.get("right_gripper_indices", [])
        self.body_indices = self.cfg.get("body_indices", [])
        self.head_indices = self.cfg.get("head_indices", [])
        self.lift_indices = self.cfg.get("lift_indices", [])

    def _setup_paths(self):
        """Setup robot paths. Override in subclass."""
        fl_ee_path = self.cfg["fl_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fl_hand_path = self.fl_ee_path

        fr_ee_path = self.cfg.get("fr_ee_path", "")
        self.fr_ee_path = f"{self.robot_prim_path}/{fr_ee_path}" if fr_ee_path else ""
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}" if self.cfg.get("fr_base_path") else ""
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        """Setup gripper keypoints. Override in subclass."""
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg.get("fr_gripper_keypoints", self.fl_gripper_keypoints)

    def _setup_collision_paths(self):
        """Setup collision paths. Override in subclass."""
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg.get("fr_filter_paths", [])]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [
            f"{self.robot_prim_path}/{p}" for p in self.cfg.get("fr_forbid_collision_paths", [])
        ]

    def _load_extra_depth(self, usd_path: str):
        """Load extra depth function from JSON file."""
        extra_depth_file = self.cfg.get("extra_depth_file")
        if extra_depth_file:
            json_path = usd_path.replace("robot.usd", extra_depth_file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_dict = json.load(f)
                keys = list(json_dict.keys())
                widths = np.array([json_dict[key]["width"] for key in keys[1:]])
                extra_depths = np.array([json_dict[key]["offset"] for key in keys[1:]])
                self._gripper_ed_func = interp1d(widths, extra_depths, kind="cubic")
                self.gripper_max_width = widths.max()
                self.gripper_min_width = widths.min()
            except Exception:
                self._gripper_ed_func = None
        else:
            self._gripper_ed_func = None

    def initialize(self, *args, **kwargs):
        super().initialize()
        self._articulation_view.initialize()
        self._resolve_runtime_joint_indices()
        self._setup_joint_velocities()
        self._setup_joint_homes()
        self._set_initial_positions()

    def _resolve_runtime_joint_indices(self):
        """Make configured joint names authoritative over asset-specific numeric indices."""

        runtime_names = list(self.dof_names)
        resolved_groups = resolve_configured_joint_groups(runtime_names, self.cfg)
        for group, (indices_field, _) in JOINT_GROUP_FIELDS.items():
            resolved = resolved_groups[group]
            configured = list(self.cfg.get(indices_field, []))
            if configured != resolved:
                LOGGER.warning(
                    "[JointIndex] robot=%s corrected %s from %s to %s using runtime dof_names",
                    self.name,
                    indices_field,
                    configured,
                    resolved,
                )
            setattr(self, indices_field, resolved)
            self.cfg[indices_field] = resolved

    def _setup_joint_velocities(self):
        all_joint_indices = (
            self.body_indices
            + self.head_indices
            + self.lift_indices
            + self.left_joint_indices
            + self.right_joint_indices
        )
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                np.array([500.0] * len(all_joint_indices)),
                joint_indices=np.array(all_joint_indices),
            )

    def _setup_joint_homes(self):
        # Get joint home positions from config
        self.left_joint_home = self.cfg["left_joint_home"]
        self.right_joint_home = self.cfg.get("right_joint_home", [])
        self.left_gripper_home = self.cfg["left_gripper_home"]
        self.right_gripper_home = self.cfg.get("right_gripper_home", [])
        self.body_home = self.cfg.get("body_home", [0.0] * len(self.body_indices))
        self.head_home = self.cfg.get("head_home", [0.0] * len(self.head_indices))
        self.lift_home = self.cfg.get("lift_home", [0.0] * len(self.lift_indices))
        self.tcp_offset = self.cfg["tcp_offset"]

        # Apply noise from *_std parameters
        left_joint_home_std = self.cfg.get("left_joint_home_std", [0.0] * len(self.left_joint_home))
        right_joint_home_std = self.cfg.get("right_joint_home_std", [0.0] * len(self.right_joint_home))

        left_noise = np.random.normal(0, left_joint_home_std)
        self.left_joint_home = (np.array(self.left_joint_home) + left_noise).tolist()

        if self.right_joint_home:
            right_noise = np.random.normal(0, right_joint_home_std)
            self.right_joint_home = (np.array(self.right_joint_home) + right_noise).tolist()

        # Gripper state
        self.left_gripper_state = self._get_gripper_state(self.left_gripper_home)
        self.right_gripper_state = self._get_gripper_state(self.right_gripper_home) if self.right_gripper_home else 1.0

    def _get_gripper_state(self, gripper_home: list) -> float:
        return 1.0 if gripper_home and gripper_home[0] > 0 else -1.0

    def _set_initial_positions(self):
        positions = (
            self.body_home
            + self.head_home
            + self.lift_home
            + self.left_joint_home
            + self.right_joint_home
            + self.left_gripper_home
            + self.right_gripper_home
        )
        indices = (
            self.body_indices
            + self.head_indices
            + self.lift_indices
            + self.left_joint_indices
            + self.right_joint_indices
            + self.left_gripper_indices
            + self.right_gripper_indices
        )
        if positions and indices:
            self._articulation_view.set_joint_positions(
                np.array(positions).reshape(1, -1),
                joint_indices=np.array(indices),
            )

    def apply_action(self, joint_positions, joint_indices, *args, **kwargs):
        positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        indices = np.asarray(joint_indices, dtype=np.int64).reshape(-1)
        if self._manipulation_base_hold_active:
            supplied = set(indices.tolist())
            missing = [
                (index, position)
                for index, position in zip(
                    self._manipulation_base_hold_indices,
                    self._manipulation_base_hold_positions,
                )
                if int(index) not in supplied
            ]
            if missing:
                indices = np.concatenate(
                    [indices, np.asarray([item[0] for item in missing], dtype=np.int64)]
                )
                positions = np.concatenate(
                    [positions, np.asarray([item[1] for item in missing], dtype=float)]
                )
        self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)

    def enable_manipulation_base_hold(self) -> None:
        """Hold explicitly configured mobile-base DOFs during Pick/Place.

        SplitAloha's planar base joints intentionally have zero drive gain in
        the delivered USD because navigation controls them.  A manipulation
        episode, however, must not let arm reaction/contact forces move those
        joints while CuRobo is executing in an arm-base frame.  Joint names in
        robot config are the contract; no asset-name or numeric-index guessing
        is used here.
        """

        config = self.cfg.get("manipulation_base_hold", {})
        if not config or not bool(config.get("enabled", False)):
            return
        joint_names = list(config.get("joint_names", []))
        if not joint_names:
            raise ValueError(
                f"robot {self.name} enables manipulation_base_hold without joint_names"
            )
        indices = np.asarray(
            resolve_joint_names(
                list(self.dof_names),
                joint_names,
                group=f"{self.name}.manipulation_base_hold",
            ),
            dtype=np.int64,
        )
        joint_state = self.get_joints_state()
        positions = np.asarray(joint_state.positions[indices], dtype=float).copy()

        articulation_controller = self.get_articulation_controller()
        kps, kds = articulation_controller.get_gains()
        max_efforts = articulation_controller.get_max_efforts()
        kps = np.asarray(kps, dtype=float).copy()
        kds = np.asarray(kds, dtype=float).copy()
        if max_efforts is None:
            raise RuntimeError(f"robot {self.name} does not expose articulation max efforts")
        max_efforts = np.asarray(max_efforts, dtype=float).copy()

        stiffness = float(config.get("stiffness", 100000.0))
        damping = float(config.get("damping", 3000.0))
        max_effort = float(config.get("max_effort", 10000.0))
        kps[indices] = stiffness
        kds[indices] = damping
        max_efforts[indices] = max_effort
        articulation_controller.set_gains(kps=kps, kds=kds)
        articulation_controller.set_max_efforts(max_efforts)
        self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)
        self._articulation_view.set_joint_velocity_targets(
            np.zeros_like(positions), joint_indices=indices
        )

        self._manipulation_base_hold_indices = indices
        self._manipulation_base_hold_positions = positions
        self._manipulation_base_hold_active = True
        LOGGER.warning(
            "[BaseHold] robot=%s joints=%s indices=%s targets=%s kp=%.1f kd=%.1f max_effort=%.1f",
            self.name,
            joint_names,
            indices.tolist(),
            np.round(positions, 6).tolist(),
            stiffness,
            damping,
            max_effort,
        )

    def get_observations(self) -> dict:
        joint_state = self.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities

        T_base_ee_fl = get_relative_transform(get_prim_at_path(self.fl_ee_path), get_prim_at_path(self.fl_base_path))
        T_world_base = tf_matrix_from_pose(*self.get_local_pose())

        obs = self._build_observations(qpos, qvel, T_base_ee_fl, T_world_base)
        return obs

    def _build_observations(self, qpos, qvel, T_base_ee_fl, T_world_base):
        obs = {
            "states.left_joint.position": qpos[self.left_joint_indices],
            "states.left_gripper.position": qpos[self.left_gripper_indices] * 2,
            "qvel": qvel,
            "T_base_ee_fl": T_base_ee_fl,
            "T_world_base": T_world_base,
        }
        if self.right_joint_indices:
            T_base_ee_fr = get_relative_transform(
                get_prim_at_path(self.fr_ee_path), get_prim_at_path(self.fr_base_path)
            )
            obs["states.right_joint.position"] = qpos[self.right_joint_indices]
            obs["states.right_gripper.position"] = qpos[self.right_gripper_indices] * 2
            obs["T_base_ee_fr"] = T_base_ee_fr
        return obs

    def _get_R_ee_graspnet(self) -> np.ndarray:
        return self.cfg["R_ee_graspnet"]

    def _get_ee_axis(self) -> str:
        return self.cfg["ee_axis"]

    def pose_post_process_fn(
        self, poses, *args, lr_arm="left", grasp_scale=1, tcp_offset=None, constraints=None, **kwargs
    ):
        if poses.shape[-2:] == (4, 4):
            return poses

        R_ee_graspnet = self._get_R_ee_graspnet()
        n_grasps = poses.shape[0]
        T_obj_tcp = np.repeat(np.eye(4)[np.newaxis, :, :], n_grasps, axis=0)
        R_ee_graspnet = np.array(R_ee_graspnet)
        T_obj_tcp[:, :3, :3] = np.matmul(poses[:, 4:13].reshape(-1, 3, 3), R_ee_graspnet.T)
        T_obj_tcp[:, :3, 3] = poses[:, 13:16] * grasp_scale
        scores = poses[:, 0]
        widths = np.clip(poses[:, 1:2], self.gripper_min_width, self.gripper_max_width)
        depths = poses[:, 3:4]

        if tcp_offset is None:
            tcp_offset = self.tcp_offset

        if self._gripper_ed_func is not None:
            depths = depths + self._gripper_ed_func(widths)

        T_obj_ee = self._calculate_ee_position(T_obj_tcp, depths, tcp_offset)

        if constraints is not None:
            T_obj_ee, scores = self._apply_constraints(T_obj_tcp, T_obj_ee, scores, constraints)

        T_obj_ee_variant = self._apply_rotation_variant(T_obj_ee)
        return np.concatenate([T_obj_ee, T_obj_ee_variant], axis=0), np.concatenate([scores, scores], axis=0)

    def _calculate_ee_position(self, T_obj_tcp, depths, tcp_offset):
        tcp_center = T_obj_tcp[:, 0:3, 3]
        ee_axis = self._get_ee_axis()
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map[ee_axis]
        axis = T_obj_tcp[:, 0:3, axis_idx]
        ee_center = tcp_center + axis * (depths - tcp_offset)
        T_obj_ee = T_obj_tcp.copy()
        T_obj_ee[:, 0:3, 3] = ee_center
        return T_obj_ee

    def _apply_constraints(self, T_obj_tcp, T_obj_ee, scores, constraints):
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis, min_ratio, max_ratio = constraints
        idx = axis_map[axis]
        max_pose, min_pose = max(T_obj_tcp[:, idx, 3]), min(T_obj_tcp[:, idx, 3])
        min_th = min_pose + min_ratio * (max_pose - min_pose)
        max_th = min_pose + max_ratio * (max_pose - min_pose)
        flag = (T_obj_tcp[:, idx, 3] >= min_th) & (T_obj_tcp[:, idx, 3] <= max_th)
        return T_obj_ee[flag], scores[flag]

    def _apply_rotation_variant(self, T_obj_ee):
        T_obj_ee_variant = deepcopy(T_obj_ee)
        ee_axis = self._get_ee_axis()
        # Rotation matrix for 180 degree rotation around the gripper axis
        if ee_axis == "x":
            # Rotate 180 degrees around X axis
            rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape((1, 4, 4))
        elif ee_axis == "y":
            # Rotate 180 degrees around Y axis
            rot = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape((1, 4, 4))
        else:  # z
            # Rotate 180 degrees around Z axis
            rot = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).reshape((1, 4, 4))
        return np.matmul(T_obj_ee_variant, rot)
