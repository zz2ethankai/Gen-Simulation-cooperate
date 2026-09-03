"""
Template robot class for manipulator robots with configurable parameters.
All robot implementations (FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha) inherit from this class.
"""
import json
import logging
import os
import time
import traceback
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from core.robots.base_robot import register_robot
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.prims import create_prim, get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.xforms import get_world_pose as get_prim_world_pose
from scipy.interpolate import interp1d
from pxr import Usd

from core.utils.joint_index_resolver import (
    JOINT_GROUP_FIELDS,
    resolve_configured_joint_groups,
    resolve_joint_names,
)
from core.runtime import BaseHoldStrategy


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
        usd_path = self._resolve_usd_path(asset_root, cfg["path"])
        prim_path = f"{root_prim_path}/{cfg['name']}"
        create_prim(usd_path=usd_path, prim_path=prim_path)
        articulation_prim_path = self._find_articulation_root(prim_path)
        super().__init__(articulation_prim_path, cfg["name"], *args, **kwargs)

        # The articulation view must target the USD articulation root.  Keep
        # robot_prim_path at the wrapper so existing asset-relative paths stay
        # stable for robots whose root is a child prim.
        self.robot_prim_path = prim_path
        self.articulation_prim_path = articulation_prim_path

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
        self._manipulation_base_hold_saved_drive = None
        self._base_hold_strategy = None

    @staticmethod
    def _resolve_usd_path(asset_root: str, path: str) -> str:
        """Resolve both asset-root-relative and repository-relative robot paths."""

        configured = Path(str(path)).expanduser()
        if configured.is_absolute() and configured.exists():
            return str(configured.resolve())
        rooted = Path(asset_root).expanduser() / configured
        if rooted.exists():
            return str(rooted.resolve())
        if configured.exists():
            return str(configured.resolve())

        # Scene-4 task YAMLs were generated with paths such as
        # ``../../../../example_assets/...``.  Those paths were meaningful in
        # the asset-generation checkout, but the checked-in robot USDs live
        # under ``workflows/simbox/example_assets`` in the SimBox container.
        repo_root = Path(__file__).resolve().parents[4]
        repo_relative = str(configured)
        while repo_relative.startswith("../"):
            repo_relative = repo_relative[3:]
        packaged_asset = repo_root / "workflows" / "simbox" / repo_relative
        if packaged_asset.exists():
            return str(packaged_asset.resolve())
        return str(rooted.resolve())

    @staticmethod
    def _find_articulation_root(wrapper_path: str) -> str:
        """Use the applied USD articulation root when the asset has a wrapper prim."""

        wrapper = get_prim_at_path(wrapper_path)
        if not wrapper.IsValid():
            return wrapper_path
        for prim in Usd.PrimRange(wrapper):
            if any("ArticulationRootAPI" in schema for schema in prim.GetAppliedSchemas()):
                return str(prim.GetPath())
        return wrapper_path

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

    def get_armbase_world_pose(self, arm="left"):
        """Return the active arm-base pose in the Isaac world frame."""
        if arm == "left":
            base_path = self.fl_base_path
        elif arm == "right":
            base_path = self.fr_base_path
        else:
            raise ValueError(f"unsupported arm {arm!r}")
        translation, orientation = get_prim_world_pose(base_path)
        return np.asarray(translation, dtype=np.float32), np.asarray(orientation, dtype=np.float32)

    def get_armbase_world_transform(self, arm="left"):
        """Return the active arm-base transform in the Isaac world frame."""
        translation, orientation = self.get_armbase_world_pose(arm)
        return tf_matrix_from_pose(translation, orientation)

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
        # Robot.initialize() owns articulation-view initialization in Isaac 6.
        # Forward the active physics view instead of creating a second implicit
        # view, and do not initialize _articulation_view a second time.
        super().initialize(*args, **kwargs)
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

    def set_mobile_base_world_pose(self, translation, orientation):
        """Place the robot so its mobile base frame matches the requested world pose."""
        if not hasattr(self, "get_mobile_base_pose"):
            raise AttributeError(f"{type(self).__name__} must define get_mobile_base_pose()")

        root_translation, root_orientation = self.get_world_pose()
        mobile_translation, mobile_orientation = self.get_mobile_base_pose()
        world_root = tf_matrix_from_pose(root_translation, root_orientation)
        world_mobile = tf_matrix_from_pose(mobile_translation, mobile_orientation)
        root_mobile = np.linalg.inv(world_root) @ world_mobile

        desired_mobile = tf_matrix_from_pose(
            np.asarray(translation, dtype=np.float32),
            np.asarray(orientation, dtype=np.float32),
        )
        desired_root = desired_mobile @ np.linalg.inv(root_mobile)
        desired_translation, desired_orientation = pose_from_tf_matrix(desired_root)
        self.set_world_pose(position=desired_translation, orientation=desired_orientation)

    def reset_mobile_base_world_state(self, translation, orientation):
        """Reset the mobile base pose and clear physical base velocities."""
        self.set_mobile_base_world_pose(translation, orientation)
        self.set_world_velocity(np.zeros(6, dtype=np.float32))
        if getattr(self, "num_dof", 0):
            zero_dof = np.zeros((1, int(self.num_dof)), dtype=np.float32)
            self._articulation_view.set_joint_velocities(zero_dof)

        base_interface = self.get_base_interface()
        steering_indices = list(base_interface["steering_joint_indices"])
        wheel_indices = list(base_interface["wheel_joint_indices"])

        if steering_indices:
            target_steering = np.asarray(
                self.get_base_initial_steering_positions(),
                dtype=np.float32,
            ).reshape(-1)
            if target_steering.size != len(steering_indices) or not np.all(np.isfinite(target_steering)):
                raise ValueError("Initial steering positions must match steering joints and be finite")
            steering_joint_indices = np.asarray(steering_indices, dtype=np.int32)
            self._articulation_view.set_joint_positions(
                target_steering.reshape(1, -1),
                joint_indices=steering_joint_indices,
            )
            self._articulation_view.set_joint_position_targets(
                target_steering.reshape(1, -1),
                joint_indices=steering_joint_indices,
            )
            self._articulation_view.set_joint_velocities(
                np.zeros((1, len(steering_indices)), dtype=np.float32),
                joint_indices=steering_joint_indices,
            )

        if wheel_indices:
            wheel_joint_indices = np.asarray(wheel_indices, dtype=np.int32)
            zero_wheel = np.zeros((1, len(wheel_indices)), dtype=np.float32)
            self._articulation_view.set_joint_velocity_targets(
                zero_wheel,
                joint_indices=wheel_joint_indices,
            )
            self._articulation_view.set_joint_velocities(
                zero_wheel,
                joint_indices=wheel_joint_indices,
            )

    def apply_action(self, joint_positions, joint_indices, *args, **kwargs):
        """Apply the caller's arm/gripper action without adding base DOFs."""

        positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        indices = np.asarray(joint_indices, dtype=np.int64).reshape(-1)
        if positions.size != indices.size:
            raise ValueError(
                "TemplateRobot action position count must match joint index count"
            )
        self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)
        self.reapply_manipulation_base_hold()

    def _manipulation_base_hold_config(self):
        """Return this robot's explicit base-hold configuration.

        Concrete mobile robots may provide a small code-level default when
        their hold contract is intrinsic to the asset.  Ordinary robots keep
        the feature disabled unless their config opts in.
        """

        config = self.cfg.get("manipulation_base_hold", {})
        return config if isinstance(config, dict) else {}

    def _get_manipulation_base_hold_strategy(self):
        if self._base_hold_strategy is None:
            self._base_hold_strategy = BaseHoldStrategy(
                self._manipulation_base_hold_config(),
                port=self,
            )
        return self._base_hold_strategy

    # BaseHoldPort implementation.  These methods deliberately sit on the
    # robot boundary so the strategy remains independent of Isaac Sim types.
    def resolve_joint_indices(self, joint_names):
        return resolve_joint_names(
            list(self.dof_names),
            list(joint_names),
            group=f"{self.name}.manipulation_base_hold",
        )

    def read_joint_positions(self, indices):
        state = self._get_joints_state_compat()
        positions = np.asarray(state.positions, dtype=float)
        if positions.ndim > 1:
            positions = positions[0]
        return positions[np.asarray(indices, dtype=np.int64)].copy()

    def get_drive_state(self, indices):
        controller = self.get_articulation_controller()
        kps, kds = controller.get_gains()
        max_efforts = controller.get_max_efforts()
        if max_efforts is None:
            raise RuntimeError(f"robot {self.name} does not expose articulation max efforts")
        indices = np.asarray(indices, dtype=np.int64)
        return (
            np.asarray(kps, dtype=float).reshape(-1)[indices].copy(),
            np.asarray(kds, dtype=float).reshape(-1)[indices].copy(),
            np.asarray(max_efforts, dtype=float).reshape(-1)[indices].copy(),
        )

    def set_drive_state(self, indices, kps, kds, max_efforts):
        controller = self.get_articulation_controller()
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        kps = np.asarray(kps, dtype=float).reshape(-1)
        kds = np.asarray(kds, dtype=float).reshape(-1)
        max_efforts = np.asarray(max_efforts, dtype=float).reshape(-1)
        if not (kps.size == kds.size == max_efforts.size == indices.size):
            raise ValueError("base hold drive vectors must match joint indices")
        current_kps, current_kds = controller.get_gains()
        current_max_efforts = controller.get_max_efforts()
        if current_max_efforts is None:
            raise RuntimeError(f"robot {self.name} does not expose articulation max efforts")
        current_kps = np.asarray(current_kps, dtype=float).reshape(-1).copy()
        current_kds = np.asarray(current_kds, dtype=float).reshape(-1).copy()
        current_max_efforts = np.asarray(current_max_efforts, dtype=float).reshape(-1).copy()
        current_kps[indices] = kps
        current_kds[indices] = kds
        current_max_efforts[indices] = max_efforts
        controller.set_gains(kps=current_kps, kds=current_kds)
        controller.set_max_efforts(current_max_efforts)

    def set_position_targets(self, indices, positions):
        self._articulation_view.set_joint_position_targets(
            np.asarray(positions, dtype=float).reshape(-1),
            joint_indices=np.asarray(indices, dtype=np.int32).reshape(-1),
        )

    def set_velocity_targets(self, indices, velocities):
        self._articulation_view.set_joint_velocity_targets(
            np.asarray(velocities, dtype=float).reshape(-1),
            joint_indices=np.asarray(indices, dtype=np.int32).reshape(-1),
        )

    def _sync_manipulation_base_hold_state(self):
        strategy = self._get_manipulation_base_hold_strategy()
        self._manipulation_base_hold_active = bool(strategy.active)
        self._manipulation_base_hold_indices = np.asarray(strategy.indices, dtype=np.int64)
        self._manipulation_base_hold_positions = np.asarray(strategy.target_positions, dtype=float)
        saved_drive = strategy.saved_drive_state
        self._manipulation_base_hold_saved_drive = (
            None
            if saved_drive is None
            else {
                "indices": self._manipulation_base_hold_indices.copy(),
                "kps": np.asarray(saved_drive[0], dtype=float).copy(),
                "kds": np.asarray(saved_drive[1], dtype=float).copy(),
                "max_efforts": np.asarray(saved_drive[2], dtype=float).copy(),
            }
        )
        return strategy

    def enable_manipulation_base_hold(self) -> None:
        """Hold explicitly configured mobile-base DOFs during Pick/Place.

        SplitAloha's planar base joints intentionally have zero drive gain in
        the delivered USD because navigation controls them.  A manipulation
        episode, however, must not let arm reaction/contact forces move those
        joints while CuRobo is executing in an arm-base frame.  Joint names in
        robot config are the contract; no asset-name or numeric-index guessing
        is used here.
        """

        strategy = self._sync_manipulation_base_hold_state()
        if not strategy.enable():
            return
        self._sync_manipulation_base_hold_state()
        config = strategy.config
        # Hold activation is an expected manipulation boundary, not a
        # recovery or configuration warning. Keep the detailed audit visible
        # at INFO while reserving WARNING for a failed hold operation.
        LOGGER.info(
            "[BaseHold] robot=%s joints=%s indices=%s targets=%s kp=%.1f kd=%.1f max_effort=%.1f",
            self.name,
            list(config.joint_names),
            list(strategy.indices),
            np.round(np.asarray(strategy.target_positions), 6).tolist(),
            config.stiffness,
            config.damping,
            config.max_effort,
        )

    def suspend_manipulation_base_hold(self) -> bool:
        """Restore navigation drive gains while a mobile-base skill is active."""
        strategy = self._sync_manipulation_base_hold_state()
        result = strategy.suspend()
        self._sync_manipulation_base_hold_state()
        return result

    def resume_manipulation_base_hold(self) -> None:
        """Hold the base at its current pose after navigation finishes."""
        strategy = self._sync_manipulation_base_hold_state()
        if strategy.resume():
            self._sync_manipulation_base_hold_state()

    def recapture_manipulation_base_hold(self) -> bool:
        """Refresh an active hold after a reset or explicit base-pose write."""

        strategy = self._sync_manipulation_base_hold_state()
        result = strategy.recapture()
        self._sync_manipulation_base_hold_state()
        return result

    def reapply_manipulation_base_hold(self) -> bool:
        """Reapply hold gains and targets for an action/physics boundary."""

        strategy = self._sync_manipulation_base_hold_state()
        result = strategy.reapply()
        self._sync_manipulation_base_hold_state()
        return result

    def get_observations(self) -> dict:
        joint_state = self._get_joints_state_compat()
        qpos, qvel = joint_state.positions, joint_state.velocities

        T_base_ee_fl = get_relative_transform(get_prim_at_path(self.fl_ee_path), get_prim_at_path(self.fl_base_path))
        try:
            base_translation, base_orientation = self.get_local_pose()
        except Exception as exc:
            self._write_observation_failure_debug(
                exc,
                stage="get_local_pose",
                qpos=qpos,
                qvel=qvel,
            )
            raise
        T_world_base = tf_matrix_from_pose(base_translation, base_orientation)

        obs = self._build_observations(qpos, qvel, T_base_ee_fl, T_world_base)
        return obs

    def _get_joints_state_compat(self):
        """Read joint state across Isaac Sim 4.x and 6.x articulation wrappers."""
        get_joints_state = getattr(self, "get_joints_state", None)
        joint_state = get_joints_state() if callable(get_joints_state) else None
        if joint_state is not None:
            return joint_state

        # Isaac Sim 6 can briefly return None from the aggregate
        # ``get_joints_state`` view while the articulation is synchronizing.
        # Its scalar position/velocity accessors remain the supported public
        # fallback and expose the same data without relying on the old
        # JointsState wrapper.
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        if positions is None or velocities is None:
            raise RuntimeError(
                f"robot {self.name} has no readable articulation joint state"
            )
        return SimpleNamespace(positions=positions, velocities=velocities)

    def _write_observation_failure_debug(self, exc: Exception, *, stage: str, qpos=None, qvel=None):
        payload = {
            "robot_name": str(getattr(self, "name", "")),
            "robot_prim_path": str(getattr(self, "robot_prim_path", "")),
            "stage": str(stage),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            "joint_state_summary": {
                "qpos": self._array_debug_summary(qpos),
                "qvel": self._array_debug_summary(qvel),
            },
            "articulation": self._articulation_pose_debug(),
            "base_driver": self._base_bridge_debug(),
        }
        output_dir = os.environ.get("SIMBOX_DEBUG_OUTPUT_DIR", "output/simbox_debug")
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"robot_observation_failure_{self._safe_debug_name()}_{time.time_ns()}.json"
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._json_safe(payload), f, indent=2)
            print(f"[TemplateRobot] observation failure debug written to {path}", flush=True)
        except Exception as dump_exc:  # pylint: disable=broad-except
            print(
                "[TemplateRobot] failed to write observation failure debug: "
                f"{type(dump_exc).__name__}: {dump_exc}",
                flush=True,
            )

    def _safe_debug_name(self) -> str:
        name = str(getattr(self, "name", "") or self.cfg.get("name", "robot"))
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name) or "robot"

    def _articulation_pose_debug(self) -> dict:
        view = getattr(self, "_articulation_view", None)
        payload = {
            "view_type": type(view).__name__ if view is not None else "",
            "prim_paths": self._safe_get_attr(view, "prim_paths"),
            "body_names": self._safe_get_attr(view, "body_names"),
            "dof_names": self._safe_get_attr(view, "dof_names"),
        }
        if view is None:
            return payload
        for method_name in ("get_world_poses", "get_local_poses"):
            method = getattr(view, method_name, None)
            if not callable(method):
                payload[method_name] = {"available": False}
                continue
            try:
                positions, orientations = method()
                orientations_array = np.asarray(orientations, dtype=np.float64)
                norms = np.linalg.norm(orientations_array.reshape(-1, 4), axis=1)
                zero_norm_indices = np.where(norms <= 1.0e-8)[0].astype(int).tolist()
                payload[method_name] = {
                    "positions": positions,
                    "orientations": orientations,
                    "orientation_norms": norms,
                    "zero_norm_orientation_indices": zero_norm_indices,
                }
            except Exception as exc:  # pylint: disable=broad-except
                payload[method_name] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        return payload

    def _base_bridge_debug(self) -> dict:
        bridge = getattr(self, "_simbox_local_base_driver", None)
        if bridge is None:
            return {"available": False}
        payload = {
            "available": True,
            "type": type(bridge).__name__,
        }
        for name in ("get_logging_action_snapshot", "get_logging_state_snapshot"):
            method = getattr(bridge, name, None)
            if not callable(method):
                continue
            try:
                payload[name] = method()
            except Exception as exc:  # pylint: disable=broad-except
                payload[name] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        for attr in (
            "_navigation_active",
            "_has_command",
            "_last_requested_steering",
            "_last_requested_wheel_velocities",
            "_last_applied_wheel_velocities",
            "_last_step_dt",
            "_non_finite_state_detected",
            "_non_finite_state_reason",
        ):
            payload[attr.lstrip("_")] = self._safe_get_attr(bridge, attr)
        history = getattr(bridge, "_debug_command_history", None)
        if history is not None:
            try:
                payload["debug_command_history_tail"] = list(history)[-8:]
            except Exception as exc:  # pylint: disable=broad-except
                payload["debug_command_history_tail"] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        return payload

    @classmethod
    def _array_debug_summary(cls, value) -> dict:
        if value is None:
            return {"available": False}
        try:
            array = np.asarray(value)
            flat = array.reshape(-1)
            return {
                "available": True,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.all(np.isfinite(flat))) if flat.size else True,
                "min": float(np.nanmin(flat)) if flat.size else None,
                "max": float(np.nanmax(flat)) if flat.size else None,
                "sample": flat[:24],
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    @classmethod
    def _safe_get_attr(cls, obj, attr):
        try:
            return getattr(obj, attr)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        try:
            return cls._json_safe(list(value))
        except TypeError:
            return str(value)

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
        depths = poses[:, 3:4] * grasp_scale

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
