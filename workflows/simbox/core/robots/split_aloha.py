"""Physical and virtual-base SplitAloha robot implementations."""
from copy import deepcopy
import math
import os

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import tf_matrix_from_pose
from isaacsim.core.utils.xforms import get_world_pose as get_prim_world_pose
from pxr import PhysxSchema, UsdPhysics


# pylint: disable=line-too-long,unused-argument
@register_robot
class SplitAlohaActual(TemplateRobot):
    """SplitAloha dual-arm robot driven through its physical 4WIS wheel base."""

    def __init__(self, *args, **kwargs):
        self.base_cfg = {}
        self.base_steering_joint_names = []
        self.base_wheel_joint_names = []
        self.base_steering_joint_indices = []
        self.base_wheel_joint_indices = []
        self._base_initial_steering_positions = None
        self.mobile_base_prim_path = None
        self._wheel_collision_paths = []
        self._disabled_collision_paths = []
        self._wheel_physics_material_path = None
        self._gripper_physics_material_path = None
        self._wheel_joint_paths = []
        self._steering_joint_paths = []
        super().__init__(*args, **kwargs)
        self.base_cfg = deepcopy(self.cfg.get("base", {}))
        self.base_steering_joint_names = list(self.base_cfg.get("steering_joint_names", []))
        self.base_wheel_joint_names = list(self.base_cfg.get("wheel_joint_names", []))
        self._setup_mobile_base_interface()
        self._configure_mobile_base_wheel_drives()
        self._configure_gripper_physics_material()

    def _configure_gripper_physics_material(self):
        """Apply the optional material used by the validated grasp pipeline."""

        static_friction = self.cfg.get("gripper_static_friction", None)
        if static_friction is None:
            return
        dynamic_friction = self.cfg.get("gripper_dynamic_friction", static_friction)
        try:
            static_friction = float(static_friction)
            dynamic_friction = float(dynamic_friction)
        except (TypeError, ValueError):
            raise ValueError("SplitAloha gripper friction values must be numeric") from None
        if (
            not math.isfinite(static_friction)
            or not math.isfinite(dynamic_friction)
            or static_friction < 0.0
            or dynamic_friction < 0.0
        ):
            raise ValueError("SplitAloha gripper friction values must be finite and non-negative")

        try:
            from omni.physx.scripts import physicsUtils, utils
        except ImportError:
            return

        stage = get_prim_at_path(self.robot_prim_path).GetStage()
        material_path = f"{self.robot_prim_path}/Looks/gripper_physics_material"
        material_prim = get_prim_at_path(material_path)
        if not material_prim.IsValid():
            utils.addRigidBodyMaterial(
                stage,
                material_path,
                staticFriction=static_friction,
                dynamicFriction=dynamic_friction,
                restitution=0.0,
            )
        else:
            material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
            material_api.CreateStaticFrictionAttr().Set(static_friction)
            material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
        self._gripper_physics_material_path = material_path

        for arm in ("fl", "fr"):
            for relative_path in self.cfg.get(f"{arm}_filter_paths", []):
                collision_path = f"{self.robot_prim_path}/{str(relative_path).strip('/')}/collisions"
                collision_prim = get_prim_at_path(collision_path)
                if not collision_prim.IsValid():
                    continue
                physicsUtils.add_physics_material_to_prim(
                    stage,
                    collision_prim,
                    material_path,
                )

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def _setup_paths(self):
        fl_ee_path = self.cfg["fl_ee_path"]
        fr_ee_path = self.cfg["fr_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fr_ee_path = f"{self.robot_prim_path}/{fr_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg["fr_gripper_keypoints"]

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_filter_paths"]]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_forbid_collision_paths"]]

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.05 else -1.0

    def _setup_joint_velocities(self):
        # SplitAloha has 12 joints for velocity control
        all_joint_indices = self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                [500.0] * 12,
                joint_indices=all_joint_indices,
            )

    def _set_initial_positions(self):
        positions = self.left_joint_home + self.right_joint_home + self.left_gripper_home + self.right_gripper_home
        indices = (
            self.left_joint_indices + self.right_joint_indices + self.left_gripper_indices + self.right_gripper_indices
        )
        if positions and indices:
            self._articulation_view.set_joint_positions(positions, joint_indices=indices)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._setup_base_joint_indices()
        self._capture_base_initial_steering_positions()

    def _setup_base_joint_indices(self):
        dof_names = list(self._articulation_view.dof_names)
        self.base_steering_joint_indices = [dof_names.index(name) for name in self.base_steering_joint_names]
        self.base_wheel_joint_indices = [dof_names.index(name) for name in self.base_wheel_joint_names]

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
        mobile_root = os.path.dirname(os.path.dirname(self.fl_base_path))
        base_frame = str(self.base_cfg.get("base_frame", "base_link"))
        candidate_path = f"{mobile_root}/{base_frame}"
        if get_prim_at_path(candidate_path).IsValid():
            self.mobile_base_prim_path = candidate_path
        else:
            self.mobile_base_prim_path = None

        # Only keep the real wheel collision geometry in the physical drive path.
        # Steering support / fork geometry can easily catch the floor or low obstacles
        # and destabilize the base when we are trying to validate wheel-ground contact.
        self._wheel_collision_paths = [
            f"{mobile_root}/fl_wheel_link/collisions",
            f"{mobile_root}/fr_wheel_link/collisions",
            f"{mobile_root}/rl_wheel_link/collisions",
            f"{mobile_root}/rr_wheel_link/collisions",
        ]
        self._disabled_collision_paths = [
            f"{mobile_root}/fl_steering_wheel_link/collisions",
            f"{mobile_root}/fr_steering_wheel_link/collisions",
            f"{mobile_root}/rl_steering_wheel_link/collisions",
            f"{mobile_root}/rr_steering_wheel_link/collisions",
        ]
        self._wheel_physics_material_path = f"{mobile_root}/Looks/wheel_physics_material"
        self._wheel_joint_paths = [
            f"{mobile_root}/fl_steering_wheel_link/fl_wheel",
            f"{mobile_root}/fr_steering_wheel_link/fr_wheel",
            f"{mobile_root}/rl_steering_wheel_link/rl_wheel",
            f"{mobile_root}/rr_steering_wheel_link/rr_wheel",
        ]
        self._steering_joint_paths = [f"{mobile_root}/{base_frame}/{joint_name}" for joint_name in self.base_steering_joint_names]

    def _configure_mobile_base_wheel_drives(self):
        try:
            from pxr import UsdPhysics  # pylint: disable=import-outside-toplevel
            from omni.physx.scripts import physicsUtils, utils  # pylint: disable=import-outside-toplevel
        except ImportError:
            return

        static_friction = float(self.base_cfg.get("wheel_static_friction", 1.5))
        dynamic_friction = float(self.base_cfg.get("wheel_dynamic_friction", static_friction))
        restitution = float(self.base_cfg.get("wheel_restitution", 0.0))
        wheel_drive_stiffness = float(self.base_cfg.get("wheel_drive_stiffness", 0.0))
        wheel_drive_damping = float(self.base_cfg.get("wheel_drive_damping", 150.0))
        wheel_drive_max_force = float(self.base_cfg.get("wheel_drive_max_force", 300.0))
        steering_drive_stiffness = float(self.base_cfg.get("steering_drive_stiffness", 1.0e7))
        steering_drive_damping = float(self.base_cfg.get("steering_drive_damping", 1.0e5))
        steering_drive_max_force = float(self.base_cfg.get("steering_drive_max_force", 1.0e6))
        stage = get_prim_at_path(self.robot_prim_path).GetStage()
        if self._wheel_physics_material_path and not get_prim_at_path(self._wheel_physics_material_path).IsValid():
            utils.addRigidBodyMaterial(
                stage,
                self._wheel_physics_material_path,
                staticFriction=static_friction,
                dynamicFriction=dynamic_friction,
                restitution=restitution,
            )

        for collision_path in self._wheel_collision_paths:
            prim = get_prim_at_path(collision_path)
            if not prim.IsValid():
                continue
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr().Set(True)
            if self._wheel_physics_material_path:
                physicsUtils.add_physics_material_to_prim(stage, prim, self._wheel_physics_material_path)

        for collision_path in self._disabled_collision_paths:
            prim = get_prim_at_path(collision_path)
            if not prim.IsValid():
                continue
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr().Set(False)

        for joint_path in self._steering_joint_paths:
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                continue
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive_api:
                continue
            drive_api.GetStiffnessAttr().Set(steering_drive_stiffness)
            drive_api.GetDampingAttr().Set(steering_drive_damping)
            drive_api.GetMaxForceAttr().Set(steering_drive_max_force)

        for joint_path in self._wheel_joint_paths:
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                continue
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive_api:
                continue
            drive_api.GetStiffnessAttr().Set(wheel_drive_stiffness)
            drive_api.GetDampingAttr().Set(wheel_drive_damping)
            drive_api.GetMaxForceAttr().Set(wheel_drive_max_force)

    def get_base_initial_steering_positions(self):
        if self._base_initial_steering_positions is None:
            self._capture_base_initial_steering_positions()
        return np.asarray(self._base_initial_steering_positions, dtype=np.float32).copy()

    @staticmethod
    def _is_finite_scalar(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_finite_vec3(cls, value):
        try:
            return value is not None and all(cls._is_finite_scalar(component) for component in value)
        except TypeError:
            return False

    @classmethod
    def _is_finite_quat(cls, value):
        if value is None:
            return False
        try:
            return all(
                cls._is_finite_scalar(component)
                for component in (value.GetReal(), *value.GetImaginary())
            )
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _set_or_create_attr(attr, value, create_fn):
        authored_attr = attr if attr.IsValid() else create_fn()
        authored_attr.Set(value)

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
        if self.mobile_base_prim_path:
            return get_prim_world_pose(self.mobile_base_prim_path)
        return self.get_world_pose()

    def apply_base_command(self, steering_positions, wheel_velocities):
        steering_positions = np.asarray(steering_positions, dtype=np.float32)
        wheel_velocities = np.asarray(wheel_velocities, dtype=np.float32)
        if steering_positions.shape[0] != len(self.base_steering_joint_indices):
            raise ValueError("steering_positions size does not match steering joints")
        if wheel_velocities.shape[0] != len(self.base_wheel_joint_indices):
            raise ValueError("wheel_velocities size does not match wheel joints")
        if not np.all(np.isfinite(steering_positions)):
            fallback = self.get_base_initial_steering_positions()
            if fallback.shape[0] != steering_positions.shape[0] or not np.all(np.isfinite(fallback)):
                fallback = np.zeros_like(steering_positions)
            steering_positions = fallback.astype(np.float32)
        if not np.all(np.isfinite(wheel_velocities)):
            wheel_velocities = np.zeros_like(wheel_velocities, dtype=np.float32)

        self._articulation_view.set_joint_position_targets(
            steering_positions.reshape(1, -1),
            joint_indices=np.array(self.base_steering_joint_indices, dtype=np.int32),
        )
        self._articulation_view.set_joint_velocity_targets(
            wheel_velocities.reshape(1, -1),
            joint_indices=np.array(self.base_wheel_joint_indices, dtype=np.int32),
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

    def get_observations(self) -> dict:
        obs = super().get_observations()
        if not self.base_steering_joint_indices and not self.base_wheel_joint_indices:
            return obs

        driver = getattr(self, "_simbox_local_base_driver", None)
        if driver is not None and hasattr(driver, "get_logging_state_snapshot"):
            base_state = driver.get_logging_state_snapshot()
        else:
            translation, orientation = self.get_mobile_base_pose()
            joint_state = self.get_base_joint_state()
            base_state = {
                "pose": [
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                    float(self._yaw_from_wxyz(orientation)),
                ],
                "twist_body": [0.0, 0.0, 0.0],
                "steering_positions": [float(v) for v in joint_state["steering_positions"].tolist()],
                "wheel_positions": [float(v) for v in joint_state["wheel_positions"].tolist()],
                "steering_velocities": [float(v) for v in joint_state["steering_velocities"].tolist()],
                "wheel_velocities": [float(v) for v in joint_state["wheel_velocities"].tolist()],
            }

        obs["states.base.pose"] = np.asarray(base_state["pose"], dtype=np.float32)
        obs["states.base.twist_body"] = np.asarray(base_state["twist_body"], dtype=np.float32)
        obs["states.base.steering_positions"] = np.asarray(base_state["steering_positions"], dtype=np.float32)
        obs["states.base.wheel_positions"] = np.asarray(base_state["wheel_positions"], dtype=np.float32)
        obs["states.base.steering_velocities"] = np.asarray(base_state["steering_velocities"], dtype=np.float32)
        obs["states.base.wheel_velocities"] = np.asarray(base_state["wheel_velocities"], dtype=np.float32)
        return obs


@register_robot
class SplitAloha(SplitAlohaActual):
    """SplitAloha dual-arm robot driven by world-anchored virtual X/Y/yaw joints."""

    def __init__(self, *args, **kwargs):
        self._active_manipulator_joint_positions = None
        self._active_manipulator_joint_indices = None
        super().__init__(*args, **kwargs)

    def _setup_joint_indices(self):
        # The virtual asset fixes eight physical wheel/steering joints, so its
        # articulation DOF order intentionally differs from SplitAlohaActual.
        # Resolve all active manipulator indices by name after initialization.
        self.left_joint_indices = []
        self.right_joint_indices = []
        self.left_gripper_indices = []
        self.right_gripper_indices = []
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._setup_manipulator_joint_indices()
        self._validate_mobile_base_joint_partition()
        self._setup_joint_velocities()
        self._set_initial_positions()
        self._configure_mobile_base_wheel_drives()
        self._reset_virtual_base_joint_state(require_ready=True)
        self.recapture_manipulation_base_hold()

    def _setup_manipulator_joint_indices(self):
        dof_names = list(self._articulation_view.dof_names)
        groups = {
            "left_joint_indices": list(self.cfg["left_joint_names"]),
            "right_joint_indices": list(self.cfg["right_joint_names"]),
            "left_gripper_indices": list(self.cfg["left_gripper_joint_names"]),
            "right_gripper_indices": list(self.cfg["right_gripper_joint_names"]),
        }
        required_names = [name for names in groups.values() for name in names]
        missing = [name for name in required_names if name not in dof_names]
        if missing:
            raise KeyError(f"SplitAloha missing expected manipulator DOF names: {missing}; available={dof_names}")
        for attr, names in groups.items():
            indices = [dof_names.index(name) for name in names]
            setattr(self, attr, indices)
            self.cfg[attr] = list(indices)

    def _setup_mobile_base_interface(self):
        if self.base_steering_joint_names:
            raise ValueError("SplitAloha virtual base does not support steering joints")
        if len(self.base_wheel_joint_names) != 3:
            raise ValueError("SplitAloha virtual base requires exactly three velocity joints")

        mobile_base_path = str(self.cfg["mobile_base_path"]).strip("/")
        self.mobile_base_prim_path = f"{self.robot_prim_path}/{mobile_base_path}"
        if not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            raise ValueError(f"SplitAloha virtual base prim does not exist: {self.mobile_base_prim_path}")

        relative_joint_paths = list(self.cfg["virtual_base_joint_paths"])
        if len(relative_joint_paths) != 3:
            raise ValueError("SplitAloha virtual_base_joint_paths must contain X, Y, and yaw joints")
        self._wheel_joint_paths = [f"{self.robot_prim_path}/{path.strip('/')}" for path in relative_joint_paths]
        self._wheel_collision_paths = []
        self._disabled_collision_paths = []
        self._wheel_physics_material_path = None

    def _virtual_base_drive_parameters(self):
        stiffness = float(self.base_cfg["virtual_drive_stiffness"])
        if not math.isclose(stiffness, 0.0, abs_tol=1.0e-8):
            raise ValueError("SplitAloha velocity drives require virtual_drive_stiffness: 0.0")
        dampings = np.asarray(
            [
                self.base_cfg["virtual_linear_drive_damping"],
                self.base_cfg["virtual_linear_drive_damping"],
                self.base_cfg["virtual_yaw_drive_damping"],
            ],
            dtype=np.float32,
        )
        max_forces = np.asarray(
            [
                self.base_cfg["virtual_linear_drive_max_force"],
                self.base_cfg["virtual_linear_drive_max_force"],
                self.base_cfg["virtual_yaw_drive_max_force"],
            ],
            dtype=np.float32,
        )
        platform_cfg = self.base_cfg["platform"]
        local_navigation_cfg = platform_cfg["local_navigation"]
        hard_limits = local_navigation_cfg["controller_hard_limits"]
        max_velocity = np.asarray(hard_limits["max_velocity"], dtype=np.float32).reshape(-1)
        min_velocity = np.asarray(hard_limits["min_velocity"], dtype=np.float32).reshape(-1)
        if max_velocity.size != 3 or min_velocity.size != 3:
            raise ValueError("SplitAloha virtual controller velocity limits must be 3-element lists")
        max_velocities = np.maximum(np.abs(max_velocity), np.abs(min_velocity))
        if (
            not np.all(np.isfinite(dampings))
            or not np.all(np.isfinite(max_forces))
            or not np.all(np.isfinite(max_velocities))
            or np.any(dampings <= 0.0)
            or np.any(max_forces <= 0.0)
            or np.any(max_velocities <= 0.0)
        ):
            raise ValueError("SplitAloha virtual drive parameters must be positive finite values")
        return stiffness, dampings, max_forces, max_velocities.astype(np.float32)

    def _configure_mobile_base_wheel_drives(self):
        stiffness, dampings, max_forces, max_velocities = self._virtual_base_drive_parameters()
        drive_specs = (
            ("linear", dampings[0], max_forces[0], max_velocities[0]),
            ("linear", dampings[1], max_forces[1], max_velocities[1]),
            ("angular", dampings[2], max_forces[2], max_velocities[2]),
        )
        for joint_path, (drive_name, damping, max_force, max_velocity) in zip(
            self._wheel_joint_paths,
            drive_specs,
        ):
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                raise ValueError(f"SplitAloha virtual joint prim does not exist: {joint_path}")
            drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
            if not drive:
                raise ValueError(f"SplitAloha virtual joint is missing {drive_name} drive: {joint_path}")
            drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(float(damping))
            drive.CreateMaxForceAttr().Set(float(max_force))
            drive.CreateTargetVelocityAttr().Set(0.0)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(float(max_velocity))
        self._configure_virtual_base_runtime_drives(require_ready=False)

    def _configure_virtual_base_runtime_drives(self, *, require_ready: bool):
        if len(self.base_wheel_joint_indices) != 3:
            if require_ready:
                raise ValueError("SplitAloha virtual base joint indices are not initialized")
            return
        articulation_view = getattr(self, "_articulation_view", None)
        if articulation_view is None or not getattr(articulation_view, "_is_initialized", False):
            if require_ready:
                raise ValueError("SplitAloha articulation view is not initialized")
            return
        if not articulation_view.is_physics_handle_valid():
            if require_ready:
                raise ValueError("SplitAloha articulation physics handle is invalid")
            return

        _, dampings, max_forces, max_velocities = self._virtual_base_drive_parameters()
        indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
        zeros = np.zeros((1, 3), dtype=np.float32)
        articulation_view.set_max_joint_velocities(max_velocities.reshape(1, 3), joint_indices=indices)
        articulation_view.set_max_efforts(max_forces.reshape(1, 3), joint_indices=indices)
        articulation_view.set_gains(kps=zeros, kds=dampings.reshape(1, 3), joint_indices=indices)
        articulation_view.set_joint_velocity_targets(zeros, joint_indices=indices)

    def _validate_mobile_base_joint_partition(self):
        base_indices = set(self.base_wheel_joint_indices)
        manip_indices = (
            set(self.left_joint_indices)
            | set(self.right_joint_indices)
            | set(self.left_gripper_indices)
            | set(self.right_gripper_indices)
        )
        overlap = sorted(base_indices & manip_indices)
        if overlap:
            dof_names = list(self._articulation_view.dof_names)
            raise ValueError(
                "SplitAloha virtual base mapping overlaps manipulator joints: "
                f"{[dof_names[index] for index in overlap]}"
            )

    def _set_initial_positions(self):
        super()._set_initial_positions()
        positions = self.left_joint_home + self.right_joint_home + self.left_gripper_home + self.right_gripper_home
        indices = (
            self.left_joint_indices
            + self.right_joint_indices
            + self.left_gripper_indices
            + self.right_gripper_indices
        )
        if positions and indices:
            positions_array = np.asarray(positions, dtype=np.float32).reshape(1, -1)
            indices_array = np.asarray(indices, dtype=np.int32)
            self._articulation_view.set_joint_position_targets(positions_array, joint_indices=indices_array)
            self._articulation_view.set_joint_velocities(np.zeros_like(positions_array), joint_indices=indices_array)
            self._active_manipulator_joint_positions = positions_array.copy()
            self._active_manipulator_joint_indices = indices_array.copy()

    def _reset_virtual_base_joint_state(self, *, require_ready: bool):
        if len(self.base_wheel_joint_indices) != 3:
            if require_ready:
                raise ValueError("SplitAloha virtual base joint indices are not initialized")
            return
        indices = np.asarray(self.base_wheel_joint_indices, dtype=np.int32)
        zeros = np.zeros((1, 3), dtype=np.float32)
        self._articulation_view.set_joint_positions(zeros, joint_indices=indices)
        self._articulation_view.set_joint_velocities(zeros, joint_indices=indices)
        self._configure_virtual_base_runtime_drives(require_ready=require_ready)

    def get_mobile_base_pose(self):
        if not self.mobile_base_prim_path or not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            raise ValueError("SplitAloha virtual mobile base prim is not initialized")
        return get_prim_world_pose(self.mobile_base_prim_path)

    def get_world_pose(self):
        if not self.mobile_base_prim_path or not get_prim_at_path(self.mobile_base_prim_path).IsValid():
            return super().get_world_pose()
        return self.get_mobile_base_pose()

    def set_mobile_base_world_pose(self, translation, orientation):
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(orientation)):
            raise ValueError("SplitAloha virtual mobile base pose must be finite")
        if float(np.linalg.norm(orientation)) <= 1.0e-8:
            raise ValueError("SplitAloha virtual mobile base orientation must be a non-zero quaternion")
        self._reset_virtual_base_joint_state(require_ready=False)
        self.set_world_pose(position=translation, orientation=orientation)

    def reset_mobile_base_world_state(self, translation, orientation):
        self.set_mobile_base_world_pose(translation, orientation)
        self.set_world_velocity(np.zeros(6, dtype=np.float32))
        if getattr(self, "num_dof", 0):
            self._articulation_view.set_joint_velocities(np.zeros((1, int(self.num_dof)), dtype=np.float32))
        self._set_initial_positions()
        self._configure_mobile_base_wheel_drives()
        self._reset_virtual_base_joint_state(require_ready=True)
        self.recapture_manipulation_base_hold()

    def apply_action(self, joint_positions, joint_indices, *args, **kwargs):
        positions = np.asarray(joint_positions, dtype=np.float32).reshape(1, -1)
        indices = np.asarray(joint_indices, dtype=np.int32).reshape(-1)
        if positions.shape[1] != indices.size:
            raise ValueError("SplitAloha action position count must match joint index count")
        if not np.all(np.isfinite(positions)):
            raise ValueError("SplitAloha action joint positions must be finite")
        overlap = sorted(set(int(index) for index in indices.tolist()) & set(self.base_wheel_joint_indices))
        if overlap:
            dof_names = list(self._articulation_view.dof_names)
            raise ValueError(
                "SplitAloha manipulator action must not target virtual base joints: "
                f"{[dof_names[index] for index in overlap]}"
            )
        self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)
        self._active_manipulator_joint_positions = positions.copy()
        self._active_manipulator_joint_indices = indices.copy()
        self.reapply_manipulation_base_hold()

    def _reapply_active_manipulator_position_target(self):
        if self._active_manipulator_joint_positions is None or self._active_manipulator_joint_indices is None:
            return
        self._articulation_view.set_joint_position_targets(
            self._active_manipulator_joint_positions,
            joint_indices=self._active_manipulator_joint_indices,
        )

    def apply_base_command(self, steering_positions, wheel_velocities, *, step_dt: float):
        steering_positions = np.asarray(steering_positions, dtype=np.float32).reshape(-1)
        body_velocity = np.asarray(wheel_velocities, dtype=np.float32).reshape(-1)
        if steering_positions.size != 0:
            raise ValueError("SplitAloha virtual base does not accept steering commands")
        if body_velocity.size != 3 or not np.all(np.isfinite(body_velocity)):
            raise ValueError("SplitAloha virtual base requires three finite velocity commands")
        if not math.isfinite(float(step_dt)) or float(step_dt) < 0.0:
            raise ValueError("SplitAloha virtual base step_dt must be finite and non-negative")
        if len(self.base_wheel_joint_indices) != 3:
            raise ValueError("SplitAloha virtual base joint indices are not initialized")

        joint_positions = self._articulation_view.get_joint_positions()[0]
        yaw = float(joint_positions[self.base_wheel_joint_indices[2]])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_body, vy_body, wz_body = (float(value) for value in body_velocity)
        joint_velocity = np.asarray(
            [
                cos_yaw * vx_body - sin_yaw * vy_body,
                sin_yaw * vx_body + cos_yaw * vy_body,
                wz_body,
            ],
            dtype=np.float32,
        ).reshape(1, 3)
        self._articulation_view.set_joint_velocity_targets(
            joint_velocity,
            joint_indices=np.asarray(self.base_wheel_joint_indices, dtype=np.int32),
        )
        self._reapply_active_manipulator_position_target()

    def get_observations(self) -> dict:
        obs = super().get_observations()
        translation, orientation = self.get_mobile_base_pose()
        obs["T_world_base"] = tf_matrix_from_pose(
            np.asarray(translation, dtype=np.float32),
            np.asarray(orientation, dtype=np.float32),
        )
        return obs
