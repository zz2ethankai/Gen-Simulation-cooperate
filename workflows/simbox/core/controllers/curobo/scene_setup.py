from __future__ import annotations
import logging
import warnings
from typing import Any
import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform
from core.controllers.curobo.components import (
    MutableExecutionState,
)
from core.utils.joint_index_resolver import JointIndexResolutionError, resolve_joint_names
LOGGER = logging.getLogger("de_logger")
class ControllerSetup:
    @property
    def ds_ratio(self) -> int:
        return int(self.execution_state.ds_ratio)
    @ds_ratio.setter
    def ds_ratio(self, value: int) -> None:
        self.execution_state.ds_ratio = max(1, int(value))
    def __init__(
        self,
        *,
        name: str,
        world: Any,
        task: Any,
        robot: Any,
        arm_spec: Any,
        robot_file: str,
        tensor_args: Any,
        phase_executor: Any,
        execution_state: MutableExecutionState | None = None,
        collision_scene_manager: Any = None,
        ignore_substring: Any = (),
    ) -> None:
        self.name = name
        self.world = world
        self.task = task
        self.robot = robot
        self.arm_spec = arm_spec
        self.robot_file = robot_file
        self.tensor_args = tensor_args
        self.phase_executor = phase_executor
        self.execution_state = execution_state or MutableExecutionState()
        self.collision_scene_manager = collision_scene_manager
        self.ignore_substring = tuple(ignore_substring or ())
        self.raw_js_names = []
        self.cmd_js_names = []
        self.arm_indices = np.array([], dtype=np.int64)
        self.gripper_indices = np.array([], dtype=np.int64)
        self.reference_prim_path = None
        self.robot_base_path = None
        self.robot_ee_path = None
        self.lr_name = None
        self.scene_port = None
        self.world_cfg = None
        self.interpolation_dt = 0.01
    @property
    def task_root_prim_path(self) -> str:
        return self.task.root_prim_path
    def _configure_execution_stride(self) -> None:
        physics_dt = float(self.world.get_physics_dt())
        interpolation_dt = float(self.interpolation_dt)
        requested_stride = max(1, int(round(physics_dt / interpolation_dt)))
        safety_cfg = self.task.cfg.get("planning", {}).get("execution_safety", {})
        max_stride = max(1, int(safety_cfg.get("max_waypoint_stride", 2)))
        self.ds_ratio = min(requested_stride, max_stride)
    def _configure_joint_indices(self, robot_file: str) -> None:
        if self.arm_spec is None:
            raise NotImplementedError
        arm = "right" if "right" in str(robot_file).lower() else "left"
        if arm not in self.arm_spec.supported_arms:
            raise NotImplementedError(
                f"{type(self).__name__} does not expose arm {arm!r}"
            )
        self.raw_js_names = list(self.arm_spec.planner_joint_names(arm))
        self.cmd_js_names = list(self.arm_spec.control_joint_names(arm))
        self.arm_indices = np.asarray(
            getattr(self.robot, f"{arm}_joint_indices"), dtype=np.int64
        )
        self.gripper_indices = np.asarray(
            getattr(self.robot, f"{arm}_gripper_indices"), dtype=np.int64
        )
        robot_view = self.task.robots[self.name]
        self._resolve_robot_frame_paths(robot_view, arm)
        self.reference_prim_path = self.robot_base_path
        self.lr_name = arm
        self.execution_state.gripper_state = (
            1.0
            if float(getattr(self.robot, f"{arm}_gripper_state")) >= 0.0
            else -1.0
        )
        configured_home = getattr(self.robot, f"{arm}_gripper_home", None)
        self.execution_state.gripper_joint_position = np.asarray(
            configured_home
            if configured_home is not None
            else self.arm_spec.gripper_home,
            dtype=float,
        ).reshape(-1)
    def _resolve_robot_frame_paths(self, robot_view: Any, arm: str) -> None:
        if arm == "left":
            base_path = robot_view.fl_base_path
            ee_path = robot_view.fl_ee_path
        elif arm == "right":
            base_path = robot_view.fr_base_path
            ee_path = robot_view.fr_ee_path
        else:
            raise ValueError(f"unsupported controller arm {arm!r}")
        if not base_path or not ee_path:
            raise ValueError(
                f"robot {self.name!r} must expose non-empty {arm} base and EE paths"
            )
        self.robot_base_path = base_path
        self.robot_ee_path = ee_path
    def _resolve_runtime_control_indices(self) -> None:
        if self.lr_name not in {"left", "right"}:
            raise JointIndexResolutionError(f"controller {self.name} did not select a left/right arm")
        runtime_names = list(self.robot.dof_names)
        resolved_arm_indices = resolve_joint_names(
            runtime_names,
            self.cmd_js_names,
            group=f"{self.name}.{self.lr_name}_arm",
        )
        configured_arm_indices = [int(index) for index in self.arm_indices]
        if configured_arm_indices != resolved_arm_indices:
            LOGGER.warning(
                "[JointIndex] controller=%s arm=%s corrected arm indices from %s to %s using runtime dof_names",
                self.name,
                self.lr_name,
                configured_arm_indices,
                resolved_arm_indices,
            )
        arm_indices_field = f"{self.lr_name}_joint_indices"
        gripper_indices_field = f"{self.lr_name}_gripper_indices"
        self.arm_indices = np.asarray(resolved_arm_indices, dtype=np.int64)
        resolved_gripper_indices = [int(index) for index in getattr(self.robot, gripper_indices_field)]
        if set(resolved_arm_indices) & set(resolved_gripper_indices):
            raise JointIndexResolutionError(
                f"controller {self.name} {self.lr_name} arm/gripper indices overlap: "
                f"arm={resolved_arm_indices}, gripper={resolved_gripper_indices}"
            )
        self.gripper_indices = np.asarray(resolved_gripper_indices, dtype=np.int64)
        setattr(self.robot, arm_indices_field, resolved_arm_indices)
        self.robot.cfg[arm_indices_field] = resolved_arm_indices
        self.robot.cfg[gripper_indices_field] = resolved_gripper_indices
        LOGGER.info(
            "[JointIndexAudit] controller=%s arm=%s joints=%s arm_indices=%s gripper_indices=%s",
            self.name,
            self.lr_name,
            list(self.cmd_js_names),
            resolved_arm_indices,
            resolved_gripper_indices,
        )
    def _load_world(self, use_default: bool = True) -> Any:
        del use_default
        if self.collision_scene_manager is None:
            raise RuntimeError("TemplateController requires CollisionSceneManager")
        self.world_cfg = self.collision_scene_manager.build_world_config(
            self.reference_prim_path,
            ignore_substring=self.ignore_substring,
        )
        return self.world_cfg
    def reset(self, runtime: Any, get_ee_pose: Any) -> None:
        runtime.reset_attachments()
        self.collision_scene_manager.sync_dynamic_poses(
            self.execution_state.step_idx, interval_steps=1, force=False
        )
        self.collision_scene_manager.audit_controller(self.scene_port)
        self.phase_executor.clear()
        self.execution_state.reset()
        if self.lr_name == "left":
            self.execution_state.gripper_state = (
                1.0 if self.robot.left_gripper_state == 1.0 else -1.0
            )
        elif self.lr_name == "right":
            self.execution_state.gripper_state = (
                1.0 if self.robot.right_gripper_state == 1.0 else -1.0
            )
        self._resolve_robot_frame_paths(self.robot, self.lr_name)
        self.T_base_ee_init = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_world_base_init = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_world_ee_init = self.T_world_base_init @ self.T_base_ee_init
        self.execution_state.ee_trans, self.execution_state.ee_ori = get_ee_pose()
        self.execution_state.ee_trans = self.tensor_args.to_device(
            self.execution_state.ee_trans
        )
        self.execution_state.ee_ori = self.tensor_args.to_device(
            self.execution_state.ee_ori
        )
