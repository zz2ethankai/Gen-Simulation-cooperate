"""Controller setup, scene synchronization, and telemetry operations."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import Any

import numpy as np
from curobo.types import JointState
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform

from core.controllers.curobo.components import ComponentState
from core.planning.domain_types import BatchPlanResult, PlanResult
from core.utils.joint_index_resolver import JointIndexResolutionError, resolve_joint_names
from core.utils.json_utils import json_ready, joint_state_json_ready
from core.controllers.curobo.trajectory import execution_trajectory_tensor
from core.visualization.curobo_trajectory import (
    CuroboTrajectoryPlannerAdapter,
    TrajectoryVisualizationFrame,
)

LOGGER = logging.getLogger("de_logger")


class ControllerSetup(ComponentState):
    @staticmethod
    def _joint_state_derivatives(sim_js):
        """Return finite measured derivatives for debug snapshots."""

        positions = sim_js.positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        size = np.asarray(positions, dtype=float).reshape(-1).size

        def field(name):
            value = getattr(sim_js, name, None)
            if value is None:
                return np.zeros(size, dtype=float)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size != size or not np.all(np.isfinite(value)):
                return np.zeros(size, dtype=float)
            return value.copy()

        return field("velocities"), field("accelerations"), field("jerks")

    @property
    def task_root_prim_path(self) -> str:
        """Expose only the task frame root needed by reference-frame consumers.

        Components must not retain the whole task just to resolve a transform.
        Setup already owns the task dependency, so publish this narrow value
        for the explicit component wiring after frame paths are resolved.
        """

        return self.task.root_prim_path

    def _configure_execution_stride(self) -> None:
        physics_dt = float(self.world.get_physics_dt())
        interpolation_dt = float(self.interpolation_dt)
        requested_stride = max(1, int(round(physics_dt / interpolation_dt)))
        safety_cfg = self.task.cfg.get("planning", {}).get("execution_safety", {})
        max_stride = max(1, int(safety_cfg.get("max_waypoint_stride", 2)))
        self.ds_ratio = min(requested_stride, max_stride)

    def _configure_joint_indices(self, robot_file: str) -> None:
        """Resolve runtime indices from the subclass's :class:`ArmSpec`."""

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
        self._gripper_state = (
            1.0
            if float(getattr(self.robot, f"{arm}_gripper_state")) >= 0.0
            else -1.0
        )
        configured_home = getattr(self.robot, f"{arm}_gripper_home", None)
        self._gripper_joint_position = np.asarray(
            configured_home
            if configured_home is not None
            else self.arm_spec.gripper_home,
            dtype=float,
        ).reshape(-1)

    def _resolve_robot_frame_paths(self, robot_view: Any, arm: str) -> None:
        """Resolve the selected arm's base and EE paths before port wiring."""

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
        """Resolve arm actuation indices by name and fail before any command can be misrouted."""

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
            self.reference_prim_path
        )
        return self.world_cfg

    @staticmethod
    def _plan_success_count(result) -> int:
        if isinstance(result, (PlanResult, BatchPlanResult)):
            return result.success_count
        raise TypeError(
            "plan diagnostics require a normalized PlanResult or BatchPlanResult"
        )

    @staticmethod
    def _error_range(result, name: str):
        if not isinstance(result, (PlanResult, BatchPlanResult)):
            raise TypeError(
                "plan diagnostics require a normalized PlanResult or BatchPlanResult"
            )
        values = result.metrics.get(name)
        if values is None:
            aliases = {
                "position_error": "position_errors",
                "rotation_error": "rotation_errors",
            }
            values = result.metrics.get(aliases.get(name, name))
        if values is None:
            return None
        try:
            values = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size == 0:
            return None
        return float(values.min()), float(values.max())

    def _log_plan_result(self, context: str, result, target=None):
        if not isinstance(result, (PlanResult, BatchPlanResult)):
            raise TypeError(
                "plan diagnostics require a normalized PlanResult or BatchPlanResult"
            )
        success_count = self._plan_success_count(result)
        pos_range = self._error_range(result, "position_error")
        rot_range = self._error_range(result, "rotation_error")
        status = result.status
        valid_query = result.metrics.get("valid_query")
        debug_info = result.metrics.get("debug_info")
        msg = (
            f"[PlanDebug] {context} robot={self.name} arm={self.lr_name} command={self._last_command_name} "
            f"batch_capability={self.batch_capability} success_count={success_count} "
            f"status={status} valid_query={valid_query}"
        )
        if debug_info is not None:
            msg += f" debug_info={debug_info}"
        if target is not None:
            msg += f" target={np.array2string(np.asarray(target), precision=4, suppress_small=True)}"
        if pos_range is not None:
            msg += f" pos_error_range=({pos_range[0]:.6f}, {pos_range[1]:.6f})"
        if rot_range is not None:
            msg += f" rot_error_range=({rot_range[0]:.6f}, {rot_range[1]:.6f})"
        LOGGER.info(msg)

    def _visualize_selected_plan(self) -> None:
        if self.trajectory_visualizer is None or self.phase_executor.current is None:
            return
        try:
            planner = self.runtime.native_planner
            frame = TrajectoryVisualizationFrame(
                name=self.name,
                arm_name=self.lr_name,
                robot_base_path=self.robot_base_path,
                task_root_path=self.task_root_prim_path,
                planner=CuroboTrajectoryPlannerAdapter(
                    kinematics=planner.kinematics,
                    tensor_args=self.tensor_args,
                ),
            )
            self.trajectory_visualizer.record_plan(
                self.phase_executor.current,
                frame=frame,
                command=self._last_command_name,
            )
        except Exception:
            # The overlay is observational and must never change controller behavior.
            LOGGER.exception(
                "[TrajectoryDebug] failed to visualize robot=%s arm=%s command=%s",
                self.name,
                self.lr_name,
                self._last_command_name,
            )

    @staticmethod
    def _debug_norm_delta(a, b):
        if a is None or b is None:
            return None
        a_arr = np.asarray(a, dtype=float).reshape(-1)
        b_arr = np.asarray(b, dtype=float).reshape(-1)
        if a_arr.shape != b_arr.shape:
            return None
        return float(np.linalg.norm(a_arr - b_arr))

    def _write_curobo_plan_debug(
        self,
        *,
        result,
        sim_js,
        js_names,
        ee_trans,
        ee_ori,
        raw_plan,
        ordered_trajectory,
        branch: str,
        selected_path_index=None,
        selected_path_source: str = "",
    ) -> None:
        try:
            os.makedirs(self._curobo_plan_debug_dir, exist_ok=True)
            self._curobo_plan_debug_counter += 1
            current_full = np.asarray(sim_js.positions, dtype=float).copy()
            current_arm = current_full[self.arm_indices].copy()
            first_ordered = None
            last_ordered = None
            if ordered_trajectory is not None and len(ordered_trajectory) > 0:
                ordered_positions, _ordered_names = execution_trajectory_tensor(
                    ordered_trajectory,
                    self.tensor_args,
                    target_joint_names=self.raw_js_names,
                    context="plan debug trajectory",
                )
                first_ordered = ordered_positions[0].detach().cpu().numpy()
                last_ordered = ordered_positions[-1].detach().cpu().numpy()

            velocities, accelerations, jerks = self._joint_state_derivatives(sim_js)
            cu_js = JointState(
                position=self.tensor_args.to_device(sim_js.positions),
                velocity=self.tensor_args.to_device(velocities),
                acceleration=self.tensor_args.to_device(accelerations),
                jerk=self.tensor_args.to_device(jerks),
                joint_names=js_names,
            )
            cu_js_ordered = cu_js.reorder(self.cmd_js_names)

            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = (
                f"{timestamp}_{int((time.time() % 1) * 1000000):06d}_"
                f"{self.name}_{self.lr_name}_plan_{self._curobo_plan_debug_counter:04d}.json"
            )
            output_path = os.path.join(self._curobo_plan_debug_dir, filename)
            if not isinstance(result, (PlanResult, BatchPlanResult)):
                raise TypeError(
                    "plan debug snapshots require a normalized PlanResult or "
                    "BatchPlanResult"
                )
            result_metrics = result.metrics

            payload = {
                "schema_version": 1,
                "controller": {
                    "name": self.name,
                    "lr_name": self.lr_name,
                    "robot_file": self.robot_file,
                    "batch_capability": bool(self.batch_capability),
                    "branch": branch,
                    "selected_path_index": json_ready(selected_path_index),
                    "selected_path_source": selected_path_source,
                },
                "joint_mapping": {
                    "robot_dof_names": json_ready(js_names),
                    "cmd_js_names": json_ready(self.cmd_js_names),
                    "raw_js_names": json_ready(self.raw_js_names),
                    "arm_indices": json_ready(self.arm_indices),
                    "gripper_indices": json_ready(self.gripper_indices),
                    "idx_list": json_ready(self.idx_list),
                },
                "goal": {
                    "ee_translation": json_ready(ee_trans),
                    "ee_orientation": json_ready(ee_ori),
                },
                "input_current_state": {
                    "sim_js": joint_state_json_ready(sim_js),
                    "current_arm_sim_order": json_ready(current_arm),
                    "curobo_input_ordered_cmd_js_names": joint_state_json_ready(cu_js_ordered),
                },
                "result_summary": {
                    "success": json_ready(result.success),
                    "status": json_ready(result.status),
                    "error": json_ready(result.error),
                    "source": json_ready(result.source),
                    "selected_candidate_index": json_ready(
                        result.selected_candidate_index
                    ),
                    "valid_query": json_ready(result_metrics.get("valid_query")),
                    "position_error": json_ready(result_metrics.get("position_error")),
                    "rotation_error": json_ready(result_metrics.get("rotation_error")),
                    "cspace_error": json_ready(result_metrics.get("cspace_error")),
                    "optimized_dt": json_ready(result_metrics.get("optimized_dt")),
                    "interpolation_dt": json_ready(result_metrics.get("interpolation_dt")),
                    "path_buffer_last_tstep": json_ready(
                        result_metrics.get("path_buffer_last_tstep")
                    ),
                    "used_graph": json_ready(result_metrics.get("used_graph")),
                    "attempts": json_ready(result_metrics.get("attempts")),
                    "trajopt_attempts": json_ready(result_metrics.get("trajopt_attempts")),
                    "metrics": json_ready(result_metrics),
                },
                "continuity": {
                    "first_ordered_minus_current_arm_norm": self._debug_norm_delta(first_ordered, current_arm),
                    "last_ordered_minus_current_arm_norm": self._debug_norm_delta(last_ordered, current_arm),
                    "first_ordered_position": json_ready(first_ordered),
                    "last_ordered_position": json_ready(last_ordered),
                },
                "raw_plan": joint_state_json_ready(raw_plan),
                "ordered_trajectory": (
                    None
                    if ordered_trajectory is None
                    else {
                        "positions": json_ready(ordered_trajectory.positions),
                        "joint_names": json_ready(ordered_trajectory.joint_names),
                        "velocities": json_ready(ordered_trajectory.velocities),
                    }
                ),
            }
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
            print(
                "[curobo-plan-debug] Wrote plan debug: "
                f"{output_path}; first_delta={payload['continuity']['first_ordered_minus_current_arm_norm']}"
            )
        except Exception as exc:  # Debug writing must never break an episode.
            print(f"[curobo-plan-debug] Failed to write plan debug: {exc}")

    def update(self) -> None:
        if getattr(self, "_world_cleanup_failed", False):
            error = getattr(self, "_world_cleanup_error", None)
            message = (
                "CuRobo world cleanup failed; refusing world update and "
                "dynamic-pose sync until reset cleanup succeeds"
            )
            if error is None:
                raise RuntimeError(message)
            raise RuntimeError(message) from error
        self.collision_scene_manager.sync_dynamic_poses(
            self._step_idx, interval_steps=1, force=False
        )
        self.collision_scene_manager.audit_controller(self.scene_port)

    def _refresh_reference_world_for_planning(self) -> None:
        """Synchronize a moved mobile reference once before a CuRobo query."""

        manager = self.collision_scene_manager
        if manager is not None:
            manager.refresh_controller_reference_world(self.scene_port)
            manager.apply_controller_planning_exclusions(self.scene_port)

    def _clear_attached_object_state(self) -> None:
        if self.runtime.native_planner is None:
            return
        self._world_cleanup_failed = True
        try:
            self.runtime.reset_attachments()
            # Native v2 detach only changes the robot attachment spheres and
            # obstacle enable masks.  It does not require a graph reset or a
            # full world reload.
            self._world_cache_invalidated = False
        except Exception as exc:
            self._world_cleanup_error = exc
            raise
        self._world_cleanup_failed = False
        self._world_cleanup_error = None

    def reset(self) -> None:
        self._clear_attached_object_state()
        self.update()
        self.phase_executor.clear()
        self.execution_state.reset()
        if self.lr_name == "left":
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif self.lr_name == "right":
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        self._resolve_robot_frame_paths(self.robot, self.lr_name)
        self.T_base_ee_init = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_world_base_init = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_world_ee_init = self.T_world_base_init @ self.T_base_ee_init
        self._ee_trans, self._ee_ori = self._get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(self._ee_trans)
        self._ee_ori = self.tensor_args.to_device(self._ee_ori)
