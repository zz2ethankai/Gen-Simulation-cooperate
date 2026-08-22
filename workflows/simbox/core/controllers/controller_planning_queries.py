"""Planning query helpers used by pick/place candidate evaluation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from core.controllers.controller_component import ComponentState
from core.controllers.trajectory_boundary import normalize_named_trajectory

LOGGER = logging.getLogger("de_logger")
JointState = None


class ControllerPlanningQueries(ComponentState):
    def _ensure_batch_planner(self):
        ensure = getattr(self.runtime, "ensure_batch_planner", None)
        if not callable(ensure):
            raise RuntimeError("batch planning was not enabled for this controller")
        return ensure()

    def _path_position_tensor(self, path):
        """Normalize a named trajectory through this component's DeviceCfg."""

        position = getattr(path, "position", None)
        if position is None:
            position = getattr(path, "positions", None)
        return normalize_named_trajectory(
            position,
            getattr(path, "joint_names", None),
            self.tensor_args,
        )

    def _path_endpoint_joint_state(self, path):
        """Build one public CuRobo JointState from a named path endpoint."""

        joint_state_type = JointState
        if joint_state_type is None:
            from curobo.types import JointState as joint_state_type
        position, names = self._path_position_tensor(path)
        return self._planner_state(
            joint_state_type.from_position(position[-1], joint_names=names)
        )

    def _plan_pose_batch_native(self, ee_trans_batch_np, ee_ori_batch_np):
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self._native_plan_pose_batch(ee_trans_batch, ee_ori_batch, sim_js, js_names)
        self._log_plan_result("plan_pose_batch", result)

        return result

    def _plan_pose_batch_from_paths(self, ee_trans_batch_np, ee_ori_batch_np, start_paths, *, context=None):
        """Plan each terminal target from its matching pre-grasp endpoint."""

        ensure_batch = getattr(self, "_ensure_batch_planner", None)
        batch_planner = (
            ensure_batch() if callable(ensure_batch) else self.runtime.batch_planner
        )
        if batch_planner is None:
            raise RuntimeError("batch planning was not enabled for this controller")
        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        if not start_paths:
            raise ValueError(
                "batch terminal planning requires at least one named pre-grasp path"
            )
        if len(ee_trans_batch_np) != len(ee_ori_batch_np) or len(ee_trans_batch_np) != len(start_paths):
            raise ValueError(
                "batch terminal planning requires equal goal-position, goal-orientation, "
                f"and pre-grasp-path counts: positions={len(ee_trans_batch_np)}, "
                f"orientations={len(ee_ori_batch_np)}, paths={len(start_paths)}"
            )

        terminal_positions = []
        expected_joint_name_set = None
        for path_index, path in enumerate(start_paths):
            if path is None:
                # Native batch results intentionally keep failed items as
                # ``None``.  The batch solver still needs a complete start
                # tensor; these fallback rows are masked by the pre-grasp
                # success mask in the evaluator and cannot become valid joint
                # candidates on their own.
                terminal_positions.append(self._arm_joint_state(self.robot.get_joints_state()))
                continue
            names = getattr(path, "joint_names", None)
            if names is None or isinstance(names, (str, bytes)):
                raise ValueError(
                    "batch pre-grasp endpoint must provide explicit joint_names: "
                    f"path_index={path_index}"
                )
            names = list(names)
            if not names or len(set(names)) != len(names):
                raise ValueError(
                    "batch pre-grasp endpoint joint_names must be non-empty and "
                    f"unique: path_index={path_index}, joint_names={names!r}"
                )
            name_set = set(names)
            if expected_joint_name_set is None:
                expected_joint_name_set = name_set
            elif name_set != expected_joint_name_set:
                raise ValueError(
                    "batch pre-grasp endpoints must use the same named joint contract: "
                    f"path_index={path_index}, expected={sorted(expected_joint_name_set)!r}, "
                    f"got={sorted(name_set)!r}"
                )

            position = getattr(path, "position", None)
            if position is None:
                raise ValueError(
                    "batch pre-grasp endpoint must provide position: "
                    f"path_index={path_index}"
                )
            if not isinstance(position, torch.Tensor):
                position = self.tensor_args.to_device(position)
            if position.ndim < 2 or position.shape[0] < 1:
                raise ValueError(
                    "batch pre-grasp endpoint position must be a non-empty "
                    "trajectory with shape [time, dof]: "
                    f"path_index={path_index}, position_shape={tuple(position.shape)}"
                )
            if position.shape[-1] != len(names):
                raise ValueError(
                    "batch pre-grasp endpoint position DOF count does not match "
                    f"its joint_names: path_index={path_index}, "
                    f"position_shape={tuple(position.shape)}, joint_names={names!r}"
                )

            terminal_positions.append(self._path_endpoint_joint_state(path))

        starts = torch.stack([state.position for state in terminal_positions])
        zeros = torch.zeros_like(starts)
        start_state = JointState(
            position=starts,
            velocity=zeros,
            acceleration=zeros,
            jerk=zeros,
            joint_names=self._planner_joint_names(),
        )
        if len(start_paths) < 1 or len(start_paths) > batch_planner.batch_size:
            raise ValueError(
                f"native batch planner accepts 1..{batch_planner.batch_size} paths"
            )
        return self._plan_batch_from_state(
            ee_trans_batch,
            ee_ori_batch,
            start_state,
            batch_size=len(start_paths),
            context=context or "plan_pose_batch_from_pregrasp",
        )

    def measure_cartesian_path(self, path, start_position, goal_position):
        """Return path/direct length ratio and maximum straight-line deviation."""

        path_position, path_names = self._path_position_tensor(path)

        # CuRobo's trajectory/result contract may be full (7 arm + 2 locked
        # fingers), while the kinematics model is deliberately active-arm
        # only.  Reduce by the explicit active names before FK; never slice a
        # nine-dimensional tensor positionally.
        active_names = list(self.raw_js_names)
        if set(path_names) != set(active_names) or path_names != active_names:
            reorder = getattr(path, "reorder", None)
            if not callable(reorder):
                raise ValueError(
                    "Cartesian path joint order/contract differs from active arm "
                    "names but the path cannot reorder by explicit names: "
                    f"path_names={path_names!r}, active_names={active_names!r}"
                )
            path = reorder(active_names)
            path_position, path_names = self._path_position_tensor(path)

        if path_position.shape[0] == 0:
            return float("inf"), float("inf")

        batch_forward_kinematic = getattr(self, "_forward_kinematic_batch", None)
        if not callable(batch_forward_kinematic):
            raise RuntimeError(
                "Cartesian path measurement requires the formal batched FK API"
            )
        positions = np.asarray(
            batch_forward_kinematic(path_position, joint_names=active_names),
            dtype=float,
        )
        if not len(positions):
            return float("inf"), float("inf")
        direct_vector = np.asarray(goal_position, dtype=float) - np.asarray(start_position, dtype=float)
        direct_length = float(np.linalg.norm(direct_vector))
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        if direct_length <= 1e-9:
            return (1.0 if path_length <= 1e-9 else float("inf")), path_length
        direction = direct_vector / direct_length
        relative = positions - np.asarray(start_position, dtype=float)
        projection = np.clip(relative @ direction, 0.0, direct_length)
        closest = np.asarray(start_position, dtype=float) + projection[:, None] * direction
        deviation = float(np.max(np.linalg.norm(positions - closest, axis=1)))
        return path_length / direct_length, deviation

    def plan_pose_from_joint_positions(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_arm_positions: Optional[np.ndarray] = None,
    ):
        """Plan without changing runtime controller state and return the path endpoint."""
        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        sim_js.positions = np.asarray(sim_js.positions, dtype=float).copy()
        if start_arm_positions is not None:
            start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
            if len(start_arm_positions) != len(self.arm_indices):
                raise ValueError(
                    "start_arm_positions must match the controller arm joint count: "
                    f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
                )
            sim_js.positions[self.arm_indices] = start_arm_positions

        result = self._native_plan_pose(ee_trans, ee_ori, sim_js, self.robot.dof_names)
        if not self._result_success(result):
            return False, None, result

        trajectory = self._command_path(self._result_path(result))
        if trajectory is None:
            return False, None, result
        trajectory_positions, _trajectory_names = self._path_position_tensor(trajectory)
        endpoint = trajectory_positions[-1]
        if hasattr(endpoint, "detach"):
            endpoint = endpoint.detach()
        if hasattr(endpoint, "cpu"):
            endpoint = endpoint.cpu()
        if hasattr(endpoint, "numpy"):
            endpoint = endpoint.numpy()
        end_arm_positions = np.asarray(endpoint, dtype=float)
        return True, end_arm_positions, result

    def plan_pose_result(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        *,
        request_metadata=None,
    ):
        """Return the single-plan result so callers can reuse its endpoint.

        The legacy boolean wrapper above remains available.  Physics-schema
        Pick needs the actual pre-grasp path because its terminal plan must
        start from that path's final joint state, not from the live initial
        articulation state.
        """

        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self._native_plan_pose(
            ee_trans,
            ee_ori,
            sim_js,
            js_names,
            request_metadata=request_metadata,
        )
        self._log_plan_result("plan_pose", result, target=ee_trans)
        return result

    def plan_pose_from_path(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_path,
        *,
        request_metadata=None,
    ):
        """Plan one terminal target from a successful pre-grasp endpoint."""

        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        start_state = self._path_endpoint_joint_state(start_path)
        return self._plan_pose_from_state(
            ee_trans,
            ee_ori,
            start_state,
            context="plan_pose_from_pregrasp",
            request_metadata=request_metadata,
        )
