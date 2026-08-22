"""Native state conversion and diagnostic planning operations."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from curobo.types import GoalToolPose, JointState
from core.controllers.controller_component import ComponentState
from core.controllers.trajectory_boundary import normalize_named_trajectory
from core.planning.domain_types import BatchPlanResult, JointTrajectory, PlanResult

LOGGER = logging.getLogger("de_logger")


class ControllerStatePlanning(ComponentState):
    @staticmethod
    def _joint_state_derivatives(sim_js):
        """Return finite joint velocity/acceleration/jerk arrays for planning.

        Isaac articulation states expose velocity consistently, while higher
        derivatives are optional across simulator versions.  Preserving the
        measured derivatives when available lets a new Pick/Place phase start
        with the actual motion state instead of an artificial full stop.
        """

        positions = sim_js.positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float).reshape(-1)
        size = positions.size

        def _field(name):
            value = getattr(sim_js, name, None)
            if value is None:
                return np.zeros(size, dtype=float)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size != size or not np.all(np.isfinite(value)):
                return np.zeros(size, dtype=float)
            return value.copy()

        return _field("velocities"), _field("accelerations"), _field("jerks")

    def _planner_joint_names(self) -> list[str]:
        return list(self.runtime.native_planner.joint_names)

    def _arm_joint_state(self, sim_js, *, repeat=1):
        """Build a position-only native-v2 start state in planner joint order.

        CuRobo plans from a kinematic configuration.  Isaac's instantaneous
        articulation derivatives describe the previous control step (and can
        contain contact-induced spikes), not boundary conditions for the new
        trajectory.  Supplying them here makes TrajOpt reject otherwise
        collision-free post-grasp paths as infeasible.  Native v1 planning
        also used zero derivatives at every phase boundary.
        """

        positions = np.asarray(sim_js.positions, dtype=float).reshape(-1)
        arm_names = list(self.raw_js_names)
        if len(arm_names) != len(self.arm_indices):
            raise ValueError("raw arm joint names and runtime arm indices have different lengths")
        arm_positions = positions[self.arm_indices]
        zeros = np.zeros_like(arm_positions)
        state = JointState(
            position=self.tensor_args.to_device(arm_positions),
            velocity=self.tensor_args.to_device(zeros),
            acceleration=self.tensor_args.to_device(zeros),
            jerk=self.tensor_args.to_device(zeros),
            joint_names=arm_names,
        ).reorder(self._planner_joint_names())
        if repeat > 1:
            state = JointState(
                position=state.position.unsqueeze(0).expand(repeat, -1).clone(),
                velocity=state.velocity.unsqueeze(0).expand(repeat, -1).clone(),
                acceleration=state.acceleration.unsqueeze(0).expand(repeat, -1).clone(),
                jerk=state.jerk.unsqueeze(0).expand(repeat, -1).clone(),
                joint_names=state.joint_names,
            )
        return state

    def _planner_state(self, state):
        names = getattr(state, "joint_names", None)
        if names is None:
            raise ValueError("native CuRobo planning states require explicit joint_names")
        names = list(names)
        if set(self._planner_joint_names()) - set(names):
            raise ValueError(
                "planning state does not contain all native planner joints: "
                f"required={self._planner_joint_names()}, got={names}"
            )
        return state.reorder(self._planner_joint_names())

    def _goal_tool_pose(self, ee_translation, ee_orientation, batch_size=1):
        position = self.tensor_args.to_device(ee_translation)
        quaternion = self.tensor_args.to_device(ee_orientation)
        if batch_size == 1:
            position = position.reshape(1, 1, 1, 1, 3)
            quaternion = quaternion.reshape(1, 1, 1, 1, 4)
        else:
            position = position.reshape(batch_size, 1, 1, 1, 3)
            quaternion = quaternion.reshape(batch_size, 1, 1, 1, 4)
        return GoalToolPose(
            tool_frames=[self.runtime.native_planner.tool_frames[0]],
            position=position,
            quaternion=quaternion,
        )

    @staticmethod
    def _result_success(result) -> bool:
        if isinstance(result, BatchPlanResult):
            return result.is_success
        if isinstance(result, PlanResult):
            return result.success
        raise TypeError(
            "controller planning queries require a normalized PlanResult or "
            "BatchPlanResult"
        )

    def _result_path(self, result, batch_index=0):
        if isinstance(result, BatchPlanResult):
            paths = result.trajectories
        elif isinstance(result, PlanResult):
            paths = (result.trajectory,)
        else:
            raise TypeError(
                "controller planning queries require a normalized PlanResult or "
                "BatchPlanResult"
            )
        if batch_index >= len(paths) or paths[batch_index] is None:
            return None
        return paths[batch_index]

    def _command_path(self, path):
        """Convert a native/typed path to a named public arm trajectory.

        PlannerRuntime normally already publishes ``JointTrajectory``.  The
        native fallback remains for legacy adapters, but native ``JointState``
        values are converted immediately and never installed in the phase
        executor or returned to skills.
        """

        if path is None:
            return None
        names = list(getattr(path, "joint_names", ()) or ())
        trajectory = (
            path
            if isinstance(path, JointTrajectory)
            else JointTrajectory.from_native(path, joint_names=names)
        )
        names = list(trajectory.joint_names)
        if set(self.raw_js_names).issubset(names):
            return trajectory.reorder(self.raw_js_names)

        # ``get_full_js`` is a native-only kinematics adapter.  Convert the
        # public trajectory back to a native state locally, then normalize its
        # result again before it crosses back into the controller boundary.
        native_from_trajectory = getattr(self.runtime, "joint_state_from_trajectory", None)
        if callable(native_from_trajectory):
            active = native_from_trajectory(trajectory)
        else:
            positions, names = normalize_named_trajectory(
                trajectory.positions,
                trajectory.joint_names,
                self.tensor_args,
                context="native fallback trajectory",
            )
            active = JointState.from_position(positions, joint_names=list(names))
        active = self._planner_state(active)
        full_native = self.runtime.native_planner.kinematics.get_full_js(active)
        full_names = list(getattr(full_native, "joint_names", ()) or ())
        full = JointTrajectory.from_native(
            full_native,
            joint_names=full_names or self._planner_joint_names(),
        )
        if not set(self.raw_js_names).issubset(full.joint_names):
            raise ValueError(
                "native planner result cannot be mapped to controller arm names: "
                f"result={full.joint_names}, arm={self.raw_js_names}"
            )
        return full.reorder(self.raw_js_names)

    def _install_command_plan(
        self,
        trajectory,
        *,
        target_position=None,
        target_orientation=None,
        phase_name: str = "unknown",
        cached: bool,
    ):
        """Install one normalized trajectory for either planning or replay."""

        if trajectory is None or len(trajectory) == 0:
            raise ValueError(f"{phase_name} received an empty native-v2 path")
        self.idx_list = list(range(len(self.raw_js_names)))
        self.phase_executor.install(trajectory)
        self._phase_plan_started = True
        if target_position is not None:
            self._ee_trans = self.tensor_args.to_device(target_position)
        if target_orientation is not None:
            self._ee_ori = self.tensor_args.to_device(target_orientation)
        self._visualize_selected_plan()
        LOGGER.info(
            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=%s",
            self.name,
            self.lr_name,
            phase_name,
            len(self.phase_executor.current),
            self.ds_ratio,
            cached,
        )
        return trajectory

    def _plan_pose_from_state(
        self,
        ee_translation,
        ee_orientation,
        start_state,
        *,
        context: Optional[str] = None,
        request_metadata=None,
    ):
        if getattr(start_state.position, "ndim", 1) == 1:
            start_state = start_state.unsqueeze(0)
        result = self.runtime.plan_pose(
            ee_translation,
            ee_orientation,
            start_state=start_state,
            context=context,
            request_metadata=request_metadata,
        )
        if context:
            self._log_plan_result(context, result, target=ee_translation)
        return result

    def _plan_batch_from_state(
        self,
        ee_translation,
        ee_orientation,
        start_state,
        *,
        batch_size: Optional[int] = None,
        context: Optional[str] = None,
        request_metadata=None,
    ):
        if batch_size is None:
            batch_size = (
                1
                if getattr(start_state.position, "ndim", 1) == 1
                else int(start_state.position.shape[0])
            )
        result = self.runtime.plan_pose_batch(
            ee_translation,
            ee_orientation,
            start_state=start_state,
            batch_size=batch_size,
            context=context,
            request_metadata=request_metadata,
        )
        if context:
            self._log_plan_result(context, result)
        return result

    def _native_plan_pose_batch(
        self,
        ee_translation_goal_batch,
        ee_orientation_goal_batch,
        sim_js,
        js_names,
        request_metadata=None,
    ):
        del js_names
        batch_planner = self.runtime.ensure_batch_planner()
        # Materialize first so the reference refresh fans out to the target
        # batch adapter as part of the same strict scene transaction.
        self._refresh_reference_world_for_planning()
        batch_size = int(batch_planner.batch_size)
        # Native v2 keeps goal tensors on the planner device.  Normalize only
        # this host-side validation boundary before checking batch length;
        # ``np.asarray(cuda_tensor)`` cannot perform the device transfer.
        if isinstance(ee_translation_goal_batch, torch.Tensor):
            ee_translation_goal_batch = ee_translation_goal_batch.detach().cpu().numpy()
        else:
            ee_translation_goal_batch = np.asarray(ee_translation_goal_batch)
        if isinstance(ee_orientation_goal_batch, torch.Tensor):
            ee_orientation_goal_batch = ee_orientation_goal_batch.detach().cpu().numpy()
        else:
            ee_orientation_goal_batch = np.asarray(ee_orientation_goal_batch)
        actual_batch_size = len(ee_translation_goal_batch)
        if actual_batch_size < 1 or actual_batch_size > batch_size or len(ee_orientation_goal_batch) != actual_batch_size:
            raise ValueError(
                f"native batch planner accepts 1..{batch_size} candidate goals"
            )
        cu_js = self._arm_joint_state(sim_js, repeat=actual_batch_size)
        return self.runtime.plan_pose_batch(
            ee_translation_goal_batch,
            ee_orientation_goal_batch,
            start_state=cu_js,
            batch_size=actual_batch_size,
            request_metadata=request_metadata,
        )

    def _native_plan_pose(
        self,
        ee_translation_goal,
        ee_orientation_goal,
        sim_js: JointState,
        js_names: list,
        request_metadata=None,
    ):
        del js_names
        self._refresh_reference_world_for_planning()
        cu_js = self._arm_joint_state(sim_js)
        return self.runtime.plan_pose(
            ee_translation_goal,
            ee_orientation_goal,
            start_state=cu_js.unsqueeze(0),
            request_metadata=request_metadata,
        )

    def _native_plan_cspace(
        self,
        goal_arm_positions: np.ndarray,
        *,
        start_state=None,
        context=None,
        request_metadata=None,
    ):
        """Run the native c-space query (kept private behind ``plan_cspace``)."""

        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        goal_arm_positions = np.asarray(goal_arm_positions, dtype=float).reshape(-1)
        if len(goal_arm_positions) != len(self.arm_indices):
            raise ValueError(
                "goal_arm_positions must match the controller arm joint count: "
                f"got {len(goal_arm_positions)}, expected {len(self.arm_indices)}"
            )
        if start_state is None:
            sim_js = self.robot.get_joints_state()
            start_state = self._arm_joint_state(sim_js)
        else:
            start_state = self._planner_state(start_state)
        result = self.runtime.plan_cspace(
            goal_arm_positions,
            start_state=start_state.unsqueeze(0)
            if getattr(start_state.position, "ndim", 1) == 1
            else start_state,
            context=context,
            request_metadata=request_metadata,
        )
        self._log_plan_result(context or "plan_cspace", result, goal_arm_positions)
        return result

    def check_current_start_state(self):
        """Validate the live articulation state against the active planning world."""

        sim_js = self.robot.get_joints_state()
        velocities, accelerations, jerks = self._joint_state_derivatives(sim_js)
        start_state = self._arm_joint_state(sim_js)
        limits = self.runtime.native_planner.kinematics.get_joint_limits()
        position = start_state.position
        valid = bool(
            torch.isfinite(position).all().item()
            and (position >= limits.position_lower_limits).all().item()
            and (position <= limits.position_upper_limits).all().item()
        )
        return valid, "valid" if valid else "joint_limit_or_non_finite"

    def diagnose_native_start_collision(self) -> dict:
        """Report native-v2 scene collision at the live arm state.

        This is a failure-path diagnostic only.  It deliberately runs after a
        caller has exhausted its planning candidates, so the temporary
        one-step FK query cannot invalidate a subsequent CUDA-graph query.
        The query uses the same attached-object spheres and scene checker as
        the native planner; it never disables or changes a world obstacle.
        """

        try:
            from core.planning.native_bridge import CollisionBuffer

            sim_js = self.robot.get_joints_state()
            active_state = self._arm_joint_state(sim_js)
            fk_state = self.runtime.native_planner.compute_kinematics(
                JointState.from_position(
                    active_state.position.unsqueeze(0).unsqueeze(0),
                    joint_names=self._planner_joint_names(),
                )
            )
            if fk_state.robot_spheres is None:
                return {"available": False, "reason": "native_fk_has_no_robot_spheres"}

            scene = self.runtime.native_scene_adapter
            buffer = CollisionBuffer.from_shape(
                fk_state.robot_spheres.shape,
                self.tensor_args,
            )
            weight = self.tensor_args.to_device([1.0])
            activation_distance = self.tensor_args.to_device([0.0])
            distances = scene.get_sphere_distance(
                fk_state,
                buffer,
                weight,
                activation_distance,
            )
            distance_cpu = distances.detach().float().cpu().numpy()[0, 0]
            spheres_cpu = fk_state.robot_spheres.detach().float().cpu().numpy()[0, 0]
            collision_mask = distance_cpu > 1e-8

            kinematics_cfg = self.runtime.native_planner.kinematics.config.kinematics_config
            attached_indices = np.asarray([], dtype=int)
            if "attached_object" in (kinematics_cfg.link_name_to_idx_map or {}):
                attached_indices = (
                    kinematics_cfg.get_sphere_index_from_link_name("attached_object")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )
            attached_mask = np.zeros(len(distance_cpu), dtype=bool)
            attached_mask[attached_indices] = True

            def _sphere_rows(mask):
                indices = np.flatnonzero(mask)
                order = indices[np.argsort(distance_cpu[indices])[::-1]]
                return [
                    {
                        "sphere_index": int(index),
                        "collision_cost": float(distance_cpu[index]),
                        "center": spheres_cpu[index, :3].tolist(),
                        "radius": float(spheres_cpu[index, 3]),
                    }
                    for index in order[:8]
                ]

            summary = {
                "available": True,
                "attached_obstacle_names": list(
                    self.runtime.attachment_runtime.attached_obstacle_names
                ),
                "num_spheres": int(len(distance_cpu)),
                "collision_cost_sum": float(np.sum(distance_cpu)),
                "collision_cost_max": float(np.max(distance_cpu)) if len(distance_cpu) else 0.0,
                "collision_sphere_count": int(np.count_nonzero(collision_mask)),
                "attached_sphere_indices": attached_indices.tolist(),
                "attached_collision_cost_sum": float(np.sum(distance_cpu[attached_mask])),
                "attached_collision_cost_max": (
                    float(np.max(distance_cpu[attached_mask]))
                    if np.any(attached_mask)
                    else 0.0
                ),
                "colliding_spheres": _sphere_rows(collision_mask),
                "colliding_attached_spheres": _sphere_rows(collision_mask & attached_mask),
            }
            LOGGER.warning(
                "[NativeCollisionDebug] robot=%s arm=%s attached=%s "
                "collision_cost_sum=%.6f collision_cost_max=%.6f "
                "collision_spheres=%d attached_cost_max=%.6f",
                self.name,
                self.lr_name,
                summary["attached_obstacle_names"],
                summary["collision_cost_sum"],
                summary["collision_cost_max"],
                summary["collision_sphere_count"],
                summary["attached_collision_cost_max"],
            )
            return summary
        except Exception as exc:  # pragma: no cover - diagnostic must not mask planning failure
            LOGGER.warning(
                "[NativeCollisionDebug] unavailable robot=%s arm=%s error=%r",
                self.name,
                self.lr_name,
                exc,
            )
            return {"available": False, "reason": repr(exc)}
