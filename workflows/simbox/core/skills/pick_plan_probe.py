"""Planning-only Pick probe that emits a structured planner result."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from core.skills.base_skill import register_skill
from core.skills.pick import Pick
from core.utils.camera_template import resolve_camera_template_pose


@register_skill
class PickPlanProbe(Pick):
    """Run Pick's direct runtime planning path without executing commands."""

    @staticmethod
    def _yaw_deg(quaternion) -> float:
        w, x, y, z = [float(value) for value in quaternion]
        return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @staticmethod
    def _angle_error_deg(left: float, right: float) -> float:
        return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)

    def _spawn_check(self) -> dict:
        expectation = dict(self.skill_cfg.get("spawn_expectation") or {})
        robot_position, robot_orientation = self.robot.get_world_pose()
        target_position, _ = self.pick_obj.get_world_pose()
        robot_position = np.asarray(robot_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        actual_yaw = self._yaw_deg(robot_orientation)
        expected_robot_xy = np.asarray(expectation.get("robot_world_xy", robot_position[:2]), dtype=float)
        expected_yaw = float(expectation.get("robot_yaw_deg", actual_yaw))
        expected_target = expectation.get("target_world_xyz")
        target_linear_velocity = np.asarray(
            self.pick_obj.get_linear_velocity(), dtype=float
        )
        target_angular_velocity = np.asarray(
            self.pick_obj.get_angular_velocity(), dtype=float
        )
        robot_joint_velocity = np.asarray(
            self.robot.get_joint_velocities(), dtype=float
        )
        max_target_linear_speed = float(np.max(np.abs(target_linear_velocity)))
        max_target_angular_speed = float(np.max(np.abs(target_angular_velocity)))
        max_robot_joint_speed = float(np.max(np.abs(robot_joint_velocity)))
        unexpected_robot_contact = 0.0
        unexpected_object_contact = 0.0
        allowed_support_contact = 0.0
        collision_world_exact = False
        collision_world_error = None
        runtime = self._require_skill_runtime()
        manager = runtime.robot_port.collision_scene_manager
        port = getattr(runtime, "scene_port", None)
        if manager is not None and port is not None:
            try:
                manager.sync_dynamic_poses(0, interval_steps=1, force=True)
                manager.audit_controller(port)
                unexpected_robot_contact = manager.get_unexpected_robot_contact_force(
                    str(port.name), str(port.lr_name)
                )
                allowed_support_contact, unexpected_object_contact = (
                    manager.get_object_environment_contact_forces(
                        self.pick_obj.name,
                        str(expectation.get("target_support") or "") or None,
                    )
                )
                manager.assert_invariants()
                collision_world_exact = True
            except Exception as exc:
                collision_world_error = str(exc)
        robot_xy_error = float(np.linalg.norm(robot_position[:2] - expected_robot_xy))
        robot_yaw_error = self._angle_error_deg(actual_yaw, expected_yaw)
        target_xy_error = None
        if expected_target is not None:
            target_xy_error = float(
                np.linalg.norm(target_position[:2] - np.asarray(expected_target, dtype=float)[:2])
            )
        finite = bool(
            np.isfinite(robot_position).all()
            and np.isfinite(target_position).all()
            and np.isfinite(target_linear_velocity).all()
            and np.isfinite(target_angular_velocity).all()
            and np.isfinite(robot_joint_velocity).all()
            and math.isfinite(actual_yaw)
        )
        stable = bool(
            finite
            and robot_xy_error <= float(expectation.get("robot_xy_tolerance_m", 0.05))
            and robot_yaw_error <= float(expectation.get("robot_yaw_tolerance_deg", 5.0))
            and (
                target_xy_error is None
                or target_xy_error <= float(expectation.get("target_xy_tolerance_m", 0.25))
            )
            and max_target_linear_speed
            <= float(expectation["max_object_linear_speed_m_s"])
            and max_target_angular_speed
            <= float(expectation["max_object_angular_speed_rad_s"])
            and max_robot_joint_speed
            <= float(expectation["max_robot_joint_speed_rad_s"])
            and unexpected_robot_contact
            <= float(expectation["max_unexpected_contact_n"])
            and unexpected_object_contact
            <= float(expectation["max_unexpected_contact_n"])
            and collision_world_exact
        )
        return {
            "stable": stable,
            "finite": finite,
            "expected_robot_world_xy": expected_robot_xy.tolist(),
            "actual_robot_world_xyz": robot_position.tolist(),
            "robot_xy_error_m": robot_xy_error,
            "expected_robot_yaw_deg": expected_yaw,
            "actual_robot_yaw_deg": actual_yaw,
            "robot_yaw_error_deg": robot_yaw_error,
            "expected_target_world_xyz": expected_target,
            "actual_target_world_xyz": target_position.tolist(),
            "target_xy_error_m": target_xy_error,
            "target_linear_velocity_m_s": target_linear_velocity.tolist(),
            "target_angular_velocity_rad_s": target_angular_velocity.tolist(),
            "robot_joint_velocity_rad_s": robot_joint_velocity.tolist(),
            "max_target_linear_speed_m_s": max_target_linear_speed,
            "max_target_angular_speed_rad_s": max_target_angular_speed,
            "max_robot_joint_speed_rad_s": max_robot_joint_speed,
            "allowed_support_contact_n": allowed_support_contact,
            "unexpected_object_contact_n": unexpected_object_contact,
            "unexpected_robot_contact_n": unexpected_robot_contact,
            "target_support": expectation.get("target_support"),
            "collision_world_exact": collision_world_exact,
            "collision_world_error": collision_world_error,
            "thresholds": {
                key: expectation[key]
                for key in (
                    "max_object_linear_speed_m_s",
                    "max_object_angular_speed_rad_s",
                    "max_robot_joint_speed_rad_s",
                    "max_unexpected_contact_n",
                )
            },
        }

    def _write_result(self, result: dict) -> None:
        result_path = Path(str(self.skill_cfg["result_path"])).expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result.update(
            {
                "candidate_id": str(self.skill_cfg["candidate_id"]),
                "target": str(self.skill_cfg["objects"][0]),
            }
        )
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(result_path)

    def _capture_diagnostics(self, spawn_check: dict) -> dict | None:
        capture = dict(self.skill_cfg.get("diagnostic_capture") or {})
        if not capture:
            return None
        output_dir = Path(str(capture["output_dir"])).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        evidence = {
            "output_dir": str(output_dir),
            "overview": {"status": "not_requested", "artifact": None},
            "trajectory": {
                "status": "not_requested",
                "artifact": None,
                "segments": [],
            },
        }
        if capture.get("trajectory"):
            trajectory = evidence["trajectory"]
            runtime = self._require_skill_runtime()
            visualizer = getattr(runtime, "trajectory_visualizer", None)
            if visualizer is None:
                trajectory["status"] = "unavailable"
                trajectory["error"] = "trajectory visualizer is not enabled"
            elif self.plan_state is None:
                trajectory["status"] = "unavailable"
                trajectory["error"] = "grasp planning did not run"
            else:
                try:
                    trajectory["segments"] = [
                        name
                        for name, path in (
                            ("pregrasp", self.plan_state.get("pregrasp_path")),
                            ("terminal_grasp", self.plan_state.get("terminal_path")),
                        )
                        if path is not None
                    ]
                    artifact = (
                        visualizer.export(output_dir)
                        if trajectory["segments"]
                        else None
                    )
                    if artifact is not None:
                        trajectory["status"] = "written"
                        trajectory["artifact"] = str(artifact.resolve())
                    else:
                        trajectory["status"] = "no_selected_path"
                except Exception as exc:
                    trajectory["status"] = "error"
                    trajectory["error"] = str(exc)
        if capture.get("overview"):
            overview = evidence["overview"]
            camera = dict(capture.get("camera") or {})
            try:
                from core.utils.camera_utils import capture_topdown_screenshot

                resolution = camera.get("resolution") or [1280, 960]
                if camera.get("template"):
                    camera.update(
                        resolve_camera_template_pose(
                            str(camera["template"]),
                            spawn_check["actual_robot_world_xyz"],
                            spawn_check["actual_robot_yaw_deg"],
                            spawn_check["actual_target_world_xyz"],
                            camera.get("template_params"),
                            camera.get("room_bounds_xy"),
                        )
                    )
                capture_topdown_screenshot(
                    str(output_dir),
                    eye=camera.get("eye"),
                    target=camera.get("target"),
                    width=int(resolution[0]),
                    height=int(resolution[1]),
                    focal_length_mm=float(camera.get("focal_length_mm", 16.0)),
                    filename="overview.png",
                )
                artifact = output_dir / "overview.png"
                overview["status"] = "written"
                overview["artifact"] = str(artifact.resolve())
            except Exception as exc:
                overview["status"] = "error"
                overview["error"] = str(exc)
        return evidence

    def generate_manip_cmds(self):
        runtime = self._require_skill_runtime()
        spawn_check = self._spawn_check()
        target_only = bool(self.skill_cfg.get("diagnostic_target_only_world", False))
        empty_world = bool(self.skill_cfg.get("diagnostic_empty_world", False))
        manager = runtime.robot_port.collision_scene_manager
        port = getattr(runtime, "scene_port", None)
        raw_curobo_only_paths = self.skill_cfg.get(
            "diagnostic_disable_curobo_obstacle_paths", []
        )
        raw_physics_and_curobo_paths = self.skill_cfg.get(
            "diagnostic_disable_physics_and_curobo_obstacle_paths", []
        )
        raw_collision_entities = self.skill_cfg.get(
            "diagnostic_disable_collision_entities", []
        )
        for field, values in (
            ("diagnostic_disable_curobo_obstacle_paths", raw_curobo_only_paths),
            (
                "diagnostic_disable_physics_and_curobo_obstacle_paths",
                raw_physics_and_curobo_paths,
            ),
            ("diagnostic_disable_collision_entities", raw_collision_entities),
        ):
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{field} must be a list")
        curobo_only_paths = [
            str(value).strip()
            for value in raw_curobo_only_paths
        ]
        physics_and_curobo_paths = [
            str(value).strip()
            for value in raw_physics_and_curobo_paths
        ]
        collision_entities = [
            str(value).strip()
            for value in raw_collision_entities
        ]
        resolved_entities = {}
        if collision_entities:
            if manager is None:
                raise RuntimeError("diagnostic entity isolation requires a scene manager")
            resolved_entities = manager.resolve_diagnostic_collision_entities(
                collision_entities
            )
            for paths in resolved_entities.values():
                for path in paths:
                    if path not in physics_and_curobo_paths:
                        physics_and_curobo_paths.append(path)
            self.task.cfg.setdefault("metadata", {}).setdefault(
                "workspace_probe", {}
            )["diagnostic_resolved_collision_entities"] = resolved_entities
        active_modes = sum(
            bool(value)
            for value in (
                curobo_only_paths,
                physics_and_curobo_paths,
                target_only,
                empty_world,
            )
        )
        if active_modes > 1:
            raise ValueError(
                "CuRobo-only, Physics+CuRobo, target-only, and empty-world "
                "diagnostic isolation modes are mutually exclusive"
            )
        if (active_modes or collision_entities) and (manager is None or port is None):
            raise RuntimeError(
                "diagnostic collision isolation requires a typed scene port"
            )
        if target_only or empty_world:
            target_paths = set(manager.records[self.pick_obj.name].collision_prim_paths)
            enabled = manager.controller_enabled[(str(port.name), str(port.lr_name))]
            curobo_only_paths = [
                path
                for path, is_enabled in enabled.items()
                if is_enabled and (empty_world or path not in target_paths)
            ]
        if curobo_only_paths:
            diagnostic_context = manager.diagnostic_curobo_obstacles_disabled(
                port, curobo_only_paths
            )
        elif physics_and_curobo_paths:
            diagnostic_context = (
                manager.diagnostic_physics_and_curobo_obstacles_disabled(
                    port, physics_and_curobo_paths
                )
            )
        else:
            diagnostic_context = nullcontext()
        if spawn_check["stable"]:
            with diagnostic_context:
                super().generate_manip_cmds()
            result = dict(self.plan_state["result"]) if self.plan_state is not None else {
                "feasible": False,
                "failure_code": "PROBE_DID_NOT_RUN",
            }
            result["diagnostic_target_only_world"] = target_only
            result["diagnostic_empty_world"] = empty_world
            result["diagnostic_world_override_ignored"] = False
        else:
            result = {
                "feasible": False,
                "arm": runtime.arm_name,
                "grasp_count": 0,
                "pregrasp_success_count": 0,
                "grasp_success_count": 0,
                "joint_success_count": 0,
                "selected_grasp_index": None,
                "selected_grasp_score": None,
                "failure_code": "PROBE_SPAWN_UNSTABLE",
            }
        result["diagnostic_disabled_curobo_obstacle_paths"] = curobo_only_paths
        result["diagnostic_disabled_physics_and_curobo_obstacle_paths"] = (
            physics_and_curobo_paths
        )
        result["diagnostic_resolved_collision_entities"] = resolved_entities
        result["spawn_check"] = spawn_check
        diagnostic_evidence = self._capture_diagnostics(spawn_check)
        if diagnostic_evidence is not None:
            result["diagnostic_evidence"] = diagnostic_evidence
        self._write_result(result)
        # Keep the standard controller contract valid without moving the robot
        # or changing the gripper.  The external validator ends the process
        # once both arm JSONs exist.
        self.manip_list = [self.measured_hold_command()]

    def is_success(self):
        # Probe execution success is separate from grasp feasibility, which is
        # recorded in the structured result JSON.
        return True

    def is_feasible(self, th=5):
        return True

    def is_record(self):
        return False
