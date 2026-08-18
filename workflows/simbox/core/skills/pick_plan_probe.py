"""Planning-only Pick probe that emits a structured CuRobo result."""

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
    """Run Pick's shared grasp evaluator without executing any commands."""

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
        manager = getattr(self.controller, "collision_scene_manager", None)
        if manager is not None:
            try:
                manager.sync_dynamic_poses(0, interval_steps=1, force=True)
                for controller in manager.controllers.values():
                    manager.audit_controller(controller)
                    unexpected_robot_contact = max(
                        unexpected_robot_contact,
                        manager.get_unexpected_robot_contact_force(
                            str(controller.name), str(controller.lr_name)
                        ),
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
            visualizer = getattr(self.controller, "trajectory_visualizer", None)
            evaluation = self.plan_evaluation
            if visualizer is None:
                trajectory["status"] = "unavailable"
                trajectory["error"] = "trajectory visualizer is not enabled"
            elif evaluation is None:
                trajectory["status"] = "unavailable"
                trajectory["error"] = "grasp planning did not run"
            else:
                try:
                    paths = (
                        ("pregrasp", evaluation.pregrasp_path),
                        ("terminal_grasp", evaluation.terminal_path),
                    )
                    for name, path in paths:
                        if path is None:
                            continue
                        full_path = self.controller.motion_gen.get_full_js(path)
                        full_path = full_path.get_ordered_joint_state(
                            self.controller.raw_js_names
                        )
                        visualizer.record_plan(self.controller, full_path, name)
                        trajectory["segments"].append(name)
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

    def simple_generate_manip_cmds(self):
        spawn_check = self._spawn_check()
        raw_curobo_only_paths = self.skill_cfg.get(
            "diagnostic_disable_curobo_obstacle_paths", []
        )
        raw_physics_and_curobo_paths = self.skill_cfg.get(
            "diagnostic_disable_physics_and_curobo_obstacle_paths", []
        )
        raw_collision_entities = self.skill_cfg.get(
            "diagnostic_disable_collision_entities", []
        )
        if not isinstance(raw_curobo_only_paths, (list, tuple)):
            raise ValueError(
                "diagnostic_disable_curobo_obstacle_paths must be a list of exact Prim paths"
            )
        if not isinstance(raw_physics_and_curobo_paths, (list, tuple)):
            raise ValueError(
                "diagnostic_disable_physics_and_curobo_obstacle_paths must be a list"
            )
        if not isinstance(raw_collision_entities, (list, tuple)):
            raise ValueError("diagnostic_disable_collision_entities must be a list")
        curobo_only_paths = [str(value).strip() for value in raw_curobo_only_paths]
        physics_and_curobo_paths = [
            str(value).strip() for value in raw_physics_and_curobo_paths
        ]
        collision_entities = [str(value).strip() for value in raw_collision_entities]
        resolved_entities: dict[str, list[str]] = {}
        if spawn_check["stable"]:
            manager = getattr(self.controller, "collision_scene_manager", None)
            target_only = bool(self.skill_cfg.get("diagnostic_target_only_world", False))
            empty_world = bool(self.skill_cfg.get("diagnostic_empty_world", False))
            dual_world_diagnostic = bool(physics_and_curobo_paths or collision_entities)
            active_modes = sum(
                bool(value)
                for value in (
                    curobo_only_paths,
                    dual_world_diagnostic,
                    target_only,
                    empty_world,
                )
            )
            if active_modes > 1:
                raise ValueError(
                    "CuRobo-only, Physics+CuRobo, target-only, and empty-world "
                    "diagnostic isolation modes are mutually exclusive"
                )
            if (curobo_only_paths or dual_world_diagnostic) and manager is None:
                raise RuntimeError(
                    "diagnostic collision isolation requires the physics_schema collision manager"
                )
            if collision_entities:
                resolved_entities = manager.resolve_diagnostic_collision_entities(
                    collision_entities
                )
                for paths in resolved_entities.values():
                    for path in paths:
                        if path not in physics_and_curobo_paths:
                            physics_and_curobo_paths.append(path)
                probe_metadata = self.task.cfg.setdefault("metadata", {}).setdefault(
                    "workspace_probe", {}
                )
                probe_metadata["diagnostic_resolved_collision_entities"] = (
                    resolved_entities
                )
            disabled_paths: list[str] = []
            original_begin_target_transit = None
            if (target_only or empty_world) and manager is not None:
                key = (str(self.controller.name), str(self.controller.lr_name))
                target_paths = set(manager.records[self.pick_obj.name].collision_prim_paths)
                enabled = manager.controller_enabled.get(key, {})
                disabled_paths = [
                    path
                    for path, is_enabled in enabled.items()
                    if is_enabled and (empty_world or path not in target_paths)
                ]
                for path in disabled_paths:
                    self.controller.motion_gen.world_collision.enable_obstacle(path, False)
                self._debug_log(
                    "diagnostic %s world disabled_obstacle_count=%d"
                    % ("empty" if empty_world else "target-only", len(disabled_paths))
                )
                if empty_world:
                    original_begin_target_transit = manager.begin_target_transit

                    def _begin_target_transit_without_world(entity_name, robot, arm):
                        record = original_begin_target_transit(entity_name, robot, arm)
                        for path in target_paths:
                            self.controller.motion_gen.world_collision.enable_obstacle(path, False)
                        return record

                    manager.begin_target_transit = _begin_target_transit_without_world
            if curobo_only_paths:
                diagnostic_context = manager.diagnostic_curobo_obstacles_disabled(
                    self.controller, curobo_only_paths
                )
            elif physics_and_curobo_paths:
                diagnostic_context = (
                    manager.diagnostic_physics_and_curobo_obstacles_disabled(
                        self.controller, physics_and_curobo_paths
                    )
                )
            else:
                diagnostic_context = nullcontext()
            try:
                with diagnostic_context:
                    super().simple_generate_manip_cmds()
            finally:
                if original_begin_target_transit is not None:
                    manager.begin_target_transit = original_begin_target_transit
                for path in disabled_paths:
                    self.controller.motion_gen.world_collision.enable_obstacle(path, True)
            result = self.plan_evaluation.result.to_dict() if self.plan_evaluation is not None else {
                "feasible": False,
                "failure_code": "PROBE_DID_NOT_RUN",
            }
            result["diagnostic_target_only_world"] = target_only
            result["diagnostic_empty_world"] = empty_world
        else:
            result = {
                "feasible": False,
                "arm": getattr(self.controller, "lr_name", None),
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
        position, orientation = self.controller.get_ee_pose()
        diagnostic_evidence = self._capture_diagnostics(spawn_check)
        if diagnostic_evidence is not None:
            result["diagnostic_evidence"] = diagnostic_evidence
        self._write_result(result)
        if diagnostic_evidence is not None:
            self.manip_list = []
            return
        # Keep the standard controller contract valid without moving the robot.
        # The external validator ends the process once both arm JSONs exist.
        self.manip_list = [
            (position, orientation, "update_pose_cost_metric", {"hold_vec_weight": None})
        ]

    def is_success(self):
        # Probe execution success is separate from grasp feasibility, which is
        # recorded in the structured result JSON.
        return True

    def is_feasible(self, th=5):
        return True

    def is_record(self):
        return False
