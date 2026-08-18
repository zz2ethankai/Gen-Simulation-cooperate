"""Planning-only Pick probe that emits a structured CuRobo result."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from core.skills.base_skill import register_skill
from core.skills.pick import Pick


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

    def simple_generate_manip_cmds(self):
        spawn_check = self._spawn_check()
        if spawn_check["stable"]:
            manager = getattr(self.controller, "collision_scene_manager", None)
            target_only = bool(self.skill_cfg.get("diagnostic_target_only_world", False))
            empty_world = bool(self.skill_cfg.get("diagnostic_empty_world", False))
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
                    self.controller.planner.scene_collision_checker.enable_obstacle(path, False)
                self._debug_log(
                    "diagnostic %s world disabled_obstacle_count=%d"
                    % ("empty" if empty_world else "target-only", len(disabled_paths))
                )
                if empty_world:
                    original_begin_target_transit = manager.begin_target_transit

                    def _begin_target_transit_without_world(entity_name, robot, arm):
                        record = original_begin_target_transit(entity_name, robot, arm)
                        for path in target_paths:
                            self.controller.planner.scene_collision_checker.enable_obstacle(path, False)
                        return record

                    manager.begin_target_transit = _begin_target_transit_without_world
            try:
                super().simple_generate_manip_cmds()
            finally:
                if original_begin_target_transit is not None:
                    manager.begin_target_transit = original_begin_target_transit
                for path in disabled_paths:
                    self.controller.planner.scene_collision_checker.enable_obstacle(path, True)
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
        result["spawn_check"] = spawn_check
        self._write_result(result)
        # Keep the standard controller contract valid without moving the robot.
        # The external validator ends the process once both arm JSONs exist.
        position, orientation = self.controller.get_ee_pose()
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
