"""Planning-only Pick probe that emits a structured planner result."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from core.skills.base_skill import register_skill
from core.skills.pick import Pick


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

    def generate_manip_cmds(self):
        runtime = self._require_skill_runtime()
        spawn_check = self._spawn_check()
        target_only = bool(self.skill_cfg.get("diagnostic_target_only_world", False))
        empty_world = bool(self.skill_cfg.get("diagnostic_empty_world", False))
        if spawn_check["stable"]:
            # Probes always use the canonical Physics-schema world.  The old
            # diagnostic flags used temporary obstacle toggles and monkey
            # patched scene transitions; probes must not mutate that world.
            super().generate_manip_cmds()
            result = self.plan_state["result"] if self.plan_state is not None else {
                "feasible": False,
                "failure_code": "PROBE_DID_NOT_RUN",
            }
            result["diagnostic_target_only_world"] = target_only
            result["diagnostic_empty_world"] = empty_world
            result["diagnostic_world_override_ignored"] = bool(target_only or empty_world)
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
        result["spawn_check"] = spawn_check
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
