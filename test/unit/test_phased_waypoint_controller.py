from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NAVIGATION_PATH = (
    REPO_ROOT / "workflows/simbox/core/skills/local_navigation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "simbox_phased_waypoint_controller",
    LOCAL_NAVIGATION_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PhasedWaypointController = MODULE.PhasedWaypointController


class PhasedWaypointControllerTest(unittest.TestCase):
    def setUp(self):
        self.goal = (0.0, 0.0, math.pi / 2.0)
        self.path = [
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
            {"x": 0.0, "y": -0.6, "yaw": math.pi / 2.0},
            {"x": 0.0, "y": -0.4, "yaw": math.pi / 2.0},
            {"x": 0.0, "y": -0.2, "yaw": math.pi / 2.0},
            {"x": 0.0, "y": 0.0, "yaw": math.pi / 2.0},
        ]
        self.controller = PhasedWaypointController(
            goal=self.goal,
            terminal_approach_distance_m=0.6,
            max_linear_velocity=0.35,
            terminal_max_linear_velocity=0.15,
            max_angular_velocity=0.5,
            waypoint_tolerance_m=0.12,
            position_tolerance_m=0.12,
            yaw_tolerance_rad=0.12,
            linear_gain=2.0,
            angular_gain=2.0,
            max_lateral_velocity=0.20,
            lateral_gain=2.0,
            lateral_alignment_enter_m=0.08,
            lateral_alignment_exit_m=0.04,
        )
        self.controller.reset(self.path)

    def test_turns_to_waypoint_before_walking(self):
        vx, vy, wz, done, debug = self.controller.command(
            (0.0, 0.0, 0.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertEqual((vx, vy), (0.0, 0.0))
        self.assertLess(wz, 0.0)
        self.assertEqual(debug["phase"], "track_path")
        self.assertEqual(debug["motion_mode"], "turn_to_waypoint")

    def test_walks_straight_after_waypoint_heading_is_aligned(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.0, 0.0, -math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertGreater(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["phase"], "track_path")
        self.assertEqual(debug["motion_mode"], "walk_straight")

    def test_moves_only_laterally_after_turning_to_segment_heading(self):
        vx, vy, wz, done, debug = self.controller.command(
            (0.10, 0.0, -math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertEqual(vx, 0.0)
        self.assertLess(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["motion_mode"], "lateral_align")

    def test_walks_forward_after_lateral_alignment(self):
        self.controller.command((0.10, 0.0, -math.pi / 2.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.02, 0.0, -math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertGreater(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["motion_mode"], "walk_straight")

    def test_recomputes_waypoint_heading_from_latest_pose(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.10, -0.05, -1.2),
            self.goal,
        )

        self.assertFalse(done)
        self.assertEqual((vx, vy), (0.0, 0.0))
        self.assertLess(wz, 0.0)
        self.assertEqual(debug["motion_mode"], "turn_to_waypoint")

    def test_aligns_final_heading_at_pre_goal_before_approach(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.0, -0.6, 0.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertEqual((vx, vy), (0.0, 0.0))
        self.assertGreater(wz, 0.0)
        self.assertEqual(debug["phase"], "align_final_approach")

    def test_walks_to_goal_after_pre_goal_alignment(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)
        self.controller.command((0.0, -0.6, 0.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.0, -0.55, math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertAlmostEqual(vx, 0.15)
        self.assertAlmostEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["phase"], "final_approach")
        self.assertEqual(debug["waypoint_index"], 2)

    def test_reacquires_position_after_final_alignment_drift(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)
        self.controller.command((0.0, -0.6, 0.0), self.goal)
        self.controller.command((0.0, -0.55, math.pi / 2.0), self.goal)
        self.controller.command((0.05, 0.05, 1.2), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.20, 0.0, math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertEqual(vx, 0.0)
        self.assertNotEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["phase"], "final_approach")
        self.assertEqual(debug["motion_mode"], "lateral_align")

    def test_advances_after_lateral_motion_passes_terminal_waypoint(self):
        self.controller.command((0.0, 0.0, 0.0), self.goal)
        self.controller.command((0.0, -0.6, math.pi / 2.0), self.goal)
        self.controller.command((0.10, -0.25, math.pi / 2.0), self.goal)

        vx, vy, wz, done, debug = self.controller.command(
            (0.03, -0.25, math.pi / 2.0),
            self.goal,
        )

        self.assertFalse(done)
        self.assertGreater(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(debug["waypoint_index"], 4)
        self.assertEqual(debug["motion_mode"], "walk_straight")

    def test_completes_only_when_position_and_yaw_match_together(self):
        _vx, _vy, _wz, done, debug = self.controller.command(
            (0.02, -0.02, math.pi / 2.0 - 0.03),
            self.goal,
        )

        self.assertTrue(done)
        self.assertEqual(debug["phase"], "done")


if __name__ == "__main__":
    unittest.main()
