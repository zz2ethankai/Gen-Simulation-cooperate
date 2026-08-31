from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NAVIGATION_PATH = (
    REPO_ROOT / "workflows/simbox/core/skills/local_navigation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "simbox_navigation_planning",
    LOCAL_NAVIGATION_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
build_navigation_plan = MODULE.build_navigation_plan


class NavigationPlanningTest(unittest.TestCase):
    def setUp(self):
        self.footprint = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2]]

    def test_terminal_approach_plans_to_pre_goal_and_appends_aligned_tail(self):
        planner = Mock()
        planner.plan.return_value = [(0.0, 0.0), (0.4, 1.0)]

        result = build_navigation_plan(
            start_pose=(0.0, 0.0, 0.0),
            goal=(1.0, 1.0, 0.0),
            static_map=None,
            footprint_points=self.footprint,
            planner_cfg={
                "terminal_approach_distance_m": 0.6,
                "terminal_approach_step_m": 0.2,
            },
            planner=planner,
        )

        planner.plan.assert_called_once_with((0.0, 0.0), (0.4, 1.0))
        self.assertEqual(result.goal, (1.0, 1.0, 0.0))
        self.assertEqual(len(result.path), 5)
        for waypoint, expected_x in zip(result.path[-3:], (0.6, 0.8, 1.0)):
            self.assertAlmostEqual(waypoint["x"], expected_x)
            self.assertAlmostEqual(waypoint["y"], 1.0)
            self.assertAlmostEqual(waypoint["yaw"], 0.0)

    def test_terminal_approach_supports_same_position_final_turn(self):
        planner = Mock()
        planner.plan.return_value = [(0.0, 0.0), (0.0, -0.6)]

        result = build_navigation_plan(
            start_pose=(0.0, 0.0, 0.0),
            goal=(0.0, 0.0, math.pi / 2.0),
            static_map=None,
            footprint_points=self.footprint,
            planner_cfg={
                "terminal_approach_distance_m": 0.6,
                "terminal_approach_step_m": 0.2,
            },
            planner=planner,
        )

        self.assertAlmostEqual(result.path[1]["x"], 0.0)
        self.assertAlmostEqual(result.path[1]["y"], -0.6)
        self.assertAlmostEqual(result.path[-1]["x"], 0.0)
        self.assertAlmostEqual(result.path[-1]["y"], 0.0)
        self.assertAlmostEqual(result.path[-1]["yaw"], math.pi / 2.0)

    def test_terminal_approach_rejects_negative_distance(self):
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            build_navigation_plan(
                start_pose=(0.0, 0.0, 0.0),
                goal=(1.0, 0.0, 0.0),
                static_map=None,
                footprint_points=self.footprint,
                planner_cfg={"terminal_approach_distance_m": -0.1},
            )


if __name__ == "__main__":
    unittest.main()
