from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NAVIGATION_PATH = (
    REPO_ROOT / "workflows/simbox/core/skills/local_navigation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "simbox_navigation_controller_factory",
    LOCAL_NAVIGATION_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NavigateControllerFactoryTest(unittest.TestCase):
    def setUp(self):
        self.build_controller = MODULE.build_navigation_controller
        self.request = {
            "controller_cfg": {
                "max_linear_velocity": 0.35,
                "max_angular_velocity": 0.5,
                "linear_gain": 2.0,
                "angular_gain": 2.0,
                "rotate_first_error_rad": 0.2,
            },
            "planner_cfg": {"terminal_approach_distance_m": 0.6},
            "goal": (1.0, 1.0, 0.0),
            "waypoint_tolerance_m": 0.25,
            "position_tolerance_m": 0.12,
            "yaw_tolerance_rad": 0.12,
        }

    def test_preserves_original_controller_when_type_is_not_configured(self):
        controller = self.build_controller(**self.request)

        self.assertIsInstance(controller, MODULE.WaypointController)
        self.assertEqual(controller.max_linear_velocity, 0.35)
        self.assertEqual(controller.max_angular_velocity, 0.5)
        self.assertEqual(controller.rotate_first_error_rad, 0.2)

    def test_selects_opt_in_phased_controller(self):
        self.request["controller_cfg"].update(
            {
                "controller_type": "phased_waypoint",
                "terminal_max_linear_velocity": 0.15,
                "max_lateral_velocity": 0.2,
            }
        )

        controller = self.build_controller(**self.request)

        self.assertIsInstance(controller, MODULE.PhasedWaypointController)
        self.assertEqual(controller.terminal_max_linear_velocity, 0.15)

    def test_rejects_unknown_controller_type(self):
        self.request["controller_cfg"]["controller_type"] = "unknown"

        with self.assertRaisesRegex(ValueError, "Unsupported navigation controller_type"):
            self.build_controller(**self.request)


if __name__ == "__main__":
    unittest.main()
