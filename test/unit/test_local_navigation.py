from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "workflows/simbox/core/mobile/local_base_driver.py"
_DRIVER_SPEC = importlib.util.spec_from_file_location("simbox_local_base_driver", _DRIVER_PATH)
_DRIVER_MODULE = importlib.util.module_from_spec(_DRIVER_SPEC)
sys.modules[_DRIVER_SPEC.name] = _DRIVER_MODULE
_DRIVER_SPEC.loader.exec_module(_DRIVER_MODULE)
LocalVirtualBaseDriver = _DRIVER_MODULE.LocalVirtualBaseDriver

_LOCAL_NAV_PATH = Path(__file__).resolve().parents[2] / "workflows/simbox/core/skills/local_navigation.py"
_LOCAL_NAV_SPEC = importlib.util.spec_from_file_location("simbox_local_navigation", _LOCAL_NAV_PATH)
_LOCAL_NAV_MODULE = importlib.util.module_from_spec(_LOCAL_NAV_SPEC)
sys.modules[_LOCAL_NAV_SPEC.name] = _LOCAL_NAV_MODULE
_LOCAL_NAV_SPEC.loader.exec_module(_LOCAL_NAV_MODULE)
ApproachConfig = _LOCAL_NAV_MODULE.ApproachConfig
check_footprint_static_collision = _LOCAL_NAV_MODULE.check_footprint_static_collision
check_path_static_collision = _LOCAL_NAV_MODULE.check_path_static_collision
choose_best_reachable_candidate = _LOCAL_NAV_MODULE.choose_best_reachable_candidate
parse_approach_config = _LOCAL_NAV_MODULE.parse_approach_config
sample_approach_candidates = _LOCAL_NAV_MODULE.sample_approach_candidates
GridAStarPlanner = _LOCAL_NAV_MODULE.GridAStarPlanner
WaypointController = _LOCAL_NAV_MODULE.WaypointController


class _FakeVirtualRobot:
    def __init__(self):
        self.name = "fake_virtual"
        self.pose = (np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.commands = []
        self.hold_suspended = 0
        self.hold_resumed = 0
        self.base_cfg = {
            "platform": {
                "profile": "virtual_base",
                "local_navigation": {
                    "controller_hard_limits": {
                        "min_velocity": [-0.5, -0.4, -0.8],
                        "max_velocity": [0.5, 0.4, 0.8],
                    }
                },
            },
            "command_timeout": 1.0,
            "wheel_radius": 1.0,
            "steering_command_sign": 1.0,
            "base_velocity_joint_names": ["x", "y", "yaw"],
            "base_velocity_command_signs": [1.0, 1.0, 1.0],
        }

    def get_base_interface(self):
        return {
            "steering_joint_names": [],
            "wheel_joint_names": ["x", "y", "yaw"],
            "steering_joint_indices": [],
            "wheel_joint_indices": [0, 1, 2],
            "base_cfg": self.base_cfg,
        }

    def get_nav_base_pose(self):
        return self.pose

    def get_base_joint_state(self):
        return {
            "steering_positions": np.zeros(0, dtype=np.float32),
            "wheel_positions": np.zeros(3, dtype=np.float32),
            "steering_velocities": np.zeros(0, dtype=np.float32),
            "wheel_velocities": np.zeros(3, dtype=np.float32),
        }

    def apply_base_command(self, steering_positions, wheel_velocities, *, step_dt):
        self.commands.append((np.asarray(steering_positions), np.asarray(wheel_velocities), float(step_dt)))

    def suspend_manipulation_base_hold(self):
        self.hold_suspended += 1
        return True

    def resume_manipulation_base_hold(self):
        self.hold_resumed += 1


class LocalNavigationTests(unittest.TestCase):
    def test_approach_sampling_is_opt_in_and_faces_target(self):
        self.assertIsNone(parse_approach_config({}))
        config = parse_approach_config(
            {
                "approach": "apple",
                "approach_min_distance": 0.5,
                "approach_max_distance": 0.8,
                "approach_sample_count": 8,
            }
        )
        self.assertIsNotNone(config)
        candidates = sample_approach_candidates(config, (1.0, 2.0))
        self.assertEqual(len(candidates), 8)
        for candidate in candidates:
            expected = math.atan2(2.0 - candidate["y"], 1.0 - candidate["x"])
            self.assertAlmostEqual(
                math.atan2(math.sin(candidate["yaw"] - expected), math.cos(candidate["yaw"] - expected)),
                0.0,
                places=6,
            )

    def test_footprint_collision_checks_padding_and_unknown_cells(self):
        image = np.full((20, 20), 254, dtype=np.uint8)
        image[10, 10] = 0
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        footprint = [[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]]
        blocked = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.05,
            # PGM row 10 corresponds to map y=0.85 because row zero is the
            # top of the image while the YAML origin is the bottom-left.
            y=0.85,
        )
        self.assertFalse(blocked["ok"])
        clear = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.05,
            y=0.85,
            footprint_padding_m=0.2,
        )
        self.assertFalse(clear["ok"])

    def test_path_collision_reports_intermediate_pose(self):
        image = np.full((30, 30), 254, dtype=np.uint8)
        image[15, 10:20] = 0
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        footprint = [[-0.04, -0.04], [-0.04, 0.04], [0.04, 0.04], [0.04, -0.04]]
        result = check_path_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            path_poses=[
                {"x": 0.5, "y": 1.55, "yaw": 0.0},
                {"x": 2.5, "y": 1.55, "yaw": 0.0},
            ],
        )
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["first_blocked_index"])

    def test_candidate_selection_requires_path_success(self):
        candidates = [
            {"x": 0.0, "y": 0.0, "path_ok": False, "distance_to_target": 0.5},
            {"x": 1.0, "y": 0.0, "static_ok": True, "path_ok": True, "distance_to_target": 0.8},
            {"x": 2.0, "y": 0.0, "static_ok": True, "path_ok": True, "distance_to_target": 0.6},
        ]
        selected = choose_best_reachable_candidate(candidates)
        self.assertEqual(selected["x"], 2.0)

    def test_astar_routes_around_inflated_wall(self):
        image = np.full((30, 30), 254, dtype=np.uint8)
        image[:, 15] = 0
        image[2:8, 15] = 254
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        planner = GridAStarPlanner(safety_distance_m=0.0)
        planner.set_static_map(
            static_map,
            footprint_points=[[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]],
        )

        path = planner.plan((0.5, 1.5), (2.5, 1.5))

        self.assertIsNotNone(path)
        self.assertGreaterEqual(len(path), 2)
        self.assertTrue(any(abs(point[1] - 1.5) > 0.1 for point in path))

    def test_waypoint_controller_rotates_world_velocity_with_current_yaw(self):
        controller = WaypointController(max_linear_velocity=0.5, rotate_first_error_rad=math.pi)
        controller.reset([{"x": 0.0, "y": 0.0, "yaw": 0.0}, {"x": 1.0, "y": 0.0, "yaw": 0.0}])

        vx, vy, _, done, _ = controller.command((0.0, 0.0, math.pi / 2.0), (1.0, 0.0, math.pi / 2.0))

        self.assertFalse(done)
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertAlmostEqual(vy, -0.5, places=6)

    def test_virtual_driver_clamps_and_applies_local_twist(self):
        robot = _FakeVirtualRobot()
        world = SimpleNamespace(current_time=1.0)
        driver = LocalVirtualBaseDriver(robot, world=world)
        driver.prepare_for_navigation()
        driver.set_command(2.0, -2.0, 2.0)

        driver.step(step_dt=0.1)

        np.testing.assert_allclose(robot.commands[-1][1], [0.5, -0.4, 0.8])
        action = driver.get_logging_action_snapshot()
        self.assertEqual(action["vx_body"], 0.5)
        self.assertAlmostEqual(action["vy_body"], -0.4)
        self.assertAlmostEqual(action["wz_body"], 0.8)
        driver.finalize_after_navigation()
        self.assertEqual(robot.hold_suspended, 1)
        self.assertEqual(robot.hold_resumed, 1)

    def test_virtual_driver_respects_body_acceleration_and_deceleration(self):
        robot = _FakeVirtualRobot()
        limits = robot.base_cfg["platform"]["local_navigation"]["controller_hard_limits"]
        limits["max_accel"] = [1.0, 1.0, 1.0]
        limits["max_decel"] = [-2.0, -2.0, -2.0]
        world = SimpleNamespace(current_time=1.0)
        driver = LocalVirtualBaseDriver(robot, world=world)
        driver.prepare_for_navigation()
        driver.set_command(0.5, 0.0, 0.0)

        driver.step(step_dt=0.1)
        np.testing.assert_allclose(robot.commands[-1][1], [0.1, 0.0, 0.0])
        driver.step(step_dt=0.1)
        np.testing.assert_allclose(robot.commands[-1][1], [0.2, 0.0, 0.0])
        driver.set_command(0.0, 0.0, 0.0)
        driver.step(step_dt=0.1)
        np.testing.assert_allclose(robot.commands[-1][1], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
