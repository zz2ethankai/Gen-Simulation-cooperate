from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "workflows/simbox/core/mobile/local_base_driver.py"
_DRIVER_SPEC = importlib.util.spec_from_file_location("simbox_local_base_driver", _DRIVER_PATH)
_DRIVER_MODULE = importlib.util.module_from_spec(_DRIVER_SPEC)
sys.modules[_DRIVER_SPEC.name] = _DRIVER_MODULE
_DRIVER_SPEC.loader.exec_module(_DRIVER_MODULE)
LocalBaseDriver = _DRIVER_MODULE.LocalBaseDriver

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

    def get_mobile_base_pose(self):
        return self.pose

    def set_mobile_base_world_pose(self, translation, orientation):
        self.pose = (np.asarray(translation, dtype=np.float32), np.asarray(orientation, dtype=np.float32))

    def apply_base_command(self, steering_positions, wheel_velocities, *, step_dt=None):
        self.commands.append(
            {
                "steering_positions": np.asarray(steering_positions, dtype=np.float32).copy(),
                "wheel_velocities": np.asarray(wheel_velocities, dtype=np.float32).copy(),
                "step_dt": step_dt,
            }
        )

    def get_base_joint_state(self):
        return {
            "steering_positions": np.zeros(0, dtype=np.float32),
            "wheel_positions": np.zeros(3, dtype=np.float32),
            "steering_velocities": np.zeros(0, dtype=np.float32),
            "wheel_velocities": np.zeros(3, dtype=np.float32),
        }

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

    def test_center_collision_ignores_footprint_padding(self):
        image = np.full((20, 20), 254, dtype=np.uint8)
        image[10, 10] = 0
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        footprint = [[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]]
        blocked = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            # PGM row 10 corresponds to world y=0.86 with the current
            # image-pixel rounding convention.
            y=0.86,
        )
        self.assertFalse(blocked["ok"])
        clear = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=0.8,
            y=0.86,
            footprint_padding_m=0.2,
        )
        self.assertTrue(clear["ok"])

    def test_path_collision_checks_discrete_waypoints_only(self):
        image = np.full((30, 30), 254, dtype=np.uint8)
        image[15, 10:20] = 0
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        footprint = [[-0.04, -0.04], [-0.04, 0.04], [0.04, 0.04], [0.04, -0.04]]
        result = check_path_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            path_poses=[
                {"x": 0.5, "y": 1.36, "yaw": 0.0},
                {"x": 2.5, "y": 1.36, "yaw": 0.0},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["sampled_pose_count"], 2)
        self.assertEqual(result["interpolated_pose_count"], 0)

    def test_candidate_selection_requires_path_success(self):
        candidates = [
            {"x": 0.0, "y": 0.0, "path_ok": False, "distance_to_target": 0.5},
            {"x": 1.0, "y": 0.0, "static_ok": True, "path_ok": True, "distance_to_target": 0.8},
            {"x": 2.0, "y": 0.0, "static_ok": True, "path_ok": True, "distance_to_target": 0.6},
        ]
        selected = choose_best_reachable_candidate(candidates)
        self.assertEqual(selected["x"], 2.0)

    def test_approach_preflight_reuses_one_static_planner(self):
        static_map = {
            "image": np.full((20, 20), 254, dtype=np.uint8),
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
        }
        candidates = [
            {"x": 0.5, "y": 0.5, "yaw": 0.0, "distance_to_target": 0.5},
            {"x": 0.6, "y": 0.5, "yaw": 0.0, "distance_to_target": 0.6},
        ]
        successful_plan = SimpleNamespace(path=[{"x": 0.0, "y": 0.0, "yaw": 0.0}])
        with patch.object(_LOCAL_NAV_MODULE, "resolve_footprint_points", return_value=[[-0.1, -0.1], [0.1, 0.1]]), patch.object(
            _LOCAL_NAV_MODULE, "sample_approach_candidates", return_value=candidates
        ), patch.object(
            _LOCAL_NAV_MODULE, "check_footprint_static_collision", return_value={"ok": True}
        ), patch.object(
            _LOCAL_NAV_MODULE, "GridAStarPlanner"
        ) as planner_cls, patch.object(_LOCAL_NAV_MODULE, "build_navigation_plan", return_value=successful_plan) as build_plan:
            planner_cls.return_value.plan_to_goals.return_value = {
                0: [(0.0, 0.0), (0.5, 0.5)],
                1: [(0.0, 0.0), (0.6, 0.5)],
            }
            goal, debug = _LOCAL_NAV_MODULE.select_approach_goal(
                approach_config=ApproachConfig(target_name="tray"),
                target_xy=(1.0, 1.0),
                start_pose=(0.0, 0.0, 0.0),
                static_map=static_map,
                base_cfg={},
            )

        self.assertEqual(goal, (0.5, 0.5, 0.0))
        self.assertEqual(planner_cls.call_count, 1)
        planner_cls.return_value.set_static_map.assert_called_once()
        planner_cls.return_value.plan_to_goals.assert_called_once()
        self.assertEqual(planner_cls.return_value.plan_to_goals.call_args.kwargs["max_solutions"], 10)
        self.assertEqual(build_plan.call_count, 2)
        self.assertTrue(all(call.kwargs["planner"] is planner_cls.return_value for call in build_plan.call_args_list))
        self.assertTrue(debug["selected"]["path_ok"])

    def test_astar_routes_around_occupied_wall(self):
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

    def test_astar_preserves_intermediate_grid_waypoints(self):
        image = np.full((30, 30), 254, dtype=np.uint8)
        image[14:17, 14:17] = 0
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        planner = GridAStarPlanner(safety_distance_m=0.0)
        planner.set_static_map(static_map, footprint_points=[[-0.02, -0.02], [0.02, 0.02]])
        path = planner.plan((0.5, 0.5), (2.5, 2.5))
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)

    def test_multi_goal_astar_stops_after_requested_solution_count(self):
        image = np.full((30, 30), 254, dtype=np.uint8)
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        planner = GridAStarPlanner(safety_distance_m=0.0)
        planner.set_static_map(static_map, footprint_points=[[-0.02, -0.02], [0.02, 0.02]])

        paths = planner.plan_to_goals(
            (0.5, 0.5),
            [(2.5, 2.5), (2.0, 0.5), (1.0, 0.5)],
            max_solutions=2,
        )

        self.assertEqual(len(paths), 2)
        self.assertTrue(set(paths).issubset({0, 1, 2}))

    def test_multi_goal_astar_accepts_approach_yaw(self):
        image = np.full((20, 20), 254, dtype=np.uint8)
        static_map = {"image": image, "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}
        planner = GridAStarPlanner(safety_distance_m=0.0)
        planner.set_static_map(static_map, footprint_points=[[-0.02, -0.02], [0.02, 0.02]])

        paths = planner.plan_to_goals((0.5, 0.5), [(1.5, 1.5, 0.7), (1.0, 0.5, -1.2)], max_solutions=2)

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(len(point) == 2 for path in paths.values() for point in path))

    def test_navigation_plan_preserves_measured_start_yaw(self):
        start_pose = (0.0, 0.0, -0.7)
        goal = (1.0, 0.0, 1.2)
        static_map = {
            "image": np.full((20, 20), 254, dtype=np.uint8),
            "resolution": 0.1,
            "origin": [-1.0, -1.0, 0.0],
        }
        footprint = [[-0.2, -0.1], [-0.2, 0.1], [0.2, 0.1], [0.2, -0.1]]
        with patch.object(GridAStarPlanner, "plan", return_value=[(0.0, 0.0), (1.0, 0.0)]), patch(
            "simbox_local_navigation.check_path_static_collision", return_value={"ok": True}
        ) as check_path:
            plan = _LOCAL_NAV_MODULE.build_navigation_plan(
                start_pose=start_pose,
                goal=goal,
                static_map=static_map,
                footprint_points=footprint,
                planner_cfg={"safety_distance_m": 0.0},
            )

        self.assertIsNotNone(plan)
        poses = check_path.call_args.kwargs["path_poses"]
        self.assertAlmostEqual(poses[0]["yaw"], start_pose[2])
        self.assertAlmostEqual(poses[-1]["yaw"], goal[2])

    def test_waypoint_controller_rotates_world_velocity_with_current_yaw(self):
        controller = WaypointController(max_linear_velocity=0.5, rotate_first_error_rad=math.pi)
        controller.reset([{"x": 0.0, "y": 0.0, "yaw": 0.0}, {"x": 1.0, "y": 0.0, "yaw": 0.0}])

        vx, vy, _, done, _ = controller.command((0.0, 0.0, math.pi / 2.0), (1.0, 0.0, math.pi / 2.0))

        self.assertFalse(done)
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertAlmostEqual(vy, -0.5, places=6)

    def test_driver_sends_unconstrained_twist_to_virtual_base_joints(self):
        robot = _FakeVirtualRobot()
        world = SimpleNamespace(current_time=1.0)
        driver = LocalBaseDriver(robot, world=world)
        driver.prepare_for_navigation()
        driver.set_command(2.0, -2.0, 2.0)

        driver.step(step_dt=0.1)
        np.testing.assert_allclose(robot.commands[-1]["steering_positions"], [], atol=1e-6)
        np.testing.assert_allclose(robot.commands[-1]["wheel_velocities"], [2.0, -2.0, 2.0], atol=1e-6)
        self.assertEqual(robot.commands[-1]["step_dt"], 0.1)
        self.assertAlmostEqual(driver.get_logging_action_snapshot()["vy_body"], -2.0)
        action = driver.get_logging_action_snapshot()
        self.assertEqual(action["execution_mode"], "virtual_base_joint_velocity_target")
        driver.finalize_after_navigation()
        np.testing.assert_allclose(robot.commands[-1]["wheel_velocities"], [0.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(robot.hold_suspended, 1)
        self.assertEqual(robot.hold_resumed, 1)

    def test_driver_does_not_apply_wheel_acceleration_shaping(self):
        robot = _FakeVirtualRobot()
        world = SimpleNamespace(current_time=1.0)
        driver = LocalBaseDriver(robot, world=world)
        driver.prepare_for_navigation()
        driver.set_command(0.5, 0.0, 0.0)

        driver.step(step_dt=0.1)
        np.testing.assert_allclose(robot.commands[-1]["wheel_velocities"], [0.5, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
