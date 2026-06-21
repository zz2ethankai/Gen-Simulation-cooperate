from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

from nav2.runtime.dynamic_goal import (
    ApproachConfig,
    check_footprint_static_collision,
    choose_best_reachable_candidate,
    parse_approach_config,
    resolve_approach_footprint_padding_m,
    sample_approach_candidates,
    sort_candidates_for_preflight,
)


class DynamicApproachTests(unittest.TestCase):
    def test_parse_approach_config_is_opt_in(self):
        self.assertIsNone(parse_approach_config({"goal": "pick"}))

        cfg = parse_approach_config(
            {
                "approach": "apple_0_id9008",
                "approach_min_distance": 0.8,
                "approach_max_distance": 1.2,
                "approach_sample_count": 8,
            }
        )
        self.assertEqual(cfg, ApproachConfig("apple_0_id9008", 0.8, 1.2, 8))

    def test_sample_approach_candidates_distance_and_yaw_face_target(self):
        cfg = ApproachConfig("obj", min_distance=0.8, max_distance=1.2, sample_count=4)
        candidates = sample_approach_candidates(cfg, (2.0, 3.0))

        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            dist = math.hypot(candidate["x"] - 2.0, candidate["y"] - 3.0)
            self.assertGreaterEqual(dist, 0.8 - 1e-9)
            self.assertLessEqual(dist, 1.2 + 1e-9)
            expected_yaw = math.atan2(3.0 - candidate["y"], 2.0 - candidate["x"])
            yaw_error = math.atan2(math.sin(candidate["yaw"] - expected_yaw), math.cos(candidate["yaw"] - expected_yaw))
            self.assertAlmostEqual(yaw_error, 0.0)

    def test_sample_approach_candidates_uniformly_spans_min_to_max_distance(self):
        cfg = ApproachConfig("obj", min_distance=0.35, max_distance=0.50, sample_count=2048)
        candidates = sample_approach_candidates(cfg, (2.0, 3.0))
        distances = [round(math.hypot(candidate["x"] - 2.0, candidate["y"] - 3.0), 6) for candidate in candidates]

        self.assertEqual(len(candidates), 2048)
        self.assertAlmostEqual(distances[0], 0.35)
        self.assertAlmostEqual(distances[-1], 0.50)
        self.assertEqual(len(set(distances)), 2048)
        expected_step = (0.50 - 0.35) / 2047.0
        self.assertAlmostEqual(distances[1] - distances[0], expected_step, places=6)

    def test_sample_approach_candidates_random_seed_is_reproducible(self):
        cfg = ApproachConfig(
            "obj",
            min_distance=0.35,
            max_distance=0.50,
            sample_count=64,
            sampling_random=True,
            sampling_seed=7,
        )
        same_cfg = ApproachConfig(
            "obj",
            min_distance=0.35,
            max_distance=0.50,
            sample_count=64,
            sampling_random=True,
            sampling_seed=7,
        )
        different_cfg = ApproachConfig(
            "obj",
            min_distance=0.35,
            max_distance=0.50,
            sample_count=64,
            sampling_random=True,
            sampling_seed=8,
        )

        candidates = sample_approach_candidates(cfg, (2.0, 3.0))
        same_candidates = sample_approach_candidates(same_cfg, (2.0, 3.0))
        different_candidates = sample_approach_candidates(different_cfg, (2.0, 3.0))

        self.assertEqual(candidates, same_candidates)
        self.assertNotEqual(candidates, different_candidates)
        for candidate in candidates:
            self.assertGreaterEqual(candidate["distance_to_target"], 0.35)
            self.assertLessEqual(candidate["distance_to_target"], 0.50)
        distances = [candidate["distance_to_target"] for candidate in candidates]
        self.assertFalse(all(left <= right for left, right in zip(distances, distances[1:])))

    def test_parse_approach_config_generates_seed_for_random_sampling(self):
        cfg = parse_approach_config({"approach": "apple", "approach_sampling_random": True})

        self.assertTrue(cfg.sampling_random)
        self.assertIsInstance(cfg.sampling_seed, int)

    def test_static_footprint_collision_free_occupied_unknown_and_bounds(self):
        static_map = {
            "image": np.full((20, 20), 254, dtype=np.int16),
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
        }
        footprint = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]

        free = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            y=1.0,
            yaw=0.0,
        )
        self.assertTrue(free["ok"])

        static_map["image"][9:12, 9:12] = 0
        occupied = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            y=1.0,
            yaw=0.0,
        )
        self.assertFalse(occupied["ok"])
        self.assertEqual(occupied["reason"], "static_footprint_collision")
        self.assertGreater(occupied["blocked_cells"], 0)

        static_map["image"][:, :] = 254
        static_map["image"][9:12, 9:12] = -1
        unknown = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            y=1.0,
            yaw=0.0,
        )
        self.assertFalse(unknown["ok"])
        self.assertGreater(unknown["unknown_cells"], 0)

        out = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=-1.0,
            y=-1.0,
            yaw=0.0,
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "footprint_out_of_bounds")

    def test_static_footprint_padding_requires_clearance(self):
        static_map = {
            "image": np.full((20, 20), 254, dtype=np.int16),
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
        }
        footprint = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
        static_map["image"][8, 9] = 0

        unpadded = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            y=1.0,
            yaw=0.0,
        )
        self.assertTrue(unpadded["ok"])

        padded = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint,
            x=1.0,
            y=1.0,
            yaw=0.0,
            footprint_padding_m=0.1,
        )
        self.assertFalse(padded["ok"])
        self.assertEqual(padded["footprint_padding_cells"], 1)
        self.assertEqual(padded["footprint_blocked_cells"], 0)
        self.assertGreater(padded["padding_blocked_cells"], 0)

    def test_approach_footprint_padding_defaults_to_nav2_skill_padding(self):
        base_cfg = {
            "nav2_skill": {
                "approach_footprint_padding": 0.04,
                "local_costmap": {"footprint_padding": 0.1},
            }
        }
        default_cfg = ApproachConfig("obj")
        explicit_cfg = ApproachConfig("obj", footprint_padding_m=0.03)

        self.assertAlmostEqual(resolve_approach_footprint_padding_m(base_cfg, default_cfg), 0.04)
        self.assertAlmostEqual(resolve_approach_footprint_padding_m(base_cfg, explicit_cfg), 0.03)

    def test_approach_footprint_padding_falls_back_to_local_costmap_padding(self):
        base_cfg = {"nav2_skill": {"local_costmap": {"footprint_padding": 0.1}}}

        self.assertAlmostEqual(resolve_approach_footprint_padding_m(base_cfg, ApproachConfig("obj")), 0.1)

    def test_reachable_candidate_selection_uses_nearest_distance_after_path_success(self):
        candidates = [
            {"index": 0, "static_ok": True, "path_ok": True, "distance_to_target": 0.9, "path_length_m": 1.0},
            {"index": 1, "static_ok": True, "path_ok": False, "distance_to_target": 0.5, "path_length_m": 0.5},
            {"index": 2, "static_ok": True, "path_ok": True, "distance_to_target": 0.7, "path_length_m": 4.0},
        ]
        self.assertEqual(choose_best_reachable_candidate(candidates)["index"], 2)

    def test_preflight_candidate_order_uses_nearest_distance(self):
        candidates = [
            {"index": 0, "distance_to_target": 0.9},
            {"index": 1, "distance_to_target": 0.5},
            {"index": 2, "distance_to_target": 0.7},
        ]

        ordered = sort_candidates_for_preflight(candidates)

        self.assertEqual([candidate["index"] for candidate in ordered], [1, 2, 0])
        self.assertEqual([candidate["preflight_rank"] for candidate in ordered], [0, 1, 2])

    def test_dynamic_plan_stops_on_first_successful_preflight_candidate(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager._dynamic_goal_active_plan_request_id = "req_approach_1"
        manager._dynamic_goal_plan_index = 0
        manager._dynamic_goal_plan_deadline = 123.0
        manager._dynamic_goal_candidates = [
            {
                "index": 1,
                "x": 1.0,
                "y": 2.0,
                "yaw": 0.3,
                "static_ok": True,
                "distance_to_target": 0.5,
                "path_ok": False,
            },
            {
                "index": 2,
                "x": 1.5,
                "y": 2.5,
                "yaw": 0.4,
                "static_ok": True,
                "distance_to_target": 0.7,
                "path_ok": False,
                "path_state": "not_requested",
            },
        ]
        manager._request_id = "req"
        manager._write_dynamic_goal_candidates_debug = lambda: None

        def publish_navigation_goal(bridge_client):
            del bridge_client
            manager._published_goal = (manager.goal_x, manager.goal_y, manager.goal_yaw)

        manager._publish_navigation_goal = publish_navigation_goal

        bridge = _FakePlanBridge(
            {
                "state": "succeeded",
                "detail": "",
                "status_code": 0,
                "planning": {"path": {"path_length_m": 4.0, "num_poses": 12}},
            }
        )

        PersistentNav2RuntimeManager._consume_dynamic_plan_result(manager, bridge)

        self.assertEqual(manager._dynamic_goal_selected["index"], 1)
        self.assertEqual(manager._published_goal, (1.0, 2.0, 0.3))
        self.assertEqual(manager._dynamic_goal_plan_index, 0)
        self.assertEqual(manager._dynamic_goal_active_plan_request_id, "")
        self.assertIsNone(manager._dynamic_goal_plan_deadline)
        self.assertEqual(bridge.plan_result_requests, [("req", "req_approach_1")])
        self.assertTrue(bridge.cleared)
        self.assertEqual(manager._dynamic_goal_candidates[1]["path_state"], "not_requested")

    def test_navigate_fixed_goal_unchanged_and_approach_skips_positions(self):
        with mock.patch.dict(sys.modules, _navigate_import_stubs()):
            sys.modules.pop("workflows.simbox.core.skills.navigate", None)
            navigate_path = Path(__file__).resolve().parents[2] / "workflows/simbox/core/skills/navigate.py"
            spec = importlib.util.spec_from_file_location("_test_navigate_skill_module", navigate_path)
            navigate_mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(navigate_mod)
            navigate_mod.configure_robot_for_nav2_skill = lambda robot, **kwargs: {}
            Navigate = navigate_mod.Navigate

            fixed_task = types.SimpleNamespace(
                cfg={"positions": {"pick": {"x": 0.5, "y": -0.25, "yaw": 0.1}}},
                fixtures={"floor": _PoseObject((2.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0))},
            )
            fixed_goal = Navigate._resolve_goal_pose(Navigate, fixed_task, {"goal": "pick"})
            self.assertAlmostEqual(fixed_goal[0], 2.5)
            self.assertAlmostEqual(fixed_goal[1], 1.25)
            self.assertAlmostEqual(fixed_goal[2], 0.1)

            class BadPositions(dict):
                def get(self, key, default=None):
                    raise AssertionError(f"positions should not be read for approach goal {key}")

            approach_task = types.SimpleNamespace(
                cfg={"positions": BadPositions({"legacy": {"x": 1.0, "y": 1.0, "yaw": 0.0}})},
                fixtures={"floor": _PoseObject((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))},
            )
            robot = types.SimpleNamespace(base_cfg={})
            skill = Navigate(
                robot,
                object(),
                approach_task,
                {"name": "navigate", "goal": "legacy", "approach": "apple_0_id9008"},
                world=object(),
            )
            self.assertEqual(skill.approach_config.target_name, "apple_0_id9008")
            self.assertEqual((skill.goal_x, skill.goal_y, skill.goal_yaw), (0.0, 0.0, 0.0))

    def test_approach_target_resolution_uses_task_objects_table(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager.approach_config = ApproachConfig("sink")
        manager.task = types.SimpleNamespace(
            _task_objects={"sink": _PoseObject((1.2, 3.4, 5.6), (0.7, 0.0, 0.0, 0.7))},
            objects={},
            fixtures={},
        )

        pose = PersistentNav2RuntimeManager._resolve_approach_target_pose(manager)

        self.assertEqual(pose["name"], "sink")
        self.assertAlmostEqual(pose["x"], 1.2)
        self.assertAlmostEqual(pose["y"], 3.4)
        self.assertAlmostEqual(pose["z"], 5.6)


class _PoseObject:
    def __init__(self, translation, orientation):
        self.translation = translation
        self.orientation = orientation

    def get_world_pose(self):
        return self.translation, self.orientation


class _FakePlanBridge:
    def __init__(self, payload):
        self.payload = payload
        self.plan_result_requests = []
        self.cleared = False

    def request_plan_result(self, *, request_id, plan_request_id):
        self.plan_result_requests.append((request_id, plan_request_id))
        return self.payload

    def clear_cached_bridge_state(self):
        self.cleared = True


def _navigate_import_stubs():
    core_mod = types.ModuleType("core")
    skills_mod = types.ModuleType("core.skills")
    base_skill_mod = types.ModuleType("core.skills.base_skill")

    class BaseSkill:
        pass

    base_skill_mod.BaseSkill = BaseSkill
    base_skill_mod.SKILL_DICT = {}
    base_skill_mod.register_skill = lambda cls: cls

    omegaconf_mod = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class OmegaConf:
        @staticmethod
        def to_container(value, resolve=True):
            del resolve
            return dict(value)

    omegaconf_mod.DictConfig = DictConfig
    omegaconf_mod.OmegaConf = OmegaConf

    omni_mod = types.ModuleType("omni")
    isaac_mod = types.ModuleType("omni.isaac")
    isaac_core_mod = types.ModuleType("omni.isaac.core")
    controllers_mod = types.ModuleType("omni.isaac.core.controllers")
    robots_mod = types.ModuleType("omni.isaac.core.robots")
    robot_mod = types.ModuleType("omni.isaac.core.robots.robot")
    tasks_mod = types.ModuleType("omni.isaac.core.tasks")
    controllers_mod.BaseController = object
    robot_mod.Robot = object
    tasks_mod.BaseTask = object

    return {
        "core": core_mod,
        "core.skills": skills_mod,
        "core.skills.base_skill": base_skill_mod,
        "omegaconf": omegaconf_mod,
        "omni": omni_mod,
        "omni.isaac": isaac_mod,
        "omni.isaac.core": isaac_core_mod,
        "omni.isaac.core.controllers": controllers_mod,
        "omni.isaac.core.robots": robots_mod,
        "omni.isaac.core.robots.robot": robot_mod,
        "omni.isaac.core.tasks": tasks_mod,
    }


if __name__ == "__main__":
    unittest.main()
