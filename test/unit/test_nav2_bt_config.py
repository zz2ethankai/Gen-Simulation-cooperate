from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import yaml

from nav2.bridge.clock import _CLOCK_RESET_GAP_SEC
from nav2.mapgen.prepare_stack import _load_robot_base_cfg
from nav2.runtime.config import (
    _build_nav2_params,
    _controller_params,
    _validate_matching_costmap_constraints,
    _write_nav2_bt_files,
    configure_base_cfg_for_nav2_skill,
)


class Nav2BehaviorTreeConfigTests(unittest.TestCase):
    def _write_trees(self):
        base_cfg = {
            "ros": {
                "nav2": {
                    "clear_global_costmap_service": "/test/clear_global",
                    "clear_local_costmap_service": "/test/clear_local",
                }
            },
            "nav2_skill": {
                "bt_navigator": {
                    "path_validity_check_hz": 2.5,
                    "planner_retry_attempt_limit": 2,
                    "follow_path_retry_attempt_limit": 3,
                    "navigation_retry_attempt_limit": 4,
                    "costmap_clear_wait_sec": 2,
                    "remove_passed_goals_radius": 0.6,
                }
            },
        }
        temp_dir = tempfile.TemporaryDirectory()
        nav_to_pose_path, nav_through_poses_path = _write_nav2_bt_files(temp_dir.name, base_cfg)
        self.addCleanup(temp_dir.cleanup)
        return ET.parse(nav_to_pose_path), ET.parse(nav_through_poses_path)

    def test_nav_to_pose_replans_only_for_invalid_path_or_controller_failure(self):
        tree, _ = self._write_trees()
        root = tree.getroot()

        navigate_recovery = root.find(".//RecoveryNode[@name='NavigateWithOnDemandReplanning']")
        self.assertIsNotNone(navigate_recovery)
        self.assertEqual(navigate_recovery.attrib["number_of_retries"], "4")
        self.assertEqual([child.tag for child in navigate_recovery], ["PipelineSequence", "RecoveryNode"])

        rate_controller = root.find(".//RateController")
        self.assertIsNotNone(rate_controller)
        self.assertEqual(rate_controller.attrib["hz"], "2.5")

        fallback = root.find(".//Fallback[@name='KeepValidPathOrReplan']")
        self.assertIsNotNone(fallback)
        self.assertEqual([child.tag for child in fallback], ["ReactiveSequence", "Sequence"])
        self.assertIsNotNone(fallback.find("./ReactiveSequence/Inverter/GlobalUpdatedGoal"))
        self.assertIsNotNone(fallback.find("./ReactiveSequence/IsPathValid"))
        self.assertIsNotNone(fallback.find("./Sequence/ComputePathToPose"))
        self.assertIsNotNone(fallback.find("./Sequence/SmoothPath"))

        local_retry = root.find(".//RecoveryNode[@name='RetryFollowPathLocally']")
        self.assertIsNotNone(local_retry)
        self.assertEqual(local_retry.attrib["number_of_retries"], "3")
        self.assertEqual([child.tag for child in local_retry], ["FollowPath", "Sequence"])
        self.assertEqual(
            local_retry.find("./Sequence/ClearEntireCostmap").attrib["service_name"],
            "/test/clear_local",
        )

        forced_replan = root.find(".//RecoveryNode[@name='ReplanAfterControllerFailure']")
        self.assertIsNotNone(forced_replan)
        self.assertEqual(forced_replan.attrib["number_of_retries"], "2")
        self.assertIsNotNone(forced_replan.find("./Sequence/ComputePathToPose"))
        self.assertIsNotNone(forced_replan.find("./Sequence/SmoothPath"))

        waits = root.findall(".//Wait")
        self.assertGreaterEqual(len(waits), 3)
        self.assertTrue(all(wait.attrib["wait_duration"] == "2" for wait in waits))

    def test_nav_through_poses_uses_the_same_on_demand_recovery_structure(self):
        _, tree = self._write_trees()
        root = tree.getroot()

        self.assertIsNotNone(
            root.find(".//RecoveryNode[@name='NavigateThroughPosesWithOnDemandReplanning']")
        )
        fallback = root.find(".//Fallback[@name='KeepValidPathOrReplanThroughPoses']")
        self.assertIsNotNone(fallback)
        self.assertIsNotNone(fallback.find("./ReactiveSequence/IsPathValid"))
        self.assertIsNotNone(fallback.find("./Sequence/RemovePassedGoals"))
        self.assertIsNotNone(fallback.find("./Sequence/ComputePathThroughPoses"))

        local_retry = root.find(".//RecoveryNode[@name='RetryFollowPathThroughPosesLocally']")
        self.assertIsNotNone(local_retry)
        self.assertEqual(local_retry.attrib["number_of_retries"], "3")

        forced_replan = root.find(
            ".//RecoveryNode[@name='ReplanThroughPosesAfterControllerFailure']"
        )
        self.assertIsNotNone(forced_replan)
        self.assertIsNotNone(forced_replan.find("./Sequence/ComputePathThroughPoses"))

    def test_external_configs_enable_required_plugins_and_stopped_goal_checking(self):
        repo_root = Path(__file__).resolve().parents[2]
        runtime_config_path = repo_root / "nav2/config/default_nav.yaml"
        params_config_path = repo_root / "nav2/config/nav2_params.yaml"
        runtime_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
        params_config = yaml.safe_load(params_config_path.read_text(encoding="utf-8"))
        bt_cfg = runtime_config["nav2_skill"]["behavior_tree"]
        native_params = params_config["params"]

        self.assertEqual(bt_cfg["path_validity_check_hz"], 1.0)
        self.assertEqual(bt_cfg["planner_retry_attempt_limit"], 1)
        self.assertEqual(bt_cfg["follow_path_retry_attempt_limit"], 1)
        self.assertEqual(bt_cfg["navigation_retry_attempt_limit"], 3)
        self.assertEqual(bt_cfg["costmap_clear_wait_sec"], 1)
        self.assertNotIn("replanning_hz", bt_cfg)
        self.assertNotIn("replan_retry_attempt_limit", bt_cfg)

        plugins = set(native_params["bt_navigator"]["ros__parameters"]["plugin_lib_names"])
        self.assertIn("nav2_globally_updated_goal_condition_bt_node", plugins)
        self.assertIn("nav2_is_path_valid_condition_bt_node", plugins)
        self.assertIn("nav2_wait_action_bt_node", plugins)
        goal_checker = native_params["controller_server"]["ros__parameters"]["general_goal_checker"]
        self.assertTrue(goal_checker["stateful"])
        self.assertEqual(goal_checker["plugin"], "nav2_controller::StoppedGoalChecker")
        self.assertEqual(goal_checker["trans_stopped_velocity"], 0.005)
        self.assertEqual(goal_checker["rot_stopped_velocity"], 0.005)
        progress_checker = native_params["controller_server"]["ros__parameters"]["progress_checker"]
        self.assertEqual(progress_checker["plugin"], "nav2_controller::PoseProgressChecker")
        self.assertEqual(progress_checker["required_movement_radius"], 0.05)
        self.assertEqual(progress_checker["required_movement_angle"], 0.1)
        self.assertEqual(progress_checker["movement_time_allowance"], 30.0)
        self.assertEqual(
            params_config["defaults"]["controller_plugin"],
            "nav2_rotation_shim_controller::RotationShimController",
        )
        rotation_shim_profile = params_config["controller_profiles"][
            "nav2_rotation_shim_controller::RotationShimController"
        ]
        self.assertEqual(
            rotation_shim_profile["primary_controller"],
            "nav2_mppi_controller::MPPIController",
        )
        mppi_profile = params_config["controller_profiles"][
            "nav2_mppi_controller::MPPIController"
        ]
        self.assertEqual(mppi_profile["model_dt"], 1.0 / 15.0)
        self.assertEqual(mppi_profile["time_steps"], 30)
        self.assertEqual(mppi_profile["model_dt"] * mppi_profile["time_steps"], 2.0)
        self.assertFalse(mppi_profile["regenerate_noises"])
        self.assertGreater(_CLOCK_RESET_GAP_SEC, mppi_profile["reset_period"])
        self.assertEqual(
            native_params["controller_server"]["ros__parameters"]["controller_frequency"],
            15.0,
        )
        self.assertEqual(
            native_params["velocity_smoother"]["ros__parameters"]["smoothing_frequency"],
            15.0,
        )

    def test_panda_omron_virtual_reduces_mppi_twirling_penalty(self):
        repo_root = Path(__file__).resolve().parents[2]
        params_config = yaml.safe_load(
            (repo_root / "nav2/config/nav2_params.yaml").read_text(encoding="utf-8")
        )
        virtual_config = yaml.safe_load(
            (repo_root / "nav2/config/panda_omron_virtual_nav.yaml").read_text(
                encoding="utf-8"
            )
        )

        _, nested_overrides = _controller_params(
            params_config,
            virtual_config["nav2_skill"],
            {
                "max_velocity": [1.0, 1.0, 1.0],
                "min_velocity": [-1.0, -1.0, -1.0],
                "max_accel": [1.0, 1.0, 1.0],
                "max_decel": [-1.0, -1.0, -1.0],
            },
        )
        follow_path = nested_overrides["FollowPath"]
        self.assertEqual(
            follow_path["plugin"],
            "nav2_rotation_shim_controller::RotationShimController",
        )
        self.assertEqual(
            follow_path["primary_controller"],
            "nav2_mppi_controller::MPPIController",
        )
        self.assertEqual(follow_path["wz_max"], 1.0)
        self.assertEqual(follow_path["rotate_to_heading_angular_vel"], 0.75)
        self.assertEqual(follow_path["az_max"], 1.0)
        self.assertEqual(follow_path["max_angular_accel"], 1.0)
        twirling_weight = follow_path["TwirlingCritic"]["cost_weight"]
        goal_angle_weight = follow_path["GoalAngleCritic"]["cost_weight"]
        self.assertTrue(follow_path["PathAlignCritic"]["use_path_orientations"])
        shared_costmap_padding = virtual_config["nav2_skill"]["costmap"]["footprint_padding"]

        self.assertEqual(twirling_weight, 0.5)
        self.assertLess(twirling_weight, goal_angle_weight)
        self.assertEqual(shared_costmap_padding, 0.04)

    def test_rotation_shim_and_mppi_share_conservative_angular_limits(self):
        repo_root = Path(__file__).resolve().parents[2]
        params_config = yaml.safe_load(
            (repo_root / "nav2/config/nav2_params.yaml").read_text(encoding="utf-8")
        )
        _, nested_overrides = _controller_params(
            params_config,
            {},
            {
                "max_velocity": [0.45, 0.32, 0.90],
                "min_velocity": [-0.40, -0.30, -0.65],
                "max_accel": [0.50, 0.60, 1.10],
                "max_decel": [-0.40, -0.50, -0.80],
            },
        )
        follow_path = nested_overrides["FollowPath"]

        self.assertEqual(follow_path["vx_max"], 0.45)
        self.assertEqual(follow_path["vx_min"], -0.40)
        self.assertEqual(follow_path["vy_max"], 0.32)
        self.assertEqual(follow_path["vy_min"], -0.30)
        self.assertEqual(follow_path["wz_max"], 0.65)
        self.assertEqual(follow_path["rotate_to_heading_angular_vel"], 0.65)
        self.assertEqual(follow_path["az_max"], 0.80)
        self.assertEqual(follow_path["max_angular_accel"], 0.80)

    def test_panda_omron_virtual_generates_shared_controller_and_costmap_limits(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_cfg = _load_robot_base_cfg(
            repo_root / "workflows/simbox/core/configs/robots/panda_omron_virtual.yaml"
        )
        base_cfg = configure_base_cfg_for_nav2_skill(base_cfg)
        params = _build_nav2_params(
            nav_to_pose_bt="navigate_to_pose.xml",
            nav_through_poses_bt="navigate_through_poses.xml",
            base_cfg=base_cfg,
            position_tolerance_m=0.1,
            yaw_tolerance_rad=0.1,
        )

        follow_path = params["controller_server"]["ros__parameters"]["FollowPath"]
        self.assertEqual(follow_path["wz_max"], 0.5)
        self.assertEqual(follow_path["rotate_to_heading_angular_vel"], 0.5)
        self.assertEqual(follow_path["az_max"], 1.0)
        self.assertEqual(follow_path["max_angular_accel"], 1.0)
        self.assertFalse(follow_path["position_checker"]["stateful"])
        self.assertTrue(follow_path["PathAlignCritic"]["use_path_orientations"])

        local = params["local_costmap"]["local_costmap"]["ros__parameters"]
        global_ = params["global_costmap"]["global_costmap"]["ros__parameters"]
        self.assertEqual(local["footprint"], global_["footprint"])
        self.assertEqual(local["footprint_padding"], 0.04)
        self.assertEqual(local["footprint_padding"], global_["footprint_padding"])
        self.assertEqual(
            local["inflation_layer"]["inflation_radius"],
            global_["inflation_layer"]["inflation_radius"],
        )
        self.assertEqual(
            local["inflation_layer"]["cost_scaling_factor"],
            global_["inflation_layer"]["cost_scaling_factor"],
        )
        self.assertEqual(
            params["velocity_smoother"]["ros__parameters"]["max_velocity"][2],
            0.5,
        )
        self.assertEqual(
            params["behavior_server"]["ros__parameters"]["max_rotational_vel"],
            0.5,
        )

    def test_global_and_local_costmap_constraint_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "footprint_padding"):
            _validate_matching_costmap_constraints(
                {
                    "footprint_padding": 0.04,
                    "inflation_layer": {"cost_scaling_factor": 3.0, "inflation_radius": 0.4},
                },
                {
                    "footprint_padding": 0.02,
                    "inflation_layer": {"cost_scaling_factor": 3.0, "inflation_radius": 0.4},
                },
            )

    def test_controller_plugin_fallback_is_rejected(self):
        repo_root = Path(__file__).resolve().parents[2]
        params_config = yaml.safe_load(
            (repo_root / "nav2/config/nav2_params.yaml").read_text(encoding="utf-8")
        )
        hard_limits = {
            "max_velocity": [1.0, 1.0, 1.0],
            "min_velocity": [-1.0, -1.0, -1.0],
            "max_accel": [1.0, 1.0, 1.0],
            "max_decel": [-1.0, -1.0, -1.0],
        }

        with self.assertRaisesRegex(ValueError, "controller plugin fallback is not supported"):
            _controller_params(
                params_config,
                {
                    "controller_server": {
                        "follow_path": {
                            "plugin": "nav2_mppi_controller::MPPIController",
                        }
                    }
                },
                hard_limits,
            )


if __name__ == "__main__":
    unittest.main()
