from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import yaml

from nav2.runtime.config import _write_nav2_bt_files


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

    def test_external_configs_enable_required_plugins_and_stateful_goal_checking(self):
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


if __name__ == "__main__":
    unittest.main()
