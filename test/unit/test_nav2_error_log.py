from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from nav2.bridge.error_log import Nav2ErrorLogBuffer, is_nav2_logger


class Nav2ErrorLogTests(unittest.TestCase):
    def test_filters_non_nav2_and_info_logs(self):
        buffer = Nav2ErrorLogBuffer()
        buffer.reset(request_id="goal-1", started_wall_time_sec=1.0)
        buffer.append(level=20, name="controller_server", message="info", wall_time_sec=2.0)
        buffer.append(level=40, name="camera", message="bad frame", wall_time_sec=3.0)
        self.assertEqual(buffer.snapshot()["entries"], [])

    def test_keeps_failure_bearing_info_log(self):
        buffer = Nav2ErrorLogBuffer()
        buffer.reset(request_id="goal-1", started_wall_time_sec=1.0)
        buffer.append(
            level=20,
            name="planner_server",
            message="Starting point in lethal space! Cannot create feasible plan",
            wall_time_sec=2.0,
        )
        self.assertEqual(buffer.snapshot()["entries"][0]["level"], "INFO")

    def test_aggregates_repeated_nav2_errors_without_losing_count(self):
        buffer = Nav2ErrorLogBuffer()
        buffer.reset(request_id="goal-1", started_wall_time_sec=1.0)
        for wall_time in (2.0, 3.0, 4.0):
            buffer.append(
                level=40,
                name="controller_server",
                message="Optimizer fail to compute path",
                function="computeVelocityCommands",
                file="controller_server.cpp",
                line=123,
                ros_time_sec=wall_time - 1.0,
                wall_time_sec=wall_time,
            )
        snapshot = buffer.snapshot()
        self.assertEqual(snapshot["message_count"], 3)
        self.assertEqual(snapshot["unique_entry_count"], 1)
        self.assertEqual(snapshot["entries"][0]["count"], 3)
        self.assertEqual(snapshot["entries"][0]["last_wall_time_sec"], 4.0)

    def test_recognizes_namespaced_nav2_logger(self):
        self.assertTrue(is_nav2_logger("/robot/controller_server"))
        self.assertTrue(is_nav2_logger("/robot/controller_server.rclcpp_action"))
        self.assertTrue(is_nav2_logger("nav2_custom_controller"))
        self.assertFalse(is_nav2_logger("simbox_camera"))

    def test_compacts_dynamic_candidate_without_path_poses(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        candidate = {
            "index": 2,
            "planning": {
                "state": "succeeded",
                "source": "compute_path_to_pose",
                "path": {
                    "num_poses": 2,
                    "path_length_m": 1.25,
                    "poses": [{"x": 0.0}, {"x": 1.0}],
                },
            },
        }
        compact = PersistentNav2RuntimeManager._compact_dynamic_goal_candidate(candidate)
        self.assertEqual(compact["planning"]["path_num_poses"], 2)
        self.assertEqual(compact["planning"]["path_length_m"], 1.25)
        self.assertNotIn("path", compact["planning"])
        self.assertIn("poses", candidate["planning"]["path"])

    def test_failure_report_falls_back_to_status_errors_and_has_no_motion_trace(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager.robot = SimpleNamespace(name="panda_omron")
        manager.result = SimpleNamespace(success=False)
        manager._request_id = "goal-1"
        manager._stack_id = "stack-1"
        manager.goal_x = 1.0
        manager.goal_y = 2.0
        manager.goal_yaw = 0.5
        status_errors = {"unique_entry_count": 1, "entries": [{"message": "No valid trajectories"}]}
        with tempfile.TemporaryDirectory() as output_dir:
            manager._goal_output_dir = output_dir
            artifacts = manager._write_nav2_error_report(
                reason="map_update_timeout",
                message="map failed",
                nav2_result={},
                nav2_status={"nav2_errors": status_errors},
                planning_payload={
                    "state": "succeeded",
                    "path": {
                        "num_poses": 2,
                        "path_length_m": 1.25,
                        "poses": [{"x": 0.0}, {"x": 1.0}],
                    },
                },
            )
            report_path = artifacts["nav2_error_report"]
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(report["nav2_errors"], status_errors)
        self.assertEqual(report["planning_summary"]["path_num_poses"], 2)
        serialized = json.dumps(report)
        self.assertNotIn('\"poses\"', serialized)
        self.assertNotIn("cmd_vel_history", serialized)
        self.assertNotIn("bridge_command_history", serialized)

    def test_success_does_not_write_error_report(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager.result = SimpleNamespace(success=True)
        with tempfile.TemporaryDirectory() as output_dir:
            manager._goal_output_dir = output_dir
            artifacts = manager._write_nav2_error_report(
                reason="goal_succeeded",
                message="ok",
                nav2_result={},
                nav2_status={},
                planning_payload={},
            )
            self.assertEqual(artifacts, {})
            self.assertFalse(os.path.exists(os.path.join(output_dir, "nav2_error_report.json")))

    def test_navigation_execution_trace_is_throttled_and_records_motion(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager.robot = SimpleNamespace(name="panda_omron")
        manager.state = manager.STATE_RUNNING
        manager.result = SimpleNamespace(
            done=False,
            success=False,
            failure_reason="",
            error_message="",
            final_world_xy=(1.0, 2.0),
            final_world_yaw=0.2,
            final_nav_xy=(1.01, 2.01),
            final_nav_yaw=0.21,
            final_distance_to_goal=0.08,
            final_nav_distance_to_goal=0.07,
            final_yaw_error_rad=0.04,
        )
        manager.goal_x = 1.05
        manager.goal_y = 2.05
        manager.goal_yaw = 0.25
        manager._request_id = "goal-1"
        manager._execution_trace = []
        manager._execution_trace_total_samples = 0
        manager._execution_trace_sample_interval_sec = 0.5
        manager._execution_trace_write_interval_sec = 0.0
        manager._execution_trace_max_samples = 10
        manager._execution_trace_started_at = 10.0
        manager._execution_trace_next_sample_at = 10.0
        manager._execution_trace_last_write_at = 10.0
        manager._controller_trace_config = {
            "follow_path": {
                "plugin": "nav2_rotation_shim_controller::RotationShimController",
                "primary_controller": "nav2_mppi_controller::MPPIController",
            }
        }
        manager._update_pose_result_fields = mock.Mock()
        manager._sim_time = mock.Mock(side_effect=[10.0, 10.2, 10.5])
        bridge = SimpleNamespace(
            latest_status={"state": "running", "detail": "goal running"},
            latest_result={},
        )
        control_snapshot = {
            "bridge": {
                "last_received_cmd_vel": {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.2},
                "last_published_pose": {
                    "actual_linear_velocity_body": [0.0, 0.0, 0.0],
                    "actual_angular_velocity_world": [0.0, 0.0, 0.18],
                },
                "received_cmd_vel_count": 20,
                "applied_driver_command_count": 19,
            }
        }

        with tempfile.TemporaryDirectory() as output_dir, mock.patch(
            "nav2.runtime.runtime.runtime_control_debug_snapshot",
            return_value=control_snapshot,
        ):
            manager._goal_output_dir = output_dir
            manager._maybe_record_execution_trace(bridge)
            manager._maybe_record_execution_trace(bridge)
            manager._maybe_record_execution_trace(bridge)
            trace_path = manager._write_execution_trace(now=10.5)
            with open(trace_path, "r", encoding="utf-8") as handle:
                trace = json.load(handle)

        self.assertEqual(trace["total_samples"], 2)
        self.assertEqual(trace["samples"][0]["motion_phase"], "rotation_only_command")
        self.assertEqual(trace["samples"][0]["command"]["angular_z"], 0.2)
        self.assertEqual(trace["samples"][0]["actual_velocity"]["angular_world"][2], 0.18)
        self.assertEqual(trace["controller_config"]["follow_path"]["primary_controller"],
                         "nav2_mppi_controller::MPPIController")

    def test_navigation_execution_trace_collection_failure_is_non_fatal(self):
        from nav2.runtime.runtime import PersistentNav2RuntimeManager

        manager = PersistentNav2RuntimeManager.__new__(PersistentNav2RuntimeManager)
        manager._execution_trace = []
        manager._goal_output_dir = "/tmp/not-written"
        manager._execution_trace_sample_interval_sec = 0.5
        manager._execution_trace_next_sample_at = 0.0
        manager._sim_time = mock.Mock(return_value=1.0)
        manager._update_pose_result_fields = mock.Mock(side_effect=RuntimeError("pose unavailable"))

        manager._maybe_record_execution_trace(SimpleNamespace())

        self.assertEqual(manager._execution_trace, [])


if __name__ == "__main__":
    unittest.main()
