from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nav2.runtime.runtime import PersistentNav2RuntimeManager
from nav2.mapgen.prepare_stack import _load_robot_base_cfg
from nav2.runtime.config import _build_nav2_params, configure_base_cfg_for_nav2_skill
from workflows.simbox.core.mobile.bridge.base_bridge import BaseBridge
from workflows.simbox.core.mobile.bridge.types import BaseCommand


class _TestBridge(BaseBridge):
    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        del steering_count, wheel_count

    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        del command
        return np.zeros(0, dtype=np.float32), np.zeros(3, dtype=np.float32)


class Nav2ControlOwnershipTests(unittest.TestCase):
    def test_generated_panda_omron_params_require_pose_and_stopped_velocity(self):
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

        goal_checker = params["controller_server"]["ros__parameters"]["general_goal_checker"]
        self.assertEqual(goal_checker["plugin"], "nav2_controller::StoppedGoalChecker")
        self.assertFalse(goal_checker["stateful"])
        self.assertEqual(goal_checker["trans_stopped_velocity"], 0.005)
        self.assertEqual(goal_checker["rot_stopped_velocity"], 0.005)

        progress_checker = params["controller_server"]["ros__parameters"]["progress_checker"]
        self.assertEqual(progress_checker["plugin"], "nav2_controller::PoseProgressChecker")
        self.assertEqual(progress_checker["required_movement_radius"], 0.05)
        self.assertEqual(progress_checker["required_movement_angle"], 0.1)
        self.assertEqual(progress_checker["movement_time_allowance"], 30.0)

    def test_bridge_does_not_write_control_without_nav2_command(self):
        bridge = object.__new__(_TestBridge)
        bridge._spin_available_callbacks = mock.Mock()
        bridge._now_sec = mock.Mock(return_value=2.0)
        bridge._last_step_time_sec = 1.0
        bridge._last_step_dt = 0.0
        bridge._navigation_active = True
        bridge._has_nav2_command = False
        bridge._publish_joint_state = mock.Mock()
        bridge._publish_odometry = mock.Mock()
        bridge._apply_robot_base_command = mock.Mock()
        bridge._rclpy = SimpleNamespace(spin_once=mock.Mock())
        bridge.node = object()

        bridge.step(step_dt=0.1)

        bridge._apply_robot_base_command.assert_not_called()
        bridge._publish_joint_state.assert_called_once_with()
        bridge._publish_odometry.assert_called_once_with()

    def test_finalize_writes_zero_base_command_before_closing_nav2_input_gate(self):
        bridge = object.__new__(_TestBridge)
        bridge._spin_available_callbacks = mock.Mock()
        bridge._now_sec = mock.Mock(return_value=2.0)
        bridge._command = BaseCommand(0.2, 0.0, 0.1, 1.9)
        bridge._last_step_command = bridge._command
        bridge._navigation_active = True
        bridge._has_nav2_command = True
        bridge._last_wheel_shaping_debug = {"source": "nav2"}
        bridge._last_requested_steering = np.array([], dtype=np.float32)
        bridge._last_requested_wheel_velocities = np.array([0.2, 0.0, 0.1], dtype=np.float32)
        bridge._last_applied_steering = np.array([], dtype=np.float32)
        bridge._last_applied_wheel_velocities = np.array([0.2, 0.0, 0.1], dtype=np.float32)
        bridge._publish_joint_state = mock.Mock()
        bridge._publish_odometry = mock.Mock()
        bridge._apply_robot_base_command = mock.Mock()
        bridge._rclpy = SimpleNamespace(spin_once=mock.Mock())
        bridge.node = object()

        bridge.finalize_after_navigation()

        self.assertFalse(bridge._navigation_active)
        self.assertFalse(bridge._has_nav2_command)
        bridge._apply_robot_base_command.assert_called_once()
        kwargs = bridge._apply_robot_base_command.call_args.kwargs
        np.testing.assert_array_equal(kwargs["steering_positions"], np.zeros(0, dtype=np.float32))
        np.testing.assert_array_equal(kwargs["wheel_velocities"], np.zeros(3, dtype=np.float32))
        self.assertEqual(kwargs["step_dt"], 1e-3)

    def test_post_success_settling_finalizes_nav2_control_immediately(self):
        bridge = SimpleNamespace(finalize_after_navigation=mock.Mock())
        manager = object.__new__(PersistentNav2RuntimeManager)
        manager.robot = SimpleNamespace(_simbox_ros_base_bridge=bridge)
        manager._base_cfg = {}
        manager._sim_time = mock.Mock(return_value=10.0)

        manager._enter_post_success_settling()

        self.assertEqual(manager.state, manager.STATE_POST_SUCCESS_SETTLING)
        self.assertEqual(manager._post_success_settle_started_at, 10.0)
        bridge.finalize_after_navigation.assert_called_once_with()

        manager._finalize_robot_bridge_after_navigation()
        bridge.finalize_after_navigation.assert_called_once_with()

    def test_pose_tolerance_does_not_preempt_running_nav2_goal(self):
        bridge_client = mock.Mock()
        bridge_client.request_status.return_value = {"state": "running"}
        bridge_client.request_result.return_value = {}

        manager = object.__new__(PersistentNav2RuntimeManager)
        manager.state = manager.STATE_RUNNING
        manager.result = SimpleNamespace(
            done=False,
            success=False,
            final_distance_to_goal=0.01,
            final_nav_distance_to_goal=0.01,
            final_yaw_error_rad=0.01,
        )
        manager.position_tolerance_m = 0.1
        manager.yaw_tolerance_rad = 0.1
        manager._runtime_deadline = 100.0
        manager._request_id = "goal"
        manager._robot_base_state_is_invalid = mock.Mock(return_value=False)
        manager._ensure_bridge_client = mock.Mock(return_value=bridge_client)
        manager._step_dt = mock.Mock(return_value=1.0 / 60.0)
        manager._update_pose_result_fields = mock.Mock()
        manager._sim_time = mock.Mock(return_value=10.0)

        manager.step()

        self.assertEqual(manager.state, manager.STATE_RUNNING)
        bridge_client.cancel_request.assert_not_called()

    def test_isaac_tolerance_check_runs_after_nav2_success(self):
        bridge_client = mock.Mock()
        bridge_client.request_status.return_value = {"state": "succeeded"}
        bridge_client.request_result.return_value = {"state": "succeeded"}

        manager = object.__new__(PersistentNav2RuntimeManager)
        manager.state = manager.STATE_RUNNING
        manager.result = SimpleNamespace(
            done=False,
            success=False,
            final_distance_to_goal=0.01,
            final_nav_distance_to_goal=0.01,
            final_yaw_error_rad=0.01,
        )
        manager.position_tolerance_m = 0.1
        manager.yaw_tolerance_rad = 0.1
        manager._base_cfg = {}
        manager._request_id = "goal"
        manager._robot_base_state_is_invalid = mock.Mock(return_value=False)
        manager._ensure_bridge_client = mock.Mock(return_value=bridge_client)
        manager._step_dt = mock.Mock(return_value=1.0 / 60.0)
        manager._update_pose_result_fields = mock.Mock()
        manager._sim_time = mock.Mock(side_effect=[10.0, 10.3])
        manager._write_debug_snapshot = mock.Mock(return_value={"control": {}})
        manager._log_result_summary = mock.Mock()
        manager._finalize_robot_bridge_after_navigation = mock.Mock()

        manager.step()

        self.assertEqual(manager.state, manager.STATE_POST_SUCCESS_SETTLING)
        self.assertFalse(manager.result.done)
        manager._finalize_robot_bridge_after_navigation.assert_called_once_with()

        manager.step()

        self.assertEqual(manager.state, manager.STATE_SUCCEEDED)
        self.assertTrue(manager.result.done)
        self.assertTrue(manager.result.success)
        self.assertEqual(manager._finalize_robot_bridge_after_navigation.call_count, 2)


if __name__ == "__main__":
    unittest.main()
