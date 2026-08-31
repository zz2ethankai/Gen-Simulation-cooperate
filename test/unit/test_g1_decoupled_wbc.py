from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.mobile.g1_decoupled_wbc import (  # noqa: E402
    G1DecoupledWbcPolicy,
    RobotState,
)
from core.mobile.g1_locomotion_driver import (  # noqa: E402
    G1NavigationCommandAdapter,
)


class _FakeSession:
    def __init__(self, action_value: float):
        self.action_value = float(action_value)
        self.inputs = []

    def get_inputs(self):
        return [type("Input", (), {"name": "obs"})()]

    def run(self, _outputs, feed):
        observation = np.asarray(feed["obs"], dtype=np.float32)
        self.inputs.append(observation.copy())
        return [np.full((1, 15), self.action_value, dtype=np.float32)]


def _state(*, yaw: float = 0.0) -> RobotState:
    return RobotState(
        body_q=np.zeros(29, dtype=np.float64),
        body_dq=np.zeros(29, dtype=np.float64),
        base_quat=np.asarray(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64
        ),
        base_ang_vel=np.zeros(3, dtype=np.float64),
        pelvis_z=0.8,
    )


class G1NavigationCommandAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = G1NavigationCommandAdapter(
            hard_limits={
                "max_velocity": [0.35, 0.20, 0.50],
                "min_velocity": [0.0, -0.20, -0.50],
                "max_accel": [0.30, 0.20, 0.50],
                "max_decel": [-0.40, -0.30, -0.70],
            },
            yaw_target_horizon_sec=0.5,
        )

    def test_aligned_translation_walks_forward_and_ignores_final_yaw_rate(self):
        command = self.adapter.translate(
            vx_body=0.8,
            vy_body=0.0,
            wz_body=0.8,
            current_yaw=0.3,
        )

        np.testing.assert_allclose(command.navigate_cmd, [0.35, 0.0, 0.0, 0.3])

    def test_pure_lateral_translation_is_forwarded_without_turning(self):
        command = self.adapter.translate(
            vx_body=0.0,
            vy_body=0.3,
            wz_body=0.0,
            current_yaw=0.0,
        )

        self.assertEqual(command.vx_body, 0.0)
        self.assertEqual(command.vy_body, 0.2)
        self.assertEqual(command.turn_flag, 0.0)
        self.assertAlmostEqual(command.target_yaw, 0.0)

    def test_diagonal_translation_still_turns_to_requested_heading(self):
        command = self.adapter.translate(
            vx_body=0.02,
            vy_body=0.3,
            wz_body=0.0,
            current_yaw=0.0,
        )

        self.assertEqual(command.vx_body, 0.0)
        self.assertEqual(command.vy_body, 0.0)
        self.assertEqual(command.turn_flag, 1.0)
        self.assertAlmostEqual(command.target_yaw, 0.25)

    def test_backward_translation_turns_around_before_walking(self):
        command = self.adapter.translate(
            vx_body=-0.2,
            vy_body=0.0,
            wz_body=0.0,
            current_yaw=np.pi - 0.1,
        )

        self.assertEqual(command.vx_body, 0.0)
        self.assertEqual(command.vy_body, 0.0)
        self.assertEqual(command.turn_flag, 1.0)
        self.assertAlmostEqual(command.target_yaw, -np.pi + 0.15)

    def test_small_path_heading_error_walks_with_held_heading(self):
        self.adapter.reset(current_yaw=0.5)
        command = self.adapter.translate(
            vx_body=0.3,
            vy_body=0.03,
            wz_body=0.0,
            current_yaw=0.5,
        )

        self.assertAlmostEqual(command.vx_body, np.hypot(0.3, 0.03))
        self.assertEqual(command.vy_body, 0.0)
        self.assertEqual(command.turn_flag, 0.0)
        self.assertAlmostEqual(command.target_yaw, 0.5)

    def test_zero_translation_uses_final_yaw_rate_target(self):
        command = self.adapter.translate(
            vx_body=0.0,
            vy_body=0.0,
            wz_body=0.5,
            current_yaw=np.pi - 0.1,
        )

        np.testing.assert_allclose(
            command.navigate_cmd,
            [0.0, 0.0, 1.0, -np.pi + 0.15],
        )

    def test_near_goal_translation_prioritizes_final_heading(self):
        command = self.adapter.translate(
            vx_body=0.0,
            vy_body=0.1,
            wz_body=-0.5,
            current_yaw=0.8,
        )

        np.testing.assert_allclose(command.navigate_cmd, [0.0, 0.0, 1.0, 0.55])

    def test_zero_command_holds_last_target_heading(self):
        self.adapter.reset(current_yaw=-0.4)
        turning = self.adapter.translate(
            vx_body=0.0,
            vy_body=0.0,
            wz_body=0.2,
            current_yaw=-0.4,
        )
        holding = self.adapter.translate(
            vx_body=0.0,
            vy_body=0.0,
            wz_body=0.0,
            current_yaw=-0.32,
        )

        self.assertAlmostEqual(turning.target_yaw, -0.3)
        self.assertAlmostEqual(holding.target_yaw, -0.3)


class G1DecoupledWbcPolicyTest(unittest.TestCase):
    def test_zero_command_uses_balance_policy_and_builds_humano_observation(self):
        balance = _FakeSession(0.0)
        walk = _FakeSession(1.0)
        policy = G1DecoupledWbcPolicy(balance_session=balance, walk_session=walk)

        command = policy.step(_state(), np.asarray([0.0, 0.0, 0.0, 0.0]), env_step_dt=0.005)

        self.assertEqual(len(balance.inputs), 1)
        self.assertEqual(len(walk.inputs), 0)
        self.assertEqual(balance.inputs[0].shape, (1, 516))
        np.testing.assert_allclose(command.q_target[:15], policy.default_lower_body_angles)
        np.testing.assert_allclose(command.q_target[15:], policy.default_upper_body_angles)

    def test_walk_policy_runs_at_50hz_and_reuses_target_for_four_200hz_steps(self):
        balance = _FakeSession(0.0)
        walk = _FakeSession(0.4)
        policy = G1DecoupledWbcPolicy(balance_session=balance, walk_session=walk)
        navigate_cmd = np.asarray([0.3, 0.0, 0.0, 0.0], dtype=np.float64)

        commands = [policy.step(_state(), navigate_cmd, env_step_dt=0.005) for _ in range(4)]

        self.assertEqual(len(walk.inputs), 1)
        for command in commands[1:]:
            np.testing.assert_allclose(command.q_target, commands[0].q_target)
        policy.step(_state(), navigate_cmd, env_step_dt=0.005)
        self.assertEqual(len(walk.inputs), 2)

    def test_lateral_velocity_reaches_humano_walk_observation(self):
        walk = _FakeSession(0.4)
        policy = G1DecoupledWbcPolicy(
            balance_session=_FakeSession(0.0),
            walk_session=walk,
        )

        policy.step(
            _state(),
            np.asarray([0.0, -0.2, 0.0, 0.0]),
            env_step_dt=0.005,
        )

        self.assertEqual(len(walk.inputs), 1)
        np.testing.assert_allclose(walk.inputs[0][0, -86:-83], [0.0, -0.4, 0.0])

    def test_reset_clears_history_and_cached_action(self):
        walk = _FakeSession(0.4)
        policy = G1DecoupledWbcPolicy(balance_session=_FakeSession(0.0), walk_session=walk)
        policy.step(_state(), np.asarray([0.3, 0.0, 0.0, 0.0]), env_step_dt=0.005)

        policy.reset()

        self.assertEqual(policy.inference_count, 0)
        self.assertEqual(policy.history_length, 0)


if __name__ == "__main__":
    unittest.main()
