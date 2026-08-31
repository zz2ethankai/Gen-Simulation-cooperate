import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))


transform_stub = types.ModuleType("core.utils.transformation_utils")
transform_stub.get_fk_solution = lambda joint_position: np.eye(4)
transform_stub.pose_to_6d = lambda pose: np.zeros(6)
sys.modules.setdefault("core.utils.transformation_utils", transform_stub)

lmdb_stub = types.ModuleType("lmdb")
lmdb_stub.open = lambda *args, **kwargs: None
sys.modules.setdefault("lmdb", lmdb_stub)

from core.loggers.utils import log_dual_obs, resolve_video_sampling  # noqa: E402


class _FakeLogger:
    def __init__(self):
        self.proprio = {}
        self.objects = {}
        self.actions = {}
        self.steps = 0

    def add_proprio_data(self, robot, key, value):
        self.proprio.setdefault(robot, {}).setdefault(key, []).append(value)

    def add_object_data(self, robot, key, value):
        self.objects.setdefault(robot, {}).setdefault(key, []).append(value)

    def add_action_data(self, robot, key, value):
        self.actions.setdefault(robot, {}).setdefault(key, []).append(value)

    def count_timestep(self):
        self.steps += 1


class LogDualObsTest(unittest.TestCase):
    def test_single_arm_robot_logs_master_actions_from_obs_keys(self):
        obs = {
            "robots": {
                "panda_omron": {
                    "states.joint.position": np.zeros(7),
                    "states.gripper.position": np.array([0.08]),
                    "states.gripper.pose": np.zeros(6),
                    "qvel": np.zeros(9),
                }
            }
        }
        controllers = {"panda_omron": {"left": SimpleNamespace(_gripper_state=1.0)}}
        logger = _FakeLogger()

        log_dual_obs(logger, obs, {}, controllers)

        self.assertEqual(logger.steps, 1)
        robot_actions = logger.actions["panda_omron"]
        self.assertIn("master_actions.joint.position", robot_actions)
        self.assertIn("master_actions.gripper.position", robot_actions)
        self.assertEqual(robot_actions["master_actions.gripper.openness"], [1.0])

    def test_dual_arm_robot_logs_left_and_right_master_actions_from_obs_keys(self):
        obs = {
            "robots": {
                "custom_dual": {
                    "states.left_joint.position": np.zeros(7),
                    "states.right_joint.position": np.ones(7),
                    "states.left_gripper.position": np.array([0.08]),
                    "states.right_gripper.position": np.array([0.0]),
                }
            }
        }
        controllers = {
            "custom_dual": {
                "left": SimpleNamespace(_gripper_state=1.0),
                "right": SimpleNamespace(_gripper_state=-1.0),
            }
        }
        logger = _FakeLogger()

        log_dual_obs(logger, obs, {}, controllers)

        self.assertEqual(logger.steps, 1)
        robot_actions = logger.actions["custom_dual"]
        self.assertIn("master_actions.left_joint.position", robot_actions)
        self.assertIn("master_actions.right_joint.position", robot_actions)
        self.assertEqual(robot_actions["master_actions.left_gripper.openness"], [1.0])
        self.assertEqual(robot_actions["master_actions.right_gripper.openness"], [0.0])

    def test_humanoid_logs_actual_low_level_locomotion_targets(self):
        obs = {
            "robots": {
                "unitree_g1": {
                    "states.body_joint.position": np.arange(29, dtype=float),
                    "states.body_joint.velocity": np.zeros(29),
                    "states.base.position": np.array([0.0, 0.0, 0.8]),
                    "states.base.orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                    "qvel": np.zeros(35),
                }
            }
        }
        base_bridges = {
            "unitree_g1": SimpleNamespace(
                get_logging_action_snapshot=lambda: {
                    "vx_body": 0.2,
                    "vy_body": 0.0,
                    "wz_body": 0.1,
                    "locomotion_mode": 1,
                    "joint_position_targets": np.full(29, 0.25),
                    "joint_velocity_targets": np.zeros(29),
                    "joint_efforts": np.full(29, 0.5),
                },
                get_logging_state_snapshot=lambda: {
                    "actual_vx_body": 0.18,
                    "actual_vy_body": 0.0,
                    "actual_wz_body": 0.09,
                },
            )
        }
        logger = _FakeLogger()

        log_dual_obs(logger, obs, {}, {}, base_bridges=base_bridges)

        actions = logger.actions["unitree_g1"]
        np.testing.assert_allclose(actions["master_actions.body_joint.position"][0], 0.25)
        np.testing.assert_allclose(actions["master_actions.body_joint.velocity"][0], 0.0)
        np.testing.assert_allclose(actions["master_actions.body_joint.effort"][0], 0.5)
        self.assertEqual(actions["base_actions.locomotion_mode"], [1])


class ResolveVideoSamplingTest(unittest.TestCase):
    def test_configured_video_rate_decimates_200_hz_physics_to_50_hz(self):
        video_fps, frame_stride = resolve_video_sampling(1.0 / 200.0, 50)

        self.assertEqual(video_fps, 50)
        self.assertEqual(frame_stride, 4)

    def test_configured_video_rate_uses_rendering_cadence_when_available(self):
        video_fps, frame_stride = resolve_video_sampling(
            1.0 / 200.0,
            50,
            rendering_dt=1.0 / 50.0,
        )

        self.assertEqual(video_fps, 50)
        self.assertEqual(frame_stride, 1)

    def test_default_video_rate_keeps_existing_physics_step_cadence(self):
        video_fps, frame_stride = resolve_video_sampling(
            1.0 / 200.0,
            None,
            rendering_dt=1.0 / 50.0,
        )

        self.assertEqual(video_fps, 200)
        self.assertEqual(frame_stride, 1)

    def test_default_video_rate_preserves_existing_per_step_recording(self):
        video_fps, frame_stride = resolve_video_sampling(1.0 / 30.0, None)

        self.assertEqual(video_fps, 30)
        self.assertEqual(frame_stride, 1)


if __name__ == "__main__":
    unittest.main()
