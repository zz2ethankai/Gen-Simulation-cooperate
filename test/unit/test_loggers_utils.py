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

from core.loggers.utils import log_dual_obs  # noqa: E402
from core.loggers.lmdb_logger import _filter_missing_image_frames  # noqa: E402


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
    def test_missing_image_frames_are_dropped_with_original_step_ids(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        last = np.ones((2, 2, 3), dtype=np.uint8)

        frames, step_ids = _filter_missing_image_frames(
            [first, None, last], [3, 4, 8]
        )

        self.assertEqual(step_ids, [3, 8])
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0], first)
        np.testing.assert_array_equal(frames[1], last)

        empty_frames, empty_step_ids = _filter_missing_image_frames([None, None], [1, 2])
        self.assertEqual(empty_frames, [])
        self.assertEqual(empty_step_ids, [])

    @staticmethod
    def _runtime(gripper_state, arm_name="left"):
        return SimpleNamespace(
            arm_name=arm_name,
            execution_status=lambda: SimpleNamespace(gripper_state=gripper_state),
        )

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
        runtime_ports = {"panda_omron": {"left": self._runtime(1.0)}}
        logger = _FakeLogger()

        log_dual_obs(logger, obs, {}, runtime_ports)

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
        runtime_ports = {
            "custom_dual": {
                "left": self._runtime(1.0),
                "right": self._runtime(-1.0, arm_name="right"),
            }
        }
        logger = _FakeLogger()

        log_dual_obs(logger, obs, {}, runtime_ports)

        self.assertEqual(logger.steps, 1)
        robot_actions = logger.actions["custom_dual"]
        self.assertIn("master_actions.left_joint.position", robot_actions)
        self.assertIn("master_actions.right_joint.position", robot_actions)
        self.assertEqual(robot_actions["master_actions.left_gripper.openness"], [1.0])
        self.assertEqual(robot_actions["master_actions.right_gripper.openness"], [0.0])


if __name__ == "__main__":
    unittest.main()
