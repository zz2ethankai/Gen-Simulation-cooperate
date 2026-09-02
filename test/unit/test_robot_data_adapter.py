from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

transform_stub = types.ModuleType("core.utils.transformation_utils")
transform_stub.get_fk_solution = lambda joint_position: np.eye(4)
transform_stub.pose_to_6d = lambda pose: np.zeros(6)
sys.modules.setdefault("core.utils.transformation_utils", transform_stub)

lmdb_logger_stub = types.ModuleType("core.loggers.lmdb_logger")
lmdb_logger_stub.LmdbLogger = object
sys.modules.setdefault("core.loggers.lmdb_logger", lmdb_logger_stub)

from core.loggers.utils import log_dual_obs  # noqa: E402


def _runtime(gripper_state):
    return types.SimpleNamespace(
        execution=types.SimpleNamespace(
            execution_status=lambda: types.SimpleNamespace(
                gripper_state=gripper_state
            )
        )
    )


class FakeLogger:
    def __init__(self, adapter):
        self.robot_data_adapters = {"robot_0": adapter}
        self.proprio = {}
        self.objects = {}
        self.actions = {}
        self.timesteps = 0

    def add_proprio_data(self, robot, key, value):
        self.proprio[(robot, key)] = value

    def add_object_data(self, robot, key, value):
        self.objects[(robot, key)] = value

    def add_action_data(self, robot, key, value):
        self.actions[(robot, key)] = value

    def count_timestep(self):
        self.timesteps += 1


def test_dual_arm_action_schema_is_driven_by_profile_adapter():
    adapter = {
        "arms": {
            "left": {
                "action_name": "left",
                "joint_position_key": "states.left_joint.position",
                "gripper_position_key": "states.left_gripper.position",
            },
            "right": {
                "action_name": "right",
                "joint_position_key": "states.right_joint.position",
                "gripper_position_key": "states.right_gripper.position",
            },
        }
    }
    logger = FakeLogger(adapter)
    controllers = {
        "robot_0": {
            "left": _runtime(1.0),
            "right": _runtime(-1.0),
        }
    }
    observations = {
        "robots": {
            "robot_0": {
                "states.left_joint.position": np.array([1.0, 2.0]),
                "states.right_joint.position": np.array([3.0, 4.0]),
                "states.left_gripper.position": np.array([0.05]),
                "states.right_gripper.position": np.array([0.01]),
            }
        },
        "objects": {"cup": {"pose": [0.0, 0.0, 0.0]}},
    }
    action = {
        "robot_0": {
            "raw_action": [
                {"lr_name": "right", "arm_action": np.array([5.0, 6.0])}
            ]
        }
    }

    log_dual_obs(logger, observations, action, controllers)

    np.testing.assert_array_equal(
        logger.actions[("robot_0", "master_actions.left_joint.position")],
        [1.0, 2.0],
    )
    np.testing.assert_array_equal(
        logger.actions[("robot_0", "master_actions.right_joint.position")],
        [5.0, 6.0],
    )
    assert logger.actions[("robot_0", "master_actions.left_gripper.openness")] == 1.0
    assert logger.actions[("robot_0", "master_actions.right_gripper.openness")] == 0.0
    assert logger.objects[("robot_0", "cup/pose")] == [0.0, 0.0, 0.0]
    assert logger.timesteps == 1


def test_single_arm_adapter_preserves_unprefixed_dataset_schema():
    adapter = {
        "arms": {
            "left": {
                "action_name": "",
                "joint_position_key": "states.joint.position",
                "gripper_position_key": "states.gripper.position",
                "gripper_pose_key": "states.gripper.pose",
            }
        }
    }
    logger = FakeLogger(adapter)
    controllers = {"robot_0": {"left": _runtime(1.0)}}
    observations = {
        "robots": {
            "robot_0": {
                "states.joint.position": np.arange(7),
                "states.gripper.position": np.array([0.04]),
                "states.gripper.pose": np.arange(6),
            }
        },
        "objects": {},
    }

    log_dual_obs(logger, observations, {}, controllers)

    assert ("robot_0", "master_actions.joint.position") in logger.actions
    assert ("robot_0", "master_actions.gripper.position") in logger.actions
    assert ("robot_0", "master_actions.gripper.pose") in logger.actions


def test_raw_action_does_not_eagerly_read_missing_observation_fallback():
    adapter = {
        "arms": {
            "left": {
                "action_name": "left",
                "joint_position_key": "states.joint.position",
                "gripper_position_key": "states.gripper.position",
                "gripper_pose_key": None,
            }
        }
    }
    logger = FakeLogger(adapter)
    controllers = {"robot_0": {"left": _runtime(1.0)}}
    observations = {
        "robots": {
            "robot_0": {
                # Deliberately omit states.joint.position: the raw action is
                # authoritative for this step and the fallback must stay lazy.
                "states.gripper.position": np.array([0.04]),
            }
        },
        "objects": {},
    }
    raw_action = np.arange(7)
    action = {
        "robot_0": {
            "raw_action": [
                {"lr_name": "left", "arm_action": raw_action}
            ]
        }
    }

    log_dual_obs(logger, observations, action, controllers)

    np.testing.assert_array_equal(
        logger.actions[("robot_0", "master_actions.left_joint.position")],
        raw_action,
    )
