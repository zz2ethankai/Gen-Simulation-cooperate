"""Regression tests for trajectory replacement in the execution component."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.controller_execution import ControllerExecution  # noqa: E402
from core.controllers.controller_component import ComponentPort  # noqa: E402
from core.controllers.phase_executor import PhaseExecutor  # noqa: E402
from core.planning.domain_types import JointTrajectory  # noqa: E402
import core.controllers.controller_execution as execution_module  # noqa: E402


class _ArticulationAction:
    def __init__(self, joint_positions=None, joint_velocities=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_indices = joint_indices


class _TensorArgs:
    @staticmethod
    def to_device(value):
        return torch.as_tensor(value, dtype=torch.float32)


class _Controller:
    def __init__(self):
        self.name = "test_robot"
        self.tensor_args = _TensorArgs()
        self.arm_indices = np.array([0, 1, 2])
        self.gripper_indices = np.array([3])
        self.robot = SimpleNamespace(
            dof_names=["joint_0", "joint_1", "joint_2", "gripper"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.array([0.1, 0.2, 0.3, 0.01]),
                velocities=np.zeros(4),
            ),
        )
        self._ee_trans = torch.zeros(3)
        self._ee_ori = torch.zeros(4)
        self._step_idx = 12
        self.num_last_cmd = 3
        self.num_plan_failed = 0
        self.arm_spec = None
        self._gripper_state = 1.0
        self._gripper_joint_position = np.array([1.0])
        self.phase_executor = PhaseExecutor()
        self.ds_ratio = 1
        self.lr_name = "left"
        self._last_arm_action = np.array([0.7, 0.8, 0.9])
        self._last_command_name = "test"
        self._phase_plan_started = False
        self._phase_plan_failed = False
        self._last_commanded_arm_position = None
        stale_waypoint = JointTrajectory(
            positions=[[1.1, 1.2, 1.3]],
            joint_names=("joint_0", "joint_1", "joint_2"),
        )
        self.phase_executor.install(stale_waypoint)
        self.plan_calls = 0

    def _native_plan_pose(self, _ee_trans, _ee_ori, _sim_js, _js_names):
        self.plan_calls += 1
        return SimpleNamespace(success=torch.tensor([False]))

    @staticmethod
    def _log_plan_result(*args, **kwargs):
        del args, kwargs

    @staticmethod
    def _result_success(result):
        return bool(result.success.any().item())

    @staticmethod
    def get_gripper_action():
        return np.array([0.01])

    def _make_action(self, arm_action, gripper_action):
        return {
            "arm_action": np.asarray(arm_action),
            "gripper_action": np.asarray(gripper_action),
        }


def test_failed_replan_holds_current_joints_instead_of_replaying_stale_plan(monkeypatch):
    controller = _Controller()
    execution = ControllerExecution(
        ComponentPort(
            {
                "name": controller.name,
                "tensor_args": controller.tensor_args,
                "arm_indices": controller.arm_indices,
                "gripper_indices": controller.gripper_indices,
                "robot": controller.robot,
                "_ee_trans": controller._ee_trans,
                "_ee_ori": controller._ee_ori,
                "_step_idx": controller._step_idx,
                "num_last_cmd": controller.num_last_cmd,
                "num_plan_failed": controller.num_plan_failed,
                "arm_spec": controller.arm_spec,
                "_gripper_state": controller._gripper_state,
                "_gripper_joint_position": controller._gripper_joint_position,
                "phase_executor": controller.phase_executor,
                "ds_ratio": controller.ds_ratio,
                "lr_name": controller.lr_name,
                "_last_arm_action": controller._last_arm_action,
                "_last_command_name": controller._last_command_name,
                "_phase_plan_started": controller._phase_plan_started,
                "_phase_plan_failed": controller._phase_plan_failed,
                "_last_commanded_arm_position": controller._last_commanded_arm_position,
                "_native_plan_pose": controller._native_plan_pose,
                "_log_plan_result": controller._log_plan_result,
                "_result_success": controller._result_success,
                "get_gripper_action": controller.get_gripper_action,
                "_make_action": controller._make_action,
            }
        )
    )
    monkeypatch.setattr(execution_module, "ArticulationAction", _ArticulationAction)
    stale_waypoint = np.asarray(controller.phase_executor.current.positions[0], dtype=float).copy()

    action = execution.ee_forward(np.ones(3), np.ones(4))

    current_arm_position = controller.robot.get_joints_state().positions[controller.arm_indices]
    np.testing.assert_allclose(action["arm_action"], current_arm_position)
    assert not np.allclose(action["arm_action"], stale_waypoint)
    assert controller.plan_calls == 1
    assert execution.num_plan_failed == 1
    assert execution.phase_executor.current is None
    assert execution.phase_executor.index == 0
