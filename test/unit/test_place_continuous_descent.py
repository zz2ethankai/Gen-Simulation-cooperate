"""Offline coverage for continuous Physics-schema Place execution guards."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.curobo.execution import ControllerExecution  # noqa: E402
from core.controllers.curobo.components import ComponentPort  # noqa: E402
from core.controllers.curobo.phase_execution import PhaseExecutor  # noqa: E402
from core.planning.domain_types import JointTrajectory  # noqa: E402


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


class _Controller:
    name = "panda_omron"
    lr_name = "left"
    ds_ratio = 2

    def __init__(self):
        self.phase_executor = PhaseExecutor()

    @staticmethod
    def get_ee_pose():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _forward_kinematic_batch(joint_positions, *, joint_names):
        del joint_names
        values = np.asarray(joint_positions, dtype=float)
        return values[:, :3]


def _command(**overrides):
    params = {
        "max_cartesian_step_m": 0.01,
        "max_path_length_ratio": 1.5,
        "max_path_deviation_m": 0.01,
    }
    params.update(overrides)
    return SimpleNamespace(
        target_position=np.array([0.0, 0.0, -0.08]),
        params=params,
        phase=SimpleNamespace(value="terminal_place_descent"),
    )


def _plan(points):
    return JointTrajectory(
        positions=[np.asarray(point, dtype=float).tolist() for point in points],
        joint_names=("joint_0", "joint_1", "joint_2"),
    )


def _execution_port(controller):
    return ComponentPort(
        {
            "name": controller.name,
            "lr_name": controller.lr_name,
            "ds_ratio": controller.ds_ratio,
            "phase_executor": controller.phase_executor,
            "get_ee_pose": controller.get_ee_pose,
            "_forward_kinematic_batch": controller._forward_kinematic_batch,
            "tensor_args": SimpleNamespace(
                to_device=lambda value: np.asarray(value, dtype=float)
            ),
        }
    )


def test_continuous_descent_accepts_bounded_straight_path():
    controller = _Controller()
    execution = ControllerExecution(_execution_port(controller))
    validate = execution._validate_continuous_place_plan
    points = [np.array([0.0, 0.0, -0.002 * index]) for index in range(41)]

    assert validate(_command(), _plan(points)) is True


def test_continuous_descent_rejects_large_per_frame_advance():
    controller = _Controller()
    execution = ControllerExecution(_execution_port(controller))
    validate = execution._validate_continuous_place_plan
    points = [np.array([0.0, 0.0, -0.006 * index]) for index in range(15)]

    assert validate(_command(), _plan(points)) is False


def test_contact_stop_clears_plan_without_losing_active_phase():
    command = _command()
    phase_executor = PhaseExecutor()
    phase_executor.install(
        JointTrajectory(positions=[[0.0]], joint_names=("joint_0",))
    )
    port = ComponentPort(
        {
            "_active_phase_command": command,
            "phase_executor": phase_executor,
            "_phase_plan_finished": False,
            "_last_arm_action": np.ones(6),
            "name": "panda_omron",
            "lr_name": "left",
        }
    )

    execution = ControllerExecution(port)
    execution.complete_terminal_place_on_contact(command)

    assert execution._active_phase_command is command
    assert execution.phase_executor.current is None
    assert execution.phase_executor.index == 0
    assert execution._phase_plan_finished is True
    assert execution._last_arm_action is None
    assert command.params["contact_stop_logged"] is True
