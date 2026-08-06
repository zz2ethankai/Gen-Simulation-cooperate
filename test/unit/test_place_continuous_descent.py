"""Offline coverage for continuous Physics-schema Place execution guards."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = (
    ROOT / "workflows" / "simbox" / "core" / "controllers" / "template_controller.py"
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _load_controller_method(method_name: str):
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {
        "LOGGER": _Logger(),
        "MotionPhaseCommand": object,
        "np": np,
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, CONTROLLER_PATH, "exec"), namespace)
    return namespace[method_name]


class _Controller:
    name = "panda_omron"
    lr_name = "left"
    ds_ratio = 2

    @staticmethod
    def get_ee_pose():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def forward_kinematic(joints):
        return np.asarray(joints[:3], dtype=float), np.array([1.0, 0.0, 0.0, 0.0])


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
    return SimpleNamespace(position=[_Tensor(point) for point in points])


def test_continuous_descent_accepts_bounded_straight_path():
    validate = _load_controller_method("_validate_continuous_place_plan")
    points = [np.array([0.0, 0.0, -0.002 * index]) for index in range(41)]

    assert validate(_Controller(), _command(), _plan(points)) is True


def test_continuous_descent_rejects_large_per_frame_advance():
    validate = _load_controller_method("_validate_continuous_place_plan")
    points = [np.array([0.0, 0.0, -0.006 * index]) for index in range(15)]

    assert validate(_Controller(), _command(), _plan(points)) is False


def test_contact_stop_clears_plan_without_losing_active_phase():
    complete = _load_controller_method("complete_terminal_place_on_contact")
    command = _command()
    controller = SimpleNamespace(
        _active_phase_command=command,
        cmd_plan=object(),
        cmd_idx=7,
        _phase_plan_finished=False,
        _last_arm_action=np.ones(6),
        name="panda_omron",
        lr_name="left",
    )

    complete(controller, command)

    assert controller._active_phase_command is command
    assert controller.cmd_plan is None
    assert controller.cmd_idx == 0
    assert controller._phase_plan_finished is True
    assert controller._last_arm_action is None
    assert command.params["contact_stop_logged"] is True
