"""Tests for preserving measured motion state across manipulation replans."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
    / "template_controller.py"
)


def _load_derivative_helper():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_joint_state_derivatives"
    )
    namespace = {"np": np}
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["_joint_state_derivatives"]


def test_joint_state_derivatives_preserve_measured_values():
    helper = _load_derivative_helper()
    state = SimpleNamespace(
        positions=np.zeros(3),
        velocities=np.array([0.1, -0.2, 0.3]),
        accelerations=np.array([0.4, 0.5, -0.6]),
        jerks=np.array([-0.7, 0.8, 0.9]),
    )

    velocity, acceleration, jerk = helper(state)

    np.testing.assert_allclose(velocity, state.velocities)
    np.testing.assert_allclose(acceleration, state.accelerations)
    np.testing.assert_allclose(jerk, state.jerks)


def test_joint_state_derivatives_fall_back_for_missing_or_invalid_fields():
    helper = _load_derivative_helper()
    state = SimpleNamespace(
        positions=np.zeros(3),
        velocities=np.array([0.1, np.nan, 0.3]),
    )

    velocity, acceleration, jerk = helper(state)

    np.testing.assert_array_equal(velocity, np.zeros(3))
    np.testing.assert_array_equal(acceleration, np.zeros(3))
    np.testing.assert_array_equal(jerk, np.zeros(3))
