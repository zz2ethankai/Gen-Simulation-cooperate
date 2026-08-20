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


def _load_retarget_helper():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "retarget_pick_phase_commands"
    )

    def tf_matrix_from_pose(translation, orientation):
        x, y, z, w = np.asarray(orientation, dtype=float)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        matrix[:3, 3] = np.asarray(translation, dtype=float)
        return matrix

    def pose_from_tf_matrix(matrix):
        return np.asarray(matrix[:3, 3]), np.array([0.0, 0.0, 0.0, 1.0])

    namespace = {
        "np": np,
        "MotionPhaseCommand": type("MotionPhaseCommand", (), {}),
        "tf_matrix_from_pose": tf_matrix_from_pose,
        "pose_from_tf_matrix": pose_from_tf_matrix,
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["retarget_pick_phase_commands"]


def test_retarget_translation_uses_object_center_delta_not_rotation_lever_arm():
    helper = _load_retarget_helper()
    old_translation = np.array([-0.176, 0.114, 0.784])
    current_translation = old_translation + np.array([0.007, -0.003, 0.005])
    angle = np.deg2rad(9.0)
    old_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    current_orientation = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

    controller = SimpleNamespace(
        _pick_plan_references={
            "target": {
                "object_pose": (old_translation.copy(), old_orientation.copy()),
                "world_armbase_tf": np.eye(4),
            }
        },
        _get_pick_object_world_pose=lambda _name: (
            current_translation.copy(),
            current_orientation.copy(),
        ),
        get_pick_armbase_transform=lambda: np.eye(4),
    )

    translation_delta, rotation_delta = helper(controller, "target", [])

    np.testing.assert_allclose(translation_delta, np.linalg.norm(current_translation - old_translation))
    assert abs(translation_delta - 0.00911043358) < 1e-8
    assert abs(rotation_delta - 9.0) < 1e-6
