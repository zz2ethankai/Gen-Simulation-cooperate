"""Focused host tests for batched Cartesian FK validation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
    / "template_controller.py"
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


def _load_controller_method(method_name, **namespace):
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
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
    method_namespace = {"np": np, "torch": torch, **namespace}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, _CONTROLLER_PATH, "exec"), method_namespace)
    return method_namespace[method_name]


class _NamedPath:
    def __init__(self, position, joint_names):
        self.position = position
        self.joint_names = list(joint_names)

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]
        return _NamedPath(self.position[..., indices], joint_names)


def test_measure_cartesian_path_uses_one_batched_fk_after_name_reorder():
    class _Controller:
        measure_cartesian_path = _load_controller_method(
            "measure_cartesian_path"
        )

        def __init__(self):
            self.raw_js_names = ["a", "b"]
            self.batch_inputs = []

        def _forward_kinematic_batch(self, joint_positions, **kwargs):
            del kwargs
            self.batch_inputs.append(joint_positions)
            points = np.asarray(joint_positions.detach().cpu())
            return np.concatenate([points, np.zeros((len(points), 1))], axis=1)

    path = _NamedPath(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        ["b", "a", "locked_finger"],
    )
    controller = _Controller()

    ratio, deviation = controller.measure_cartesian_path(
        path, np.zeros(3), np.array([0.5, 0.5, 0.0])
    )

    assert ratio == pytest.approx(1.0)
    assert deviation == pytest.approx(0.0)
    assert len(controller.batch_inputs) == 1
    assert tuple(controller.batch_inputs[0].shape) == (2, 2)
    torch.testing.assert_close(
        controller.batch_inputs[0], torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    )


def test_continuous_place_validation_uses_one_batched_fk():
    validate = _load_controller_method(
        "_validate_continuous_place_plan",
        LOGGER=_Logger(),
        MotionPhaseCommand=object,
    )

    class _Controller:
        name = "stub"
        lr_name = "left"
        ds_ratio = 2

        def __init__(self):
            self.batch_inputs = []

        @staticmethod
        def get_ee_pose():
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def _forward_kinematic_batch(self, joint_positions, **kwargs):
            del kwargs
            self.batch_inputs.append(joint_positions)
            return np.asarray(joint_positions.detach().cpu())[:, :3]

    command = SimpleNamespace(
        target_position=np.array([0.0, 0.0, -0.08]),
        params={
            "max_cartesian_step_m": 0.01,
            "max_path_length_ratio": 1.5,
            "max_path_deviation_m": 0.01,
        },
    )
    plan = SimpleNamespace(
        position=torch.tensor(
            [[0.0, 0.0, -0.002 * index] for index in range(41)]
        )
    )
    controller = _Controller()

    assert validate(controller, command, plan) is True
    assert len(controller.batch_inputs) == 1
    assert tuple(controller.batch_inputs[0].shape) == (41, 3)


def test_batch_fk_helper_builds_named_state_and_transfers_once():
    class _JointState:
        received = None

        @classmethod
        def from_position(cls, position, joint_names):
            cls.received = (position, list(joint_names))
            return SimpleNamespace(position=position, joint_names=list(joint_names))

    class _ToolPoses:
        @staticmethod
        def get_link_pose(_link_name):
            return SimpleNamespace(
                position=torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.float32,
                )
            )

    class _Kinematics:
        tool_frames = ["tool"]

        def __init__(self):
            self.calls = []

        def compute_kinematics(self, state):
            self.calls.append(state)
            return SimpleNamespace(tool_poses=_ToolPoses())

    helper = _load_controller_method(
        "_forward_kinematic_batch", JointState=_JointState
    )
    controller = SimpleNamespace(
        tensor_args=SimpleNamespace(
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
        ),
        kin_model=_Kinematics(),
        _planner_joint_names=lambda: ["joint_a", "joint_b"],
    )

    result = helper(controller, np.zeros((2, 2), dtype=np.float64))

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert len(controller.kin_model.calls) == 1
    assert _JointState.received[1] == ["joint_a", "joint_b"]
    assert tuple(_JointState.received[0].shape) == (2, 2)


def test_batch_fk_helper_reorders_source_names_before_native_fk():
    class _JointState:
        received = None

        @classmethod
        def from_position(cls, position, joint_names):
            cls.received = (position, list(joint_names))
            return SimpleNamespace(position=position, joint_names=list(joint_names))

    class _ToolPoses:
        @staticmethod
        def get_link_pose(_link_name):
            return SimpleNamespace(position=torch.zeros((2, 3)))

    class _Kinematics:
        tool_frames = ["tool"]

        @staticmethod
        def compute_kinematics(_state):
            return SimpleNamespace(tool_poses=_ToolPoses())

    helper = _load_controller_method(
        "_forward_kinematic_batch", JointState=_JointState
    )
    controller = SimpleNamespace(
        tensor_args=SimpleNamespace(
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
        ),
        kin_model=_Kinematics(),
        _planner_joint_names=lambda: ["joint_a", "joint_b"],
    )

    helper(
        controller,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        joint_names=["joint_b", "joint_a"],
    )

    torch.testing.assert_close(
        _JointState.received[0], torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    )
    assert _JointState.received[1] == ["joint_a", "joint_b"]
