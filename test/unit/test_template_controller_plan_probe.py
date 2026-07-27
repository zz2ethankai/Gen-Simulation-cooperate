"""Regression tests for side-effect-free controller planning probes."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
    / "template_controller.py"
)


def _load_plan_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_forward_from_joint_positions"
    )
    namespace = {
        "np": np,
        "Optional": Optional,
        "filter_paths_by_position_error": lambda paths, errors: [True] * len(paths),
        "filter_paths_by_rotation_error": lambda paths, errors: [True] * len(paths),
        "sort_by_difference_js": lambda paths, weights=None: list(reversed(range(len(paths)))),
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["test_forward_from_joint_positions"]


class _Plan:
    def __init__(self, endpoint):
        self._states = [SimpleNamespace(position=torch.tensor(endpoint))]

    def get_ordered_joint_state(self, joint_names):
        assert joint_names == ["arm_0", "arm_1"]
        return self

    def __getitem__(self, index):
        return self._states[index]


class _Result:
    def __init__(self, success, endpoint):
        self.success = torch.tensor([success])
        self._plan = _Plan(endpoint)

    def get_interpolated_plan(self):
        return self._plan


class _BatchResult:
    def __init__(self):
        self.success = torch.tensor([True, True])
        self.position_error = torch.zeros(2)
        self.rotation_error = torch.zeros(2)
        self._paths = [_Plan([0.3, 0.4]), _Plan([0.7, 0.8])]

    def get_successful_paths(self):
        return self._paths


class _Controller:
    test_forward_from_joint_positions = _load_plan_probe()

    def __init__(self, success=True):
        self.arm_indices = np.array([1, 3])
        self.raw_js_names = ["arm_0", "arm_1"]
        self.use_batch = False
        self.num_plan_failed = 4
        self.cmd_plan = "runtime-plan"
        self.robot = SimpleNamespace(
            dof_names=["base", "arm_0", "gripper", "arm_1"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.array([9.0, 0.1, 0.02, 0.2]),
                velocities=np.zeros(4),
            ),
        )
        self.success = success
        self.planning_start = None

    def plan(self, ee_trans, ee_ori, sim_js, js_names):
        self.planning_start = np.asarray(sim_js.positions).copy()
        return _Result(self.success, [0.7, 0.8])


def test_plan_probe_uses_requested_start_and_preserves_runtime_state():
    controller = _Controller(success=True)

    success, endpoint, _ = controller.test_forward_from_joint_positions(
        np.ones(3),
        np.ones(4),
        start_arm_positions=np.array([0.4, 0.5]),
    )

    assert success is True
    np.testing.assert_allclose(controller.planning_start, [9.0, 0.4, 0.02, 0.5])
    np.testing.assert_allclose(endpoint, [0.7, 0.8])
    assert controller.num_plan_failed == 4
    assert controller.cmd_plan == "runtime-plan"


def test_failed_plan_probe_returns_no_endpoint_or_runtime_failure_side_effect():
    controller = _Controller(success=False)

    success, endpoint, _ = controller.test_forward_from_joint_positions(np.ones(3), np.ones(4))

    assert success is False
    assert endpoint is None
    assert controller.num_plan_failed == 4
    assert controller.cmd_plan == "runtime-plan"


def test_batch_plan_probe_returns_endpoint_from_runtime_preferred_path():
    controller = _Controller(success=True)
    controller.use_batch = True
    controller.tensor_args = SimpleNamespace(to_device=lambda value: value)
    controller.motion_gen = SimpleNamespace(get_full_js=lambda path: path)
    controller._get_sort_path_weights = lambda: None
    controller.plan = lambda *args, **kwargs: _BatchResult()

    success, endpoint, _ = controller.test_forward_from_joint_positions(np.ones(3), np.ones(4))

    assert success is True
    np.testing.assert_allclose(endpoint, [0.7, 0.8])
