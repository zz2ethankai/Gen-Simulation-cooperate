"""Regression tests for trajectory replacement in TemplateController."""

import ast
from pathlib import Path
from types import SimpleNamespace

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


class _ArticulationAction:
    def __init__(self, joint_positions, joint_velocities=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_indices = joint_indices


def _load_ee_forward():
    """Load the method alone so this unit test does not require Isaac Sim or cuRobo."""
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "ee_forward"
    )
    namespace = {
        "np": np,
        "torch": torch,
        "ArticulationAction": _ArticulationAction,
        "LOGGER": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["ee_forward"]


class _TensorArgs:
    @staticmethod
    def to_device(value):
        return torch.as_tensor(value, dtype=torch.float32)


class _Controller:
    ee_forward = _load_ee_forward()

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
        self.use_batch = False
        self.ds_ratio = 1
        self.lr_name = "left"
        self._last_arm_action = np.array([0.7, 0.8, 0.9])
        self._last_command_name = "test"
        self._phase_plan_started = False
        self._phase_plan_failed = False
        self._last_commanded_arm_position = None
        stale_waypoint = SimpleNamespace(
            position=torch.tensor([1.1, 1.2, 1.3]),
            velocity=torch.zeros(3),
        )
        self.cmd_plan = [stale_waypoint]
        self.cmd_idx = 0
        self.plan_calls = 0

    def plan(self, ee_trans, ee_ori, sim_js, js_names):
        self.plan_calls += 1
        return SimpleNamespace(success=torch.tensor([False]))

    @staticmethod
    def _log_plan_result(*args, **kwargs):
        return None

    @staticmethod
    def get_gripper_action():
        return np.array([0.01])


def test_failed_replan_holds_current_joints_instead_of_replaying_stale_plan():
    controller = _Controller()
    stale_waypoint = controller.cmd_plan[0].position.numpy().copy()

    action = controller.ee_forward(np.ones(3), np.ones(4))

    current_arm_position = controller.robot.get_joints_state().positions[controller.arm_indices]
    np.testing.assert_allclose(action["arm_action"], current_arm_position)
    assert not np.allclose(action["arm_action"], stale_waypoint)
    assert controller.plan_calls == 1
    assert controller.num_plan_failed == 1
    assert controller.cmd_plan is None
    assert controller.cmd_idx == 0
