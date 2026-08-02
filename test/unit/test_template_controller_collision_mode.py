"""Regression tests for per-Skill controller collision-world switching."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
    / "template_controller.py"
)


def _load_activate_collision_world_mode():
    """Load the method alone so the test does not require Isaac Sim or cuRobo."""

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
        and node.name == "activate_collision_world_mode"
    )
    namespace = {"LOGGER": SimpleNamespace(info=lambda *args, **kwargs: None)}
    method_module = ast.fix_missing_locations(
        ast.Module(body=[method_node], type_ignores=[])
    )
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["activate_collision_world_mode"]


def _load_plan_joint_positions(joint_state_cls):
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_joint_positions"
    )
    namespace = {"np": np, "JointState": joint_state_cls}
    method_module = ast.fix_missing_locations(
        ast.Module(body=[method_node], type_ignores=[])
    )
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["plan_joint_positions"]


class _Manager:
    def __init__(self):
        self.calls = []
        self.world = object()

    def prepare_controller_for_legacy(self, controller):
        self.calls.append(("prepare_legacy", controller.collision_world_mode))

    def build_world_config(self, reference_prim_path):
        self.calls.append(("build_physics", reference_prim_path))
        return self.world

    def resume_controller_physics_world(self, controller):
        self.calls.append(("resume_physics", controller.collision_world_mode))

    def refresh_controller_reference_world(self, controller):
        self.calls.append(("refresh_physics", controller.collision_world_mode))


class _Controller:
    activate_collision_world_mode = _load_activate_collision_world_mode()

    def __init__(self, mode, *, attached=False, manager=None):
        self.name = "robot"
        self.lr_name = "left"
        self.reference_prim_path = "/World/task/robot/base"
        self.collision_world_mode = mode
        self.collision_scene_manager = manager
        self.attached = attached
        self.calls = []

    def clear_plan_and_hold(self):
        self.calls.append("clear_plan")

    def _legacy_update(self):
        self.calls.append("legacy_update")

    def _update_world_if_changed(self, world):
        self.calls.append(("physics_update", world))

    def _configure_execution_stride(self):
        self.calls.append("configure_stride")

    def has_attached_collision_spheres(self):
        return self.attached


def test_switches_from_physics_to_legacy_before_legacy_planning():
    manager = _Manager()
    controller = _Controller("physics_schema", manager=manager)

    controller.activate_collision_world_mode("legacy_stage_scan")

    assert controller.collision_world_mode == "legacy_stage_scan"
    assert manager.calls == [("prepare_legacy", "physics_schema")]
    assert controller.calls == ["clear_plan", "legacy_update", "configure_stride"]


def test_switches_from_legacy_to_fresh_physics_world():
    manager = _Manager()
    controller = _Controller("legacy_stage_scan", manager=manager)

    controller.activate_collision_world_mode("physics_schema")

    assert controller.collision_world_mode == "physics_schema"
    assert manager.calls == [
        ("build_physics", controller.reference_prim_path),
        ("resume_physics", "physics_schema"),
    ]
    assert controller.calls == [
        "clear_plan",
        ("physics_update", manager.world),
        "configure_stride",
    ]


def test_rejects_legacy_attached_state_transfer_into_physics():
    manager = _Manager()
    controller = _Controller("legacy_stage_scan", attached=True, manager=manager)

    with pytest.raises(RuntimeError, match="cannot be transferred"):
        controller.activate_collision_world_mode("physics_schema")

    assert controller.collision_world_mode == "legacy_stage_scan"
    assert manager.calls == []


def test_passthrough_does_not_change_the_active_collision_world():
    manager = _Manager()
    controller = _Controller("physics_schema", manager=manager)

    controller.activate_collision_world_mode("passthrough")

    assert controller.collision_world_mode == "physics_schema"
    assert manager.calls == []
    assert controller.calls == []


def test_same_physics_mode_refreshes_a_moved_mobile_reference():
    manager = _Manager()
    controller = _Controller("physics_schema", manager=manager)

    controller.activate_collision_world_mode("physics_schema")

    assert manager.calls == [("refresh_physics", "physics_schema")]
    assert controller.calls == []


def test_joint_goal_planning_preserves_measured_start_and_exact_arm_goal():
    class JointState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get_ordered_joint_state(self, names):
            self.ordered_names = list(names)
            return self

        def unsqueeze(self, _axis):
            return self

    captured = {}

    class MotionGen:
        @staticmethod
        def plan_single_js(start, goal, config):
            captured.update(start=start, goal=goal, config=config)
            return "result"

    controller = SimpleNamespace(
        arm_indices=np.array([1, 2]),
        cmd_js_names=["joint_1", "joint_2"],
        robot=SimpleNamespace(
            dof_names=["base", "joint_1", "joint_2", "gripper"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.array([9.0, 1.0, 2.0, 0.03]),
                velocities=np.array([0.0, 0.1, 0.2, 0.0]),
            ),
        ),
        tensor_args=SimpleNamespace(to_device=lambda value: np.asarray(value)),
        _joint_state_derivatives=lambda _state: (
            np.array([0.0, 0.1, 0.2, 0.0]),
            np.zeros(4),
            np.zeros(4),
        ),
        motion_gen=MotionGen(),
        plan_config=SimpleNamespace(clone=lambda: "plan-config"),
        _log_plan_result=lambda context, result, target=None: captured.update(
            log=(context, result, target)
        ),
    )
    method = _load_plan_joint_positions(JointState)

    result = method(controller, np.array([3.0, 4.0]))

    assert result == "result"
    np.testing.assert_allclose(captured["start"].position, [9.0, 1.0, 2.0, 0.03])
    np.testing.assert_allclose(captured["goal"].position, [9.0, 3.0, 4.0, 0.03])
    np.testing.assert_allclose(captured["goal"].velocity, np.zeros(4))
    assert captured["start"].ordered_names == ["joint_1", "joint_2"]
    assert captured["goal"].ordered_names == ["joint_1", "joint_2"]
    assert captured["config"] == "plan-config"
    assert captured["log"][0:2] == ("plan_joint_positions", "result")
    np.testing.assert_allclose(captured["log"][2], [3.0, 4.0])
