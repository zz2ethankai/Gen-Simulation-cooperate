"""Regression tests for per-Skill controller collision-world switching."""

import ast
from pathlib import Path
from types import SimpleNamespace

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
