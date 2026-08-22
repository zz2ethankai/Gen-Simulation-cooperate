"""Runtime world binding tests for the single Physics-schema workflow."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.config_contract import (  # noqa: E402
    DIRECT_EXECUTION_MODE,
    PASSTHROUGH_MODE,
    PHYSICS_SCHEMA_MODE,
)


_WORKFLOW_PATH = ROOT / "workflows/simbox_dual_workflow.py"


def _load_activation_method():
    tree = ast.parse(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    workflow_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SimBoxDualWorkFlow"
    )
    method_node = next(
        node
        for node in workflow_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_activate_skill_collision_world"
    )
    namespace = {
        "PASSTHROUGH_MODE": PASSTHROUGH_MODE,
        "PHYSICS_SCHEMA_MODE": PHYSICS_SCHEMA_MODE,
        "DIRECT_EXECUTION_MODE": DIRECT_EXECUTION_MODE,
    }
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, _WORKFLOW_PATH, "exec"), namespace)
    return namespace["_activate_skill_collision_world"]


class _Workflow:
    _activate_skill_collision_world = _load_activation_method()

    def __init__(self, attached_entity):
        self.collision_scene_manager = SimpleNamespace(
            get_attached_entity=lambda _robot, _arm: attached_entity
        )

    @staticmethod
    def _skill_display_name(skill):
        return skill.skill_cfg["name"]


class _Runtime:
    def __init__(self):
        self.name = "robot"
        self.arm_name = "left"


def test_direct_home_does_not_bind_a_physics_attachment():
    workflow = _Workflow("apple")
    skill = SimpleNamespace(
        collision_world_mode=PHYSICS_SCHEMA_MODE,
        skill_cfg={"name": "heuristic__skill", "mode": "home"},
        execution_mode=DIRECT_EXECUTION_MODE,
        skill_runtime=None,
    )

    mode = workflow._activate_skill_collision_world(skill)

    assert mode == DIRECT_EXECUTION_MODE
    assert not hasattr(skill, "_physics_schema_active_object")
    assert skill.effective_collision_world_mode == DIRECT_EXECUTION_MODE


def test_planned_operation_skill_reuses_physics_world_without_switching():
    workflow = _Workflow("apple")
    skill = SimpleNamespace(
        collision_world_mode=PHYSICS_SCHEMA_MODE,
        skill_cfg={"name": "pick"},
        skill_runtime=_Runtime(),
    )

    mode = workflow._activate_skill_collision_world(skill)

    assert mode == PHYSICS_SCHEMA_MODE
    assert skill._physics_schema_active_object == "apple"
    assert skill.effective_collision_world_mode == PHYSICS_SCHEMA_MODE


def test_passthrough_skill_does_not_require_a_controller_or_world_switch():
    workflow = _Workflow(None)
    skill = SimpleNamespace(
        collision_world_mode=PASSTHROUGH_MODE,
        skill_cfg={"name": "observe_hold"},
    )

    assert workflow._activate_skill_collision_world(skill) == PASSTHROUGH_MODE
    assert skill.effective_collision_world_mode == PASSTHROUGH_MODE
