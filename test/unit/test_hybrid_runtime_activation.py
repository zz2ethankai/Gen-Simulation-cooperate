"""Runtime Hybrid resolution tests without importing Isaac Sim."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.config_contract import (  # noqa: E402
    PASSTHROUGH_MODE,
    resolve_runtime_skill_collision_world_mode,
)


_WORKFLOW_PATH = ROOT / "workflows" / "simbox_dual_workflow.py"


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
        "resolve_runtime_skill_collision_world_mode": resolve_runtime_skill_collision_world_mode,
        "LOGGER": SimpleNamespace(info=lambda *args, **kwargs: None),
    }
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, _WORKFLOW_PATH, "exec"), namespace)
    return namespace["_activate_skill_collision_world"]


class _Workflow:
    _activate_skill_collision_world = _load_activation_method()

    def __init__(self, attached_entity):
        self.requested_collision_world_mode = "auto"
        self.collision_scene_manager = SimpleNamespace(
            get_attached_entity=lambda _robot, _arm: attached_entity
        )

    @staticmethod
    def _skill_display_name(skill):
        return skill.skill_cfg["name"]


class _Controller:
    def __init__(self):
        self.name = "robot"
        self.lr_name = "left"
        self.activations = []

    def activate_collision_world_mode(self, mode):
        self.activations.append(mode)


def _home_skill():
    return SimpleNamespace(
        collision_world_mode="legacy_stage_scan",
        skill_cfg={"name": "heuristic__skill", "mode": "home"},
        controller=_Controller(),
    )


def test_attached_home_is_promoted_to_physics_runtime_adapter():
    workflow = _Workflow("apple")
    skill = _home_skill()

    mode = workflow._activate_skill_collision_world(skill)

    assert mode == "physics_schema"
    assert skill.controller.activations == ["physics_schema"]
    assert skill._physics_schema_active_object == "apple"
    assert skill.effective_collision_world_mode == "physics_schema"


def test_post_detach_home_returns_to_legacy_fallback():
    workflow = _Workflow(None)
    skill = _home_skill()

    mode = workflow._activate_skill_collision_world(skill)

    assert mode == "legacy_stage_scan"
    assert skill.controller.activations == ["legacy_stage_scan"]
    assert skill._physics_schema_active_object is None
    assert skill.effective_collision_world_mode == "legacy_stage_scan"
