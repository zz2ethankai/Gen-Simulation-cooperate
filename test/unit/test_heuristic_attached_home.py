"""Offline contracts for the direct heuristic Home path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
_SKILL_PATH = ROOT / "workflows" / "simbox" / "core" / "skills" / "heuristic_skill.py"

import sys

sys.path.insert(0, str(ROOT / "workflows" / "simbox"))

from core.planning.motion_command import MotionPhase  # noqa: E402
from core.skills.base_skill import BaseSkill  # noqa: E402


def _skill_class():
    tree = ast.parse(_SKILL_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Heuristic_Skill"
    )


def _method(name):
    method_node = next(
        node
        for node in _skill_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"np": np, "MotionPhase": MotionPhase}
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, _SKILL_PATH, "exec"), namespace)
    return namespace[name]


def test_home_interpolation_emits_direct_commands_and_exact_endpoint():
    skill = SimpleNamespace(
        move_steps=3,
        _gripper_state=-1.0,
        t_eps=0.088,
        joint_command=BaseSkill.joint_command,
    )

    commands = _method("_build_joint_traj")(
        skill,
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 3.0]),
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )

    assert len(commands) == 3
    assert all(command.is_direct for command in commands)
    np.testing.assert_allclose(commands[0].direct_joint_action, [1 / 3, 5 / 3])
    np.testing.assert_allclose(commands[-1].direct_joint_action, [1.0, 3.0])


def test_heuristic_home_contains_no_physics_schema_carry_planner():
    source = _SKILL_PATH.read_text(encoding="utf-8")

    assert "_physics_schema_generate_manip_cmds" not in source
    assert "plan_cspace(candidate" not in source
    assert "NO_COLLISION_FREE_CARRY_HOME_PLAN" not in source
    assert "get_contact" not in source
    assert "direct_joint_action" in source
