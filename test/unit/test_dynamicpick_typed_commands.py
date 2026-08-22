"""Offline contract tests for the dynamic-pick typed command boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.planning.motion_command import MotionPhase, MotionPhaseCommand


ROOT = Path(__file__).resolve().parents[2]
_SKILL_PATH = ROOT / "workflows" / "simbox" / "core" / "skills" / "dynamicpick.py"


def _load_simple_generator():
    tree = ast.parse(_SKILL_PATH.read_text(encoding="utf-8"))
    skill_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Dynamicpick"
    )
    method_node = next(
        node
        for node in skill_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "simple_generate_manip_cmds"
    )
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, _SKILL_PATH, "exec"), namespace)
    return namespace["simple_generate_manip_cmds"]


def test_simple_generate_delegates_prediction_and_keeps_dynamic_target():
    target = "conveyor_item"
    command = MotionPhaseCommand(
        phase=MotionPhase.TRANSIT_PREGRASP,
        target_position=np.zeros(3),
        target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        active_object=target,
    )
    skill = SimpleNamespace(manip_list=[], meet_pose_o2w=(np.ones(3), np.ones(4)))
    observed = []

    def predict():
        observed.append(skill.meet_pose_o2w)
        skill.manip_list = [command]

    skill.predict_manip_cmds = predict
    result = _load_simple_generator()(skill)

    assert result == [command]
    assert result[0].active_object == target
    assert observed == [skill.meet_pose_o2w]
