"""Offline tests for the attached-object carry-home adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.planning.domain_types import PlanResult


ROOT = Path(__file__).resolve().parents[2]
_SKILL_PATH = (
    ROOT / "workflows" / "simbox" / "core" / "skills" / "heuristic_skill.py"
)


class _Command:
    def __init__(self, phase, target_position, target_orientation, **kwargs):
        self.phase = phase
        self.target_position = np.asarray(target_position)
        self.target_orientation = np.asarray(target_orientation)
        self.__dict__.update(kwargs)


def _load_method():
    tree = ast.parse(_SKILL_PATH.read_text(encoding="utf-8"))
    skill_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Heuristic_Skill"
    )
    method_node = next(
        node
        for node in skill_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_physics_schema_generate_manip_cmds"
    )
    namespace = {
        "np": np,
        "MotionPhase": SimpleNamespace(CARRY_HOME="carry_home"),
        "MotionPhaseCommand": _Command,
    }
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(module, _SKILL_PATH, "exec"), namespace)
    return namespace["_physics_schema_generate_manip_cmds"]


def _skill(*, gripper_state=-1.0, mode="home", results=None):
    manager = SimpleNamespace(calls=[])
    manager.assert_attached_owner = lambda *args: manager.calls.append(args)
    if results is None:
        results = [True]
    plan_results = [
        PlanResult(
            success=bool(success),
            trajectory=f"joint-path-{index}",
        )
        for index, success in enumerate(results)
    ]
    planned_goals = []

    def plan_cspace(goal, *, context=None):
        assert context == "carry_home"
        planned_goals.append(np.asarray(goal).copy())
        return plan_results[len(planned_goals) - 1]

    controller = SimpleNamespace(
        name="robot",
        lr_name="left",
        collision_scene_manager=manager,
        num_plan_failed=4,
        plan_cspace=plan_cspace,
        forward_kinematic=lambda goal: (
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    contact_view = object()
    task = SimpleNamespace(
        pickcontact_views={"robot": {"left": {"apple": contact_view}}}
    )
    value = SimpleNamespace(
        manip_list=["old"],
        failure_reason="old",
        error_message="old",
        mode=mode,
        _physics_schema_active_object="apple",
        controller=controller,
        robot=SimpleNamespace(
            name="robot",
            get_joint_positions=lambda: np.asarray([0.0, 0.0, 0.0]),
        ),
        _gripper_state=gripper_state,
        _joint_indices=np.asarray([0, 1, 2]),
        _joint_home=np.asarray([1.0, 2.0, 3.0]),
        task=task,
        skill_cfg={"physics_t_eps": 0.02},
    )
    return value, manager, contact_view, planned_goals


def test_attached_home_builds_cached_physics_phase_and_preserves_attachment():
    skill, manager, contact_view, planned_goals = _skill()

    _load_method()(skill)

    assert manager.calls == [("apple", "robot", "left")]
    assert skill.controller.num_plan_failed == 0
    assert skill._pickcontact_view is contact_view
    assert len(skill.manip_list) == 1
    command = skill.manip_list[0]
    assert command.phase == "carry_home"
    assert command.active_object == "apple"
    assert command.allow_target_finger_contact is True
    assert command.gripper_action == "close_gripper"
    assert command.params["preplanned_joint_path"].positions == "joint-path-0"
    assert command.params["home_progress"] == 1.0
    np.testing.assert_allclose(planned_goals, [[1.0, 2.0, 3.0]])
    assert command.completion_tolerance == {
        "position_m": 0.02,
        "orientation_rad": 0.05,
    }


def test_attached_home_rejects_open_gripper_configuration():
    skill, _, _, _ = _skill(gripper_state=1.0)

    with pytest.raises(RuntimeError, match="keep the gripper closed"):
        _load_method()(skill)


def test_attached_home_falls_back_to_nearest_collision_free_tuck_posture():
    skill, _, _, planned_goals = _skill(results=[False, True])

    _load_method()(skill)

    np.testing.assert_allclose(planned_goals[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(planned_goals[1], [0.75, 1.5, 2.25])
    np.testing.assert_allclose(skill._goal_joints, [0.75, 1.5, 2.25])
    assert skill.manip_list[0].params["home_progress"] == 0.75
    assert skill.manip_list[0].params["preplanned_joint_path"].positions == "joint-path-1"


def test_attached_home_reports_failure_after_all_tuck_candidates_fail():
    skill, _, _, planned_goals = _skill(results=[False] * 5)

    _load_method()(skill)

    assert len(planned_goals) == 5
    assert skill.manip_list == []
    assert skill.controller.num_plan_failed == 5
    assert skill.failure_reason == "NO_COLLISION_FREE_CARRY_HOME_PLAN"
    assert "attempted_home_progress" in skill.error_message
