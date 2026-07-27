"""Regression tests for continuous Pick candidate path validation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_PICK_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "skills"
    / "pick.py"
)


def _load_method(path, class_name, method_name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method_namespace = dict(namespace or {})
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, path, "exec"), method_namespace)
    return method_namespace[method_name]


class _RecordingController:
    def __init__(self, outcomes):
        self.reference_prim_path = "/World/robot/base"
        self.outcomes = iter(outcomes)
        self.plan_calls = []
        self.world_updates = []

    def update_specific(self, ignore_substring, reference_prim_path):
        self.world_updates.append((list(ignore_substring), reference_prim_path))

    def test_forward_from_joint_positions(self, translation, orientation, start_arm_positions=None):
        self.plan_calls.append(
            {
                "translation": np.asarray(translation).copy(),
                "orientation": np.asarray(orientation).copy(),
                "start_arm_positions": None
                if start_arm_positions is None
                else np.asarray(start_arm_positions).copy(),
            }
        )
        success, endpoint = next(self.outcomes)
        return success, endpoint, SimpleNamespace()


class _Pick:
    _validate_complete_candidate_path = _load_method(
        _PICK_PATH,
        "Pick",
        "_validate_complete_candidate_path",
    )
    _find_complete_candidate_path = _load_method(
        _PICK_PATH,
        "Pick",
        "_find_complete_candidate_path",
    )

    def __init__(self, outcomes):
        self.controller = _RecordingController(outcomes)


def test_complete_path_uses_each_previous_segment_endpoint():
    pick = _Pick(
        [
            (True, np.array([1.0, 2.0])),
            (True, np.array([3.0, 4.0])),
            (False, None),
        ]
    )

    result = pick._validate_complete_candidate_path(
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.2, 0.3, 0.4]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.2, 0.3, 0.45]),
        ["base-obstacle"],
        ["base-obstacle", "pick-object"],
        validate_postgrasp=True,
    )

    assert result == {
        "pregrasp_success": True,
        "grasp_success": True,
        "postgrasp_success": False,
    }
    assert len(pick.controller.plan_calls) == 3
    assert pick.controller.plan_calls[0]["start_arm_positions"] is None
    np.testing.assert_allclose(pick.controller.plan_calls[1]["start_arm_positions"], [1.0, 2.0])
    np.testing.assert_allclose(pick.controller.plan_calls[2]["start_arm_positions"], [3.0, 4.0])
    assert pick.controller.world_updates == [
        (["base-obstacle"], "/World/robot/base"),
        (["base-obstacle", "pick-object"], "/World/robot/base"),
    ]


def test_failed_pregrasp_stops_candidate_validation():
    pick = _Pick([(False, None)])

    result = pick._validate_complete_candidate_path(
        np.zeros(3),
        np.zeros(4),
        np.ones(3),
        np.ones(4),
        np.full(3, 2.0),
        [],
        ["pick-object"],
        validate_postgrasp=True,
    )

    assert result == {
        "pregrasp_success": False,
        "grasp_success": False,
        "postgrasp_success": False,
    }
    assert len(pick.controller.plan_calls) == 1
    assert len(pick.controller.world_updates) == 1


def test_candidate_search_continues_after_postgrasp_failure():
    pick = _Pick([])
    validations = iter(
        [
            {"pregrasp_success": True, "grasp_success": True, "postgrasp_success": False},
            {"pregrasp_success": True, "grasp_success": True, "postgrasp_success": True},
        ]
    )
    attempted_pregrasps = []

    def validate(pregrasp_translation, *args, **kwargs):
        attempted_pregrasps.append(float(pregrasp_translation[0]))
        return next(validations)

    pick._validate_complete_candidate_path = validate
    candidate_debug = {0: {}, 1: {}}
    translations = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))

    selected = pick._find_complete_candidate_path(
        [0, 1],
        translations,
        orientations,
        translations,
        orientations,
        translations,
        [],
        ["pick-object"],
        validate_postgrasp=True,
        candidate_debug_by_index=candidate_debug,
    )

    assert selected == 1
    assert attempted_pregrasps == [0.0, 1.0]
    assert candidate_debug[0]["postgrasp_success"] is False
    assert candidate_debug[1]["postgrasp_success"] is True
