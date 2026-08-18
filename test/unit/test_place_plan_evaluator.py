"""Offline contract tests for chained Place planning."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

plan_utils_module = types.ModuleType("core.utils.plan_utils")
plan_utils_module.select_index_by_priority_single = (
    lambda result: int(np.flatnonzero(result.success)[0])
)
plan_utils_module.select_index_by_priority_dual = (
    lambda pre, final: int(np.flatnonzero(pre.success & final.success)[0])
)
sys.modules["core.utils.plan_utils"] = plan_utils_module

from core.planning.place_plan_evaluator import evaluate_place_paths  # noqa: E402


class _Result:
    def __init__(self, success):
        self.success = np.asarray(success, dtype=bool)

    def get_paths(self):
        return [SimpleNamespace(position=np.zeros((2, 2))) for _ in self.success]


class _Controller:
    use_batch = True
    name = "robot"
    lr_name = "left"

    def __init__(self, pre_success, descent_success):
        self.pre_result = _Result(pre_success)
        self.descent_result = _Result(descent_success)

    def test_batch_forward(self, _positions, _orientations):
        return self.pre_result

    def test_batch_forward_from_paths(self, _positions, _orientations, _paths):
        return self.descent_result


class _Manager:
    def __init__(self):
        self.entered = False

    @contextmanager
    def placement_descent_planning_world(self, *identity):
        assert identity == ("object", "support", "robot", "left")
        self.entered = True
        yield ("/World/support/collider",)


def _poses():
    positions = np.zeros((3, 3), dtype=float)
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    return positions, orientations


def test_place_requires_one_candidate_to_pass_transit_and_descent():
    controller = _Controller([True, True, False], [False, True, True])
    manager = _Manager()
    pre_positions, orientations = _poses()
    descent_positions = pre_positions.copy()
    descent_positions[:, 2] = -0.1

    result = evaluate_place_paths(
        controller,
        manager,
        "object",
        "support",
        pre_positions,
        orientations,
        descent_positions,
        orientations,
        test_mode="forward",
    )

    assert result.feasible is True
    assert result.selected_index == 1
    assert result.preplace_success_count == 2
    assert result.descent_success_count == 2
    assert result.joint_success_count == 1
    assert manager.entered is True


def test_place_rejects_preplace_only_success():
    controller = _Controller([True, False, False], [False, True, False])
    manager = _Manager()
    pre_positions, orientations = _poses()
    descent_positions = pre_positions.copy()
    descent_positions[:, 2] = -0.1

    result = evaluate_place_paths(
        controller,
        manager,
        "object",
        "support",
        pre_positions,
        orientations,
        descent_positions,
        orientations,
        test_mode="forward",
    )

    assert result.feasible is False
    assert result.selected_index is None
    assert result.failure_code == "NO_COLLISION_FREE_PLACE_DESCENT_PLAN"
    assert manager.entered is True
