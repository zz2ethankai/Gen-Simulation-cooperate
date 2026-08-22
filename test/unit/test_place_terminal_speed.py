"""Offline regression coverage for Physics-schema Place descent segmentation."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from core.planning.domain_types import BatchPlanResult, PlanResult


ROOT = Path(__file__).resolve().parents[2]
PLACE_PATH = ROOT / "workflows" / "simbox" / "core" / "skills" / "place.py"


def _load_place_methods(*method_names: str):
    tree = ast.parse(PLACE_PATH.read_text(encoding="utf-8"))
    place_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Place"
    )
    methods = [
        node
        for node in place_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    namespace = {
        "np": np,
        "BatchPlanResult": BatchPlanResult,
        "PlanResult": PlanResult,
    }
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, PLACE_PATH, "exec"), namespace)
    return [namespace[name] for name in method_names]


def test_place_specific_terminal_step_wins():
    resolve_step, = _load_place_methods("_resolve_terminal_step")

    assert resolve_step(
        {"terminal_step_m": 0.005, "place_terminal_step_m": 0.01}
    ) == pytest.approx(0.01)


def test_legacy_terminal_step_remains_supported():
    resolve_step, = _load_place_methods("_resolve_terminal_step")

    assert resolve_step({"terminal_step_m": 0.01}) == pytest.approx(0.01)


def test_place_terminal_step_defaults_to_one_centimeter():
    resolve_step, = _load_place_methods("_resolve_terminal_step")

    assert resolve_step({}) == pytest.approx(0.01)


def test_continuous_place_descent_is_enabled_by_default_and_can_fallback():
    use_continuous, = _load_place_methods("_use_continuous_terminal_descent")

    assert use_continuous({}) is True
    assert use_continuous({"place_continuous_descent": False}) is False
    with pytest.raises(ValueError, match="must be a boolean"):
        use_continuous({"place_continuous_descent": "false"})


@pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf"), "bad"])
def test_place_terminal_step_must_be_positive_and_finite(value):
    resolve_step, = _load_place_methods("_resolve_terminal_step")

    with pytest.raises(ValueError, match="positive finite"):
        resolve_step({"place_terminal_step_m": value})


def test_terminal_tolerance_is_independent_and_bounded_by_step():
    resolve_tolerance, = _load_place_methods("_resolve_terminal_tolerance")

    assert resolve_tolerance({}, 0.01) == pytest.approx(0.005)
    assert resolve_tolerance({}, 0.003) == pytest.approx(0.003)
    assert resolve_tolerance(
        {"place_terminal_tolerance_m": 0.003}, 0.01
    ) == pytest.approx(0.003)
    with pytest.raises(ValueError, match="must not exceed"):
        resolve_tolerance({"place_terminal_tolerance_m": 0.02}, 0.01)


@pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf"), "bad"])
def test_place_terminal_tolerance_must_be_positive_and_finite(value):
    resolve_tolerance, = _load_place_methods("_resolve_terminal_tolerance")

    with pytest.raises(ValueError, match="positive finite"):
        resolve_tolerance({"place_terminal_tolerance_m": value}, 0.01)


def test_segmented_fallback_keeps_one_centimeter_motion_bound():
    terminal_samples, = _load_place_methods("_terminal_samples")
    start = np.asarray([0.4, -0.1, 0.19])
    goal = np.asarray([0.4, -0.1, 0.10])

    samples = terminal_samples(start, goal, 0.01)

    assert len(samples) == 9
    np.testing.assert_allclose(samples[-1], goal)
    assert max(
        np.linalg.norm(current - previous)
        for previous, current in zip([start] + samples[:-1], samples)
    ) <= 0.01 + 1e-12


def test_place_candidate_helpers_consume_typed_plan_results():
    candidate_mask, result_paths = _load_place_methods(
        "_candidate_mask", "_result_paths"
    )
    single = PlanResult(success=True, trajectory=[[0.1, 0.2]])
    batch = BatchPlanResult(
        success=[True, False],
        trajectories=[[[0.1, 0.2]], None],
    )

    assert candidate_mask(single, 1) == [True]
    assert candidate_mask(batch, 2) == [True, False]
    assert len(result_paths(single)) == 1
    assert len(result_paths(batch)) == 2
    assert result_paths(batch)[1] is None


def test_place_candidate_helpers_reject_untyped_results():
    candidate_mask, result_paths = _load_place_methods(
        "_candidate_mask", "_result_paths"
    )

    assert candidate_mask(object(), 1) == [False]
    with pytest.raises(TypeError, match="normalized PlanResult"):
        result_paths(object())
