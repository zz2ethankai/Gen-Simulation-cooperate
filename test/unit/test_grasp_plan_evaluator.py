"""Contracts for the current typed Pick/Place candidate pipeline."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "workflows" / "simbox" / "core"


def _source(name: str) -> str:
    return (CORE / "skills" / f"{name}.py").read_text(encoding="utf-8")


def _method(source: str, name: str):
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_pick_and_place_use_only_native_batch_candidate_planning():
    for name in ("pick", "place"):
        source = _source(name)
        assert "plan_pose_batch(" in source
        assert "CUROBO_BATCH_SIZE" in source
        assert "plan_pose_result(" not in source
        assert "plan_pose_from_path(" not in source
        assert "fallback" not in source.lower()


def test_candidate_sampling_has_no_fixed_twenty_candidate_cap():
    method = _method(_source("pick"), "sample_ee_pose")
    assert method.args.args[-1].arg == "max_length"
    default = method.args.defaults[-1]
    assert isinstance(default, ast.Constant) and default.value is None


def test_pre_and_terminal_masks_are_intersected_in_original_index_space():
    pick = _source("pick")
    place = _source("place")
    assert "valid_pre = np.flatnonzero(pre_mask" in pick
    assert "pre_mask & terminal_mask" in pick
    assert "valid_pre = np.flatnonzero(" in place
    assert "pre_ok & terminal_ok" in place
    assert "range(0, count, CUROBO_BATCH_SIZE)" in pick
    assert "range(0, count, CUROBO_BATCH_SIZE)" in place


def test_controller_runtime_owns_typed_planner_runtime_directly():
    source = (CORE / "controllers/curobo/runtime.py").read_text(encoding="utf-8")
    assert "class MotionPlannerRuntime(PlannerRuntime):" in source
    assert "self.planner_runtime" not in source
    assert "batch_single_fallback" not in source
    assert not (CORE / "controllers/curobo/skill_runtime.py").exists()
    assert not (CORE / "planning/grasp_plan_evaluator.py").exists()
