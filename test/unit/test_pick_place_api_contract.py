"""Static contracts for the direct typed Pick/Place batch boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "workflows" / "simbox" / "core"


def test_runtime_exposes_one_typed_batch_entrypoint():
    source = (CORE / "controllers/curobo/runtime.py").read_text(encoding="utf-8")
    assert "def plan_pose(" in source
    assert "def plan_pose_batch(" in source
    assert "def plan_cspace(" in source
    assert not (CORE / "controllers/curobo/skill_runtime.py").exists()


def test_pick_place_use_native_batch_and_no_single_fallback_lane():
    for name in ("pick", "place"):
        source = (CORE / "skills" / f"{name}.py").read_text(encoding="utf-8")
        assert "plan_pose_batch(" in source
        assert "plan_pose_result(" not in source
        assert "plan_pose_from_path(" not in source
        assert "fallback" not in source.lower()
        assert "CUROBO_BATCH_SIZE" in source


def test_batch_result_is_intersected_by_original_candidate_index():
    pick = (CORE / "skills/pick.py").read_text(encoding="utf-8")
    place = (CORE / "skills/place.py").read_text(encoding="utf-8")
    assert "pre_mask & terminal_mask" in pick
    assert "terminal_hold_mask[:] = pre_mask" in pick
    assert "pre_ok & terminal_ok" in place
