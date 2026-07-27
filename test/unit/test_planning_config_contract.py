"""Unit tests for the Physics-schema Skill migration boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.config_contract import validate_planning_contract  # noqa: E402


def _task(skill, *, both_arms=False):
    phase = {"left": [skill], "right": [skill] if both_arms else []}
    return {"skills": [{"robot": [phase]}]}


def test_standard_pick_and_place_are_accepted():
    task = {
        "skills": [
            {"robot": [{"left": [{"name": "Pick", "objects": ["a"]}], "right": []}]},
            {"robot": [{"left": [{"name": "Place", "objects": ["a", "support"]}], "right": []}]},
        ]
    }
    validate_planning_contract(task, "physics_schema")


@pytest.mark.parametrize("name", ["DynamicPick", "ManualPick", "DexPick", "DexPlace", "Open", "Close"])
def test_unmigrated_skills_require_explicit_legacy_mode(name):
    task = _task({"name": name, "objects": ["a"]})
    with pytest.raises(ValueError, match="not migrated"):
        validate_planning_contract(task, "physics_schema")
    validate_planning_contract(task, "legacy_stage_scan")


def test_concurrent_arms_are_rejected():
    with pytest.raises(ValueError, match="UNSUPPORTED_CONCURRENT_MANIPULATION"):
        validate_planning_contract(
            _task({"name": "Pick", "objects": ["a"]}, both_arms=True),
            "physics_schema",
        )


def test_two_arm_observe_hold_is_not_concurrent_manipulation():
    validate_planning_contract(
        _task({"name": "Observe_Hold", "hold_steps": 300}, both_arms=True),
        "physics_schema",
    )


def test_workspace_probe_is_allowed_only_as_sequential_validation_phases():
    probe = {"name": "pick_plan_probe", "objects": ["a"]}
    task = {
        "metadata": {"workspace_probe": {"candidate_id": "candidate_0"}},
        "skills": [
            {
                "robot": [
                    {"left": [probe], "right": []},
                    {"left": [], "right": [probe]},
                ]
            }
        ],
    }
    validate_planning_contract(task, "physics_schema")


def test_workspace_probe_cannot_leak_into_a_normal_task():
    with pytest.raises(ValueError, match="requires metadata.workspace_probe"):
        validate_planning_contract(
            _task({"name": "pick_plan_probe", "objects": ["a"]}),
            "physics_schema",
        )


def test_object_identity_arity_and_ik_only_mode_are_rejected():
    with pytest.raises(ValueError, match="exactly 2 object identities"):
        validate_planning_contract(
            _task({"name": "Place", "objects": ["a"]}), "physics_schema"
        )
    with pytest.raises(ValueError, match="test_mode=forward"):
        validate_planning_contract(
            _task({"name": "Pick", "objects": ["a"], "test_mode": "ik"}),
            "physics_schema",
        )
