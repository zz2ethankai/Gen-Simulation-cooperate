"""Unit tests for the Physics-schema Skill migration boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.config_contract import (  # noqa: E402
    HYBRID_MODE,
    LEGACY_STAGE_SCAN_MODE,
    PASSTHROUGH_MODE,
    PHYSICS_SCHEMA_MODE,
    resolve_collision_world_mode,
    resolve_skill_collision_world_mode,
    task_uses_physics_schema,
    validate_planning_contract,
)


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


def test_auto_enables_physics_schema_for_supported_task():
    task = {
        "skills": [
            {"robot": [{"left": [{"name": "Pick", "objects": ["a"]}], "right": []}]},
            {
                "robot": [
                    {
                        "left": [
                            {"name": "Place", "objects": ["a", "support"]}
                        ],
                        "right": [],
                    }
                ]
            },
        ]
    }

    mode, reason = resolve_collision_world_mode(task, "auto")

    assert mode == "physics_schema"
    assert "support physics_schema" in reason


def test_auto_uses_hybrid_for_physics_pick_place_and_legacy_home():
    task = {
        "skills": [
            {
                "robot": [
                    {
                        "left": [
                            {"name": "Pick", "objects": ["a"]},
                            {"name": "Place", "objects": ["a", "support"]},
                            {"name": "heuristic__skill", "mode": "home"},
                        ],
                        "right": [],
                    }
                ]
            }
        ]
    }

    mode, _ = resolve_collision_world_mode(task, "auto")

    assert mode == HYBRID_MODE
    assert resolve_skill_collision_world_mode("pick", "auto") == PHYSICS_SCHEMA_MODE
    assert (
        resolve_skill_collision_world_mode("heuristic__skill", "auto")
        == LEGACY_STAGE_SCAN_MODE
    )


def test_auto_keeps_physics_schema_with_observe_hold_passthrough():
    task = {
        "skills": [
            {
                "robot": [
                    {
                        "left": [
                            {"name": "Pick", "objects": ["a"]},
                            {"name": "Place", "objects": ["a", "support"]},
                            {"name": "Observe_Hold", "hold_steps": 10},
                        ],
                        "right": [],
                    }
                ]
            }
        ]
    }

    mode, _ = resolve_collision_world_mode(task, "auto")

    assert mode == PHYSICS_SCHEMA_MODE
    assert (
        resolve_skill_collision_world_mode("observe_hold", "auto")
        == PASSTHROUGH_MODE
    )


def test_auto_resolves_hybrid_for_unmigrated_manipulation_skill():
    task = {
        "skills": [
            {
                "robot": [
                    {
                        "left": [
                            {"name": "Pick", "objects": ["a"]},
                            {"name": "Place", "objects": ["a", "support"]},
                            {"name": "Close", "objects": ["drawer"]},
                        ],
                        "right": [],
                    }
                ]
            }
        ]
    }

    mode, reason = resolve_collision_world_mode(task, "auto")

    assert mode == HYBRID_MODE
    assert "close" in reason
    assert task_uses_physics_schema(mode)
    assert (
        resolve_skill_collision_world_mode("close", "auto")
        == LEGACY_STAGE_SCAN_MODE
    )


def test_auto_uses_legacy_for_unmigrated_only_task():
    task = _task({"name": "heuristic__skill", "mode": "home"})

    mode, reason = resolve_collision_world_mode(task, None)

    assert mode == "legacy_stage_scan"
    assert "no Physics-schema manipulation skills" in reason


def test_auto_does_not_enable_for_non_manipulation_only_task():
    mode, reason = resolve_collision_world_mode(
        _task({"name": "Observe_Hold", "hold_steps": 10}),
        "auto",
    )

    assert mode == "legacy_stage_scan"
    assert "no Physics-schema manipulation skills" in reason


@pytest.mark.parametrize(
    "skill",
    [
        {"name": "Close", "objects": ["drawer"]},
        {"name": "heuristic__skill", "mode": "home"},
    ],
)
def test_explicit_physics_schema_remains_strict(skill):
    task = _task(skill)

    with pytest.raises(ValueError, match="not migrated"):
        resolve_collision_world_mode(task, "physics_schema")


def test_explicit_legacy_mode_is_preserved():
    mode, reason = resolve_collision_world_mode(
        _task({"name": "DynamicPick", "objects": ["a"]}),
        "legacy_stage_scan",
    )

    assert mode == "legacy_stage_scan"
    assert reason == "explicit configuration"


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


def test_hybrid_keeps_physics_skill_contract_strict():
    with pytest.raises(ValueError, match="exactly 2 object identities"):
        validate_planning_contract(
            _task({"name": "Place", "objects": ["a"]}), HYBRID_MODE
        )
    with pytest.raises(ValueError, match="test_mode=forward"):
        validate_planning_contract(
            _task({"name": "Pick", "objects": ["a"], "test_mode": "ik"}),
            HYBRID_MODE,
        )
