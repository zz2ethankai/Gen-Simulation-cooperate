"""Unit tests for the single-world MotionPlanner configuration boundary."""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.config_contract import (  # noqa: E402
    PASSTHROUGH_MODE,
    PHYSICS_SCHEMA_MODE,
    canonicalize_planning_config,
    resolve_collision_world_mode,
    resolve_skill_collision_world_mode,
    validate_planning_contract,
    validate_planning_exclusions,
)


def _task(skill, *, both_arms=False, planning=None):
    phase = {"left": [skill], "right": [skill] if both_arms else []}
    task = {"skills": [{"robot": [phase]}]}
    if planning is not None:
        task["planning"] = planning
    return task


def test_standard_pick_and_place_are_accepted_in_the_unique_world():
    task = {
        "skills": [
            {"robot": [{"left": [{"name": "Pick", "objects": ["a"]}], "right": []}]},
            {"robot": [{"left": [{"name": "Place", "objects": ["a", "support"]}], "right": []}]},
        ]
    }
    validate_planning_contract(task, PHYSICS_SCHEMA_MODE)


@pytest.mark.parametrize("requested", [None, PHYSICS_SCHEMA_MODE])
def test_task_mode_is_always_physics_schema(requested):
    task = _task({"name": "pick", "objects": ["a"]})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        mode, _reason = resolve_collision_world_mode(task, requested)
    assert mode == PHYSICS_SCHEMA_MODE


@pytest.mark.parametrize(
    "name", ["navigate", "wait", "observe_hold", "scan", "track"]
)
def test_non_operation_skills_are_passthrough(name):
    assert resolve_skill_collision_world_mode(name, PHYSICS_SCHEMA_MODE) == PASSTHROUGH_MODE


def test_other_skills_use_physics_schema():
    for name in ("heuristic__skill", "open", "close", "artpreplan"):
        assert resolve_skill_collision_world_mode(name, PHYSICS_SCHEMA_MODE) == PHYSICS_SCHEMA_MODE


def test_unregistered_operation_skills_are_accepted_by_the_physics_contract():
    validate_planning_contract(
        _task({"name": "open", "objects": ["microwave"], "planner_setting": {}}),
        PHYSICS_SCHEMA_MODE,
    )


def test_dual_arm_operation_phases_are_accepted_for_sequential_dag_compile():
    # Existing dual-arm YAML keeps both arms in one phase.  The workflow
    # compiler adds deterministic dependency edges and executes those nodes
    # one typed command at a time; validation must not reject the source DAG.
    validate_planning_contract(
        _task({"name": "pick", "objects": ["a"]}, both_arms=True),
        PHYSICS_SCHEMA_MODE,
    )
    validate_planning_contract(
        _task({"name": "observe_hold", "hold_steps": 300}, both_arms=True),
        PHYSICS_SCHEMA_MODE,
    )


def test_workspace_probe_contract_remains_sequential_and_typed():
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
    validate_planning_contract(task, PHYSICS_SCHEMA_MODE)


def test_workspace_probe_requires_metadata():
    with pytest.raises(ValueError, match="metadata.workspace_probe"):
        validate_planning_contract(
            _task({"name": "pick_plan_probe", "objects": ["a"]}),
            PHYSICS_SCHEMA_MODE,
        )


def test_planning_exclusions_are_exact_task_entity_names():
    assert validate_planning_exclusions(["table", "sink_0_id1"]) == [
        "table",
        "sink_0_id1",
    ]
    with pytest.raises(ValueError, match="YAML list"):
        validate_planning_exclusions({"table": "reason"})
    with pytest.raises(ValueError, match="exact task entity name"):
        validate_planning_exclusions([""])
    with pytest.raises(ValueError, match="exact task entity name"):
        validate_planning_exclusions(["/World/table"])
    with pytest.raises(ValueError, match="exact task entity name"):
        validate_planning_exclusions(["table/collider"])
    with pytest.raises(ValueError, match="exact task entity name"):
        validate_planning_exclusions(["table*"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_planning_exclusions(["table", "table"])


def test_old_world_fields_are_rejected_instead_of_ignored():
    task = {
        "planning": {
            "collision_world": {
                "mode": "deprecated_world",
                "exact_exclusions": [],
            }
        },
        "skills": [
            {
                "robot": [
                    {
                        "left": [
                            {
                                "name": "pick",
                                "objects": ["a"],
                                "ignore_substring": ["table"],
                                "test_mode": "ik",
                            }
                        ],
                        "right": [],
                    }
                ]
            }
        ],
    }
    with pytest.raises(ValueError, match="only supported"):
        canonicalize_planning_config(task)


def test_legacy_neglect_names_remain_inert_for_planner_contract():
    task = {"neglect_collision_names": ["table"]}

    canonical = canonicalize_planning_config(task)

    assert canonical["planning"]["planning_exclusions"] == []
    # The simulator still consumes this field while creating PhysX support
    # groups; only the planner exclusion contract must remain typed.
    assert canonical["neglect_collision_names"] == ["table"]


def test_canonical_exact_planning_exclusions_are_preserved_over_legacy_names():
    task = {
        "planning": {"planning_exclusions": ["table"]},
        "neglect_collision_names": ["legacy_substring"],
    }

    canonical = canonicalize_planning_config(task)

    assert canonical["planning"]["planning_exclusions"] == ["table"]
    assert canonical["neglect_collision_names"] == ["legacy_substring"]
def test_all_task_yaml_documents_validate_the_physics_contract():
    """Validate every task under the repository's ``tasks[0]`` wrapper."""

    task_root = ROOT / "workflows" / "simbox" / "core" / "configs" / "tasks"
    task_files = []
    task_count = 0
    for config_path in sorted(task_root.rglob("*.yaml")):
        document = yaml.safe_load(config_path.read_text())
        if not isinstance(document, dict) or "tasks" not in document:
            continue
        tasks = document["tasks"]
        assert isinstance(tasks, list), f"{config_path}: tasks must be a list"
        task_files.append(config_path)
        for task_index, task in enumerate(tasks):
            assert isinstance(task, dict), f"{config_path}: tasks[{task_index}] must be a mapping"
            canonical = canonicalize_planning_config(task, config_path=config_path)
            validate_planning_contract(canonical, config_path=config_path)
            task_count += 1

    assert len(task_files) == 1149
    assert task_count == 1149
