"""Regression coverage for legacy nested skill DAG compilation."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.skill_dag_compiler import (  # noqa: E402
    compile_skill_dag_configs,
    generated_skill_id,
)


LEGACY_BALL_TASK = (
    ROOT
    / "workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/"
    / "omniobject3d-ball.yaml"
)


def test_representative_legacy_yaml_gets_stable_compiler_id():
    document = yaml.safe_load(LEGACY_BALL_TASK.read_text(encoding="utf-8"))
    task_cfg = document["tasks"][0]

    first = compile_skill_dag_configs(task_cfg)
    second = compile_skill_dag_configs(task_cfg)

    assert len(first) == 1
    assert first[0].skill_id == "legacy:franka:left:p0:s0:k0:pick"
    assert first[0].depends_on == ()
    assert first[0].generated_id is True
    assert first[0].skill_cfg["id"] == first[0].skill_id
    assert [item.skill_id for item in first] == [item.skill_id for item in second]
    # Compilation is in-memory; the source YAML's legacy shape remains intact.
    assert "id" not in task_cfg["skills"][0]["franka"][0]["left"][0]


def test_legacy_phase_and_sequence_order_is_a_cycle_free_barrier():
    task_cfg = {
        "skills": [
            {
                "robot": [
                    {"left": [{"name": "pick"}, {"name": "place"}]},
                    {"left": [{"name": "home"}]},
                ]
            },
            {"robot": [{"left": [{"name": "observe_hold"}]}]},
        ]
    }

    compiled = compile_skill_dag_configs(task_cfg)
    assert [item.depends_on for item in compiled] == [
        (),
        ("legacy:robot:left:p0:s0:k0:pick",),
        ("legacy:robot:left:p0:s0:k1:place",),
        ("legacy:robot:left:p0:s1:k0:home",),
    ]


def test_explicit_ids_and_dependencies_are_preserved():
    task_cfg = {
        "skills": [
            {
                "robot": [
                    {
                        "left": [
                            {"id": "pick_node", "name": "pick", "depends_on": []},
                            {"id": "place_node", "name": "place", "depends_on": ["pick_node"]},
                        ]
                    }
                ]
            }
        ]
    }

    compiled = compile_skill_dag_configs(task_cfg)
    assert [(item.skill_id, item.depends_on, item.generated_id) for item in compiled] == [
        ("pick_node", (), False),
        ("place_node", ("pick_node",), False),
    ]


def test_duplicate_ids_and_invalid_dependencies_fail_before_runtime_construction():
    duplicate = {
        "skills": [
            {"robot": [{"left": [{"id": "same", "name": "pick"}, {"id": "same", "name": "place"}]}]}
        ]
    }
    with pytest.raises(ValueError, match="Duplicate skill id.*same"):
        compile_skill_dag_configs(duplicate)

    malformed_depends = {
        "skills": [{"robot": [{"left": [{"name": "pick", "depends_on": "not-a-list"}]}]}]
    }
    with pytest.raises(TypeError, match="depends_on must be a list"):
        compile_skill_dag_configs(malformed_depends)

    unknown_depends = {
        "skills": [{"robot": [{"left": [{"id": "pick", "name": "pick", "depends_on": ["missing"]}]}]}]
    }
    with pytest.raises(ValueError, match="depends on unknown skill 'missing'"):
        compile_skill_dag_configs(unknown_depends)


def test_generated_id_components_are_encoded_without_collisions():
    assert generated_skill_id(
        robot_name="robot/a",
        controller_name="left:right",
        phase_index=1,
        sequence_index=2,
        skill_index=3,
        skill_name="Pick",
    ) == "legacy:robot%2Fa:left%3Aright:p1:s2:k3:pick"
