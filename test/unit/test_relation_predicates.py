"""Offline contracts for final semantic relation predicates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.relation_predicates import (  # noqa: E402
    RelationPredicateError,
    evaluate_compiled_place_relations,
    evaluate_relation_predicate,
)


def _bounds(minimum, maximum):
    return {"minimum": minimum, "maximum": maximum}


def _predicate(relation):
    value = {
        "relation": relation,
        "geometry_tolerance_m": 0.002,
        "support_gap_tolerance_m": 0.006,
        "minimum_support_contact_n": 0.0,
        "max_unexpected_contact_n": 5.0,
    }
    if relation in {"inside", "insert"}:
        value["container_region"] = {
            "name": "target_interior",
            "center": [0.0, 0.0],
            "inner_size": [0.8, 0.6],
            "interior_support_z": 0.2,
        }
    return value


def test_on_requires_full_footprint_vertical_alignment_and_real_contact():
    target = _bounds([-0.5, -0.5, 0.0], [0.5, 0.5, 0.2])
    object_bounds = _bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.35])

    result = evaluate_relation_predicate(
        _predicate("on"),
        object_bounds,
        target,
        support_contact_force_n=0.4,
        unexpected_object_contact_force_n=0.0,
    )

    assert result["success"] is True
    assert result["checks"] == {
        "horizontal_containment": True,
        "vertical_containment": True,
        "region_inside_target": True,
        "support_gap_ok": True,
        "support_contact": True,
        "unexpected_object_contact": True,
    }


@pytest.mark.parametrize(
    ("object_bounds", "contact_force", "failed_check"),
    [
        (_bounds([-0.1, -0.1, 0.28], [0.1, 0.1, 0.43]), 0.4, "support_gap_ok"),
        (_bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.35]), 0.0, "support_contact"),
        (_bounds([0.45, -0.1, 0.201], [0.65, 0.1, 0.35]), 0.4, "horizontal_containment"),
    ],
)
def test_on_rejects_xy_center_only_false_positives(
    object_bounds, contact_force, failed_check
):
    result = evaluate_relation_predicate(
        _predicate("on"),
        object_bounds,
        _bounds([-0.5, -0.5, 0.0], [0.5, 0.5, 0.2]),
        support_contact_force_n=contact_force,
        unexpected_object_contact_force_n=0.0,
    )

    assert result["success"] is False
    assert result["checks"][failed_check] is False


def test_inside_uses_annotated_interior_volume():
    result = evaluate_relation_predicate(
        _predicate("inside"),
        _bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.45]),
        _bounds([-0.5, -0.4, 0.0], [0.5, 0.4, 0.5]),
        support_contact_force_n=0.2,
        unexpected_object_contact_force_n=0.0,
    )

    assert result["success"] is True
    assert result["measurements"]["predicate_bounds"] == {
        "minimum": [-0.4, -0.3, 0.2],
        "maximum": [0.4, 0.3, 0.5],
    }


def test_insert_is_not_evaluated_as_inside():
    with pytest.raises(RelationPredicateError, match="unsupported strict relation"):
        evaluate_relation_predicate(
            _predicate("insert"),
            _bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.45]),
            _bounds([-0.5, -0.4, 0.0], [0.5, 0.4, 0.5]),
            support_contact_force_n=0.2,
            unexpected_object_contact_force_n=0.0,
        )


@pytest.mark.parametrize(
    ("object_bounds", "failed_check"),
    [
        (_bounds([0.3, -0.1, 0.201], [0.5, 0.1, 0.45]), "horizontal_containment"),
        (_bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.58]), "vertical_containment"),
        (_bounds([-0.1, -0.1, 0.12], [0.1, 0.1, 0.35]), "support_gap_ok"),
    ],
)
def test_inside_rejects_center_only_and_out_of_volume_results(
    object_bounds, failed_check
):
    result = evaluate_relation_predicate(
        _predicate("inside"),
        object_bounds,
        _bounds([-0.5, -0.4, 0.0], [0.5, 0.4, 0.5]),
        support_contact_force_n=0.2,
        unexpected_object_contact_force_n=0.0,
    )

    assert result["success"] is False
    assert result["checks"][failed_check] is False


def test_invalid_container_geometry_fails_instead_of_guessing():
    predicate = _predicate("inside")
    predicate["container_region"].pop("interior_support_z")

    with pytest.raises(RelationPredicateError, match="interior_support_z"):
        evaluate_relation_predicate(
            predicate,
            _bounds([-0.1, -0.1, 0.2], [0.1, 0.1, 0.3]),
            _bounds([-0.5, -0.4, 0.0], [0.5, 0.4, 0.5]),
            support_contact_force_n=0.2,
            unexpected_object_contact_force_n=0.0,
        )


def test_strict_relation_rejects_unexpected_object_contact():
    result = evaluate_relation_predicate(
        _predicate("on"),
        _bounds([-0.1, -0.1, 0.201], [0.1, 0.1, 0.35]),
        _bounds([-0.5, -0.5, 0.0], [0.5, 0.5, 0.2]),
        support_contact_force_n=0.4,
        unexpected_object_contact_force_n=5.1,
    )

    assert result["success"] is False
    assert result["checks"]["unexpected_object_contact"] is False


class _CompletedSkill:
    def __init__(self, config, evaluation=None):
        self.skill_cfg = config
        self._evaluation = evaluation
        self.is_success_called = False

    def evaluate_relation(self):
        if self._evaluation is None:
            raise AssertionError("relation evaluator must not be called")
        return self._evaluation

    def is_success(self):
        self.is_success_called = True
        return True


def test_final_relation_does_not_accept_pick_success_as_semantic_success():
    pick = _CompletedSkill({"name": "pick", "objects": ["cup"]})

    results = evaluate_compiled_place_relations(
        [
            {
                "skill": pick,
                "skill_name": "pick",
                "objects": ["cup"],
                "terminal_success": True,
            }
        ]
    )

    assert results == []
    assert pick.is_success_called is False


def test_final_relation_accepts_only_compiler_owned_place_predicate():
    legacy_place = _CompletedSkill(
        {
            "name": "place",
            "objects": ["cup", "tray"],
            "semantic_relation": "inside",
            "success_mode": "xybbox",
        },
        {"relation": "inside", "success": True},
    )
    compiled_place = _CompletedSkill(
        {
            "name": "place",
            "objects": ["cup", "tray"],
            "agent_subtask_id": "cup_transfer",
            "semantic_relation": "inside",
            "success_mode": "relation_predicate",
            "relation_predicate": _predicate("inside"),
        },
        {
            "relation": "inside",
            "success": True,
            "checks": {"support_contact": True},
            "measurements": {"support_contact_force_n": 0.2},
            "thresholds": {"minimum_support_contact_n": 0.0},
        },
    )

    results = evaluate_compiled_place_relations(
        [
            {
                "skill": legacy_place,
                "skill_name": "place",
                "objects": ["cup", "tray"],
                "terminal_success": True,
            },
            {
                "skill": compiled_place,
                "skill_name": "place",
                "objects": ["cup", "tray"],
                "terminal_success": True,
            },
        ]
    )

    assert len(results) == 1
    assert results[0]["skill"] == "place"
    assert results[0]["subtask_id"] == "cup_transfer"
    assert results[0]["relation"] == "inside"
    assert results[0]["success"] is True
    assert legacy_place.is_success_called is False
