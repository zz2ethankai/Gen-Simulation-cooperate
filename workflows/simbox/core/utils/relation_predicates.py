"""Deterministic geometric and contact predicates for final object relations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class RelationPredicateError(ValueError):
    """A relation predicate is incomplete or cannot be evaluated."""


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise RelationPredicateError(f"{label} must contain {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RelationPredicateError(f"{label} must contain {length} numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise RelationPredicateError(f"{label} must contain only finite numbers")
    return result


def _bounds(value: Any, label: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise RelationPredicateError(f"{label} must be a mapping")
    minimum = _finite_vector(value.get("minimum"), 3, f"{label}.minimum")
    maximum = _finite_vector(value.get("maximum"), 3, f"{label}.maximum")
    if any(lower > upper for lower, upper in zip(minimum, maximum)):
        raise RelationPredicateError(f"{label} minimum exceeds maximum")
    return minimum, maximum


def _nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RelationPredicateError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise RelationPredicateError(f"{label} must be finite and non-negative")
    return result


def _contains(
    outer_minimum: Sequence[float],
    outer_maximum: Sequence[float],
    inner_minimum: Sequence[float],
    inner_maximum: Sequence[float],
    tolerance: float,
) -> bool:
    return all(
        inner_lower >= outer_lower - tolerance
        and inner_upper <= outer_upper + tolerance
        for outer_lower, outer_upper, inner_lower, inner_upper in zip(
            outer_minimum,
            outer_maximum,
            inner_minimum,
            inner_maximum,
        )
    )


def evaluate_relation_predicate(
    predicate: Mapping[str, Any],
    object_bounds: Mapping[str, Any],
    target_bounds: Mapping[str, Any],
    support_contact_force_n: float,
    unexpected_object_contact_force_n: float,
) -> dict[str, Any]:
    """Evaluate an Agent-compiled relation against final world-space evidence."""

    relation = str(predicate.get("relation") or "")
    if relation not in {"on", "inside"}:
        raise RelationPredicateError(f"unsupported strict relation: {relation!r}")
    geometry_tolerance = _nonnegative(
        predicate.get("geometry_tolerance_m"),
        "geometry_tolerance_m",
    )
    support_gap_tolerance = _nonnegative(
        predicate.get("support_gap_tolerance_m"),
        "support_gap_tolerance_m",
    )
    minimum_contact = _nonnegative(
        predicate.get("minimum_support_contact_n"),
        "minimum_support_contact_n",
    )
    maximum_unexpected_contact = _nonnegative(
        predicate.get("max_unexpected_contact_n"),
        "max_unexpected_contact_n",
    )
    contact_force = _nonnegative(support_contact_force_n, "support_contact_force_n")
    unexpected_contact_force = _nonnegative(
        unexpected_object_contact_force_n,
        "unexpected_object_contact_force_n",
    )

    object_minimum, object_maximum = _bounds(object_bounds, "object_bounds")
    target_minimum, target_maximum = _bounds(target_bounds, "target_bounds")
    support_contact = contact_force > minimum_contact

    if relation == "on":
        predicate_minimum = target_minimum
        predicate_maximum = target_maximum
        horizontal_containment = _contains(
            target_minimum[:2],
            target_maximum[:2],
            object_minimum[:2],
            object_maximum[:2],
            geometry_tolerance,
        )
        vertical_containment = True
        region_inside_target = True
        support_z = target_maximum[2]
    else:
        region = predicate.get("container_region")
        if not isinstance(region, Mapping):
            raise RelationPredicateError(f"{relation} requires container_region")
        center = _finite_vector(region.get("center"), 2, "container_region.center")
        inner_size = _finite_vector(
            region.get("inner_size"),
            2,
            "container_region.inner_size",
        )
        if any(size <= 0.0 for size in inner_size):
            raise RelationPredicateError("container_region.inner_size must be positive")
        try:
            support_z = float(region.get("interior_support_z"))
        except (TypeError, ValueError) as exc:
            raise RelationPredicateError(
                "container_region.interior_support_z must be finite"
            ) from exc
        if not math.isfinite(support_z):
            raise RelationPredicateError("container_region.interior_support_z must be finite")
        predicate_minimum = (
            center[0] - inner_size[0] * 0.5,
            center[1] - inner_size[1] * 0.5,
            support_z,
        )
        predicate_maximum = (
            center[0] + inner_size[0] * 0.5,
            center[1] + inner_size[1] * 0.5,
            target_maximum[2],
        )
        if predicate_minimum[2] > predicate_maximum[2]:
            raise RelationPredicateError(
                "container interior support is above the target world bounds"
            )
        horizontal_containment = _contains(
            predicate_minimum[:2],
            predicate_maximum[:2],
            object_minimum[:2],
            object_maximum[:2],
            geometry_tolerance,
        )
        vertical_containment = _contains(
            predicate_minimum[2:],
            predicate_maximum[2:],
            object_minimum[2:],
            object_maximum[2:],
            geometry_tolerance,
        )
        region_inside_target = _contains(
            target_minimum[:2],
            target_maximum[:2],
            predicate_minimum[:2],
            predicate_maximum[:2],
            geometry_tolerance,
        )

    support_gap = object_minimum[2] - support_z
    support_gap_ok = abs(support_gap) <= support_gap_tolerance
    checks = {
        "horizontal_containment": horizontal_containment,
        "vertical_containment": vertical_containment,
        "region_inside_target": region_inside_target,
        "support_gap_ok": support_gap_ok,
        "support_contact": support_contact,
        "unexpected_object_contact": (
            unexpected_contact_force <= maximum_unexpected_contact
        ),
    }
    return {
        "relation": relation,
        "success": all(checks.values()),
        "checks": checks,
        "measurements": {
            "object_bounds": {
                "minimum": list(object_minimum),
                "maximum": list(object_maximum),
            },
            "target_bounds": {
                "minimum": list(target_minimum),
                "maximum": list(target_maximum),
            },
            "predicate_bounds": {
                "minimum": list(predicate_minimum),
                "maximum": list(predicate_maximum),
            },
            "support_gap_m": support_gap,
            "support_contact_force_n": contact_force,
            "unexpected_object_contact_force_n": unexpected_contact_force,
        },
        "thresholds": {
            "geometry_tolerance_m": geometry_tolerance,
            "support_gap_tolerance_m": support_gap_tolerance,
            "minimum_support_contact_n": minimum_contact,
            "max_unexpected_contact_n": maximum_unexpected_contact,
        },
    }


def evaluate_compiled_place_relations(
    completed_skills: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Evaluate only strict Place relations backed by compiler-owned predicates."""

    results = []
    for item in completed_skills:
        if str(item.get("skill_name", "")).lower() != "place":
            continue
        skill = item.get("skill")
        config = getattr(skill, "skill_cfg", None)
        if not isinstance(config, Mapping):
            continue
        predicate = config.get("relation_predicate")
        relation = str(config.get("semantic_relation") or "")
        if (
            str(config.get("name", "")).lower() != "place"
            or config.get("success_mode") != "relation_predicate"
            or not isinstance(predicate, Mapping)
            or relation not in {"on", "inside", "insert"}
            or str(predicate.get("relation") or "") != relation
        ):
            continue
        error = ""
        evaluation: Mapping[str, Any] = {}
        try:
            evaluator = getattr(skill, "evaluate_relation", None)
            if not callable(evaluator):
                raise RelationPredicateError("compiled Place has no relation evaluator")
            evaluated = evaluator()
            if not isinstance(evaluated, Mapping):
                raise RelationPredicateError("Place relation evaluator must return a mapping")
            if str(evaluated.get("relation") or "") != relation:
                raise RelationPredicateError(
                    "Place relation evaluator returned a different relation"
                )
            if not isinstance(evaluated.get("success"), bool):
                raise RelationPredicateError("Place relation evaluator success must be boolean")
            evaluation = evaluated
            success = evaluated["success"]
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "predicate_id": f"relation_{len(results):02d}",
                "subtask_id": str(config.get("agent_subtask_id") or ""),
                "skill": "place",
                "objects": list(item.get("objects") or []),
                "relation": relation,
                "terminal_success": bool(item.get("terminal_success")),
                "success": success,
                "checks": dict(evaluation.get("checks") or {}),
                "measurements": dict(evaluation.get("measurements") or {}),
                "thresholds": dict(evaluation.get("thresholds") or {}),
                "error": error,
            }
        )
    return results


__all__ = [
    "RelationPredicateError",
    "evaluate_compiled_place_relations",
    "evaluate_relation_predicate",
]
