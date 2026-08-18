"""Deterministic admission rules for executable semantic relations."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


INSERT_RELATION_NOT_ADMITTED = "RELATION_INSERT_NOT_ADMITTED"
RELATION_NOT_ADMITTED = "RELATION_NOT_ADMITTED"
RANDOMIZED_WORLD_CONTAINER_REGION = "CONTAINER_REGION_WORLD_FRAME_RANDOMIZED"


class RelationAdmissionError(ValueError):
    """A semantic relation cannot be evaluated by the current runtime contract."""

    def __init__(self, failure_code: str, message: str):
        super().__init__(f"{failure_code}: {message}")
        self.failure_code = failure_code


def _entity(region: Mapping[str, Any]) -> str:
    return str(region.get("object") or region.get("A") or "").strip()


def _fixed_numeric_range(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    endpoints: list[tuple[float, ...]] = []
    for endpoint in value:
        raw_values = endpoint if isinstance(endpoint, (list, tuple)) else [endpoint]
        try:
            numeric = tuple(float(item) for item in raw_values)
        except (TypeError, ValueError):
            return False
        if not numeric or not all(math.isfinite(item) for item in numeric):
            return False
        endpoints.append(numeric)
    return endpoints[0] == endpoints[1]


def _randomized_target_regions(
    target: str,
    placement_regions: Iterable[Mapping[str, Any]],
) -> list[str]:
    randomized_fields: list[str] = []
    for index, region in enumerate(placement_regions):
        if _entity(region) != target:
            continue
        random_config = region.get("random_config")
        if not isinstance(random_config, Mapping):
            continue
        for field in ("pos_range", "yaw_rotation"):
            value = random_config.get(field)
            if value is not None and not _fixed_numeric_range(value):
                randomized_fields.append(f"regions[{index}].random_config.{field}")
    return randomized_fields


def validate_relation_admission(
    relation: str,
    target: str,
    container_regions: Iterable[Mapping[str, Any]],
    placement_regions: Iterable[Mapping[str, Any]] = (),
) -> None:
    """Reject relations whose terminal predicate cannot be proven deterministically."""

    if relation == "insert":
        raise RelationAdmissionError(
            INSERT_RELATION_NOT_ADMITTED,
            "relation='insert' requires an explicit insertion axis, minimum depth, "
            "terminal orientation contract, and insertion-specific runtime evaluator",
        )
    if relation not in {"on", "inside"}:
        raise RelationAdmissionError(
            RELATION_NOT_ADMITTED,
            f"relation={relation!r} has no compiler-owned terminal predicate in v1; "
            "only 'on' and 'inside' are admitted",
        )
    if relation == "on":
        return

    container_region = next(
        (
            region
            for region in container_regions
            if str(region.get("object") or region.get("target") or "") == target
            and bool(region.get("can_receive_objects", True))
        ),
        None,
    )
    if container_region is None:
        return

    randomized_fields = _randomized_target_regions(target, placement_regions)
    if randomized_fields:
        raise RelationAdmissionError(
            RANDOMIZED_WORLD_CONTAINER_REGION,
            f"inside target {target!r} has world-space container geometry but a non-fixed "
            f"runtime pose in {randomized_fields}; target-local container transforms are not admitted in v1",
        )


__all__ = [
    "INSERT_RELATION_NOT_ADMITTED",
    "RELATION_NOT_ADMITTED",
    "RANDOMIZED_WORLD_CONTAINER_REGION",
    "RelationAdmissionError",
    "validate_relation_admission",
]
