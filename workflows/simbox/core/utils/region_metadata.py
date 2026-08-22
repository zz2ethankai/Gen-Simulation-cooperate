"""Helpers for joining authored and executable region metadata."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def canonical_region_object_name(value: Any) -> str | None:
    """Return the task-entity spelling used by executable regions.

    Converted scene YAML keeps the authored ``source_regions`` names (for
    example ``apple__0__id9008``) while the executable ``regions`` use the
    sanitized task name (``apple_0_id9008``).  Region metadata is keyed by
    object name, so compare the two spellings without changing the names in
    either config.
    """

    if value is None:
        return None
    return re.sub(r"_+", "_", str(value).strip())


def merge_source_region_sampling_metadata(cfg: Any) -> None:
    """Merge missing authored sampling flags into executable regions.

    ``regions`` owns placement coordinates and random ranges.  The richer
    ``source_regions`` entries own metadata such as ``sampling.keep_upright``.
    Only missing sampling keys are copied, and values are deep-copied so a
    reset cannot mutate the authored source metadata.
    """

    active_regions = cfg.get("regions", []) or []
    source_regions = cfg.get("source_regions", []) or []
    if not active_regions or not source_regions:
        return

    source_by_object: dict[str, Any] = {}
    for source_region in source_regions:
        if not hasattr(source_region, "get"):
            continue
        object_name = source_region.get("object") or source_region.get("A")
        if object_name is None:
            continue
        # Preserve an exact spelling first, then index the sanitized spelling
        # used by converted executable task configs.
        source_by_object.setdefault(str(object_name), source_region)
        canonical_name = canonical_region_object_name(object_name)
        if canonical_name is not None:
            source_by_object.setdefault(canonical_name, source_region)

    for active_region in active_regions:
        if not hasattr(active_region, "get"):
            continue
        object_name = active_region.get("object") or active_region.get("A")
        if object_name is None:
            continue
        source_region = source_by_object.get(str(object_name))
        if source_region is None:
            source_region = source_by_object.get(canonical_region_object_name(object_name))
        if source_region is None:
            continue
        source_sampling = source_region.get("sampling", {}) or {}
        if not hasattr(source_sampling, "items"):
            continue
        active_sampling = active_region.get("sampling")
        if active_sampling is None:
            active_sampling = {}
            active_region["sampling"] = active_sampling
        for key, value in source_sampling.items():
            if key not in active_sampling:
                active_sampling[key] = deepcopy(value)
