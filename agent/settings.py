"""Load the small deterministic policy/configuration surface of the Agent."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = AGENT_DIR / "config.yaml"


def load_agent_settings(path: Path | None = None) -> dict[str, Any]:
    source = (path or DEFAULT_CONFIG_PATH).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Agent config must be a mapping: {source}")
    return value


def merge_mappings(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_mappings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_data_generation(
    requested: bool | None,
    settings: Mapping[str, Any],
) -> bool:
    """Resolve an optional user override against the global run default."""

    if requested is not None:
        return requested
    generation = settings.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("Agent config generation must be a mapping")
    return bool(generation.get("enabled", True))
