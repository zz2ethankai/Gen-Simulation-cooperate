"""Load the small deterministic policy/configuration surface of the Agent."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from workflows.simbox.core.utils.camera_template import (
    CAMERA_TEMPLATE_DEFAULTS,
    ROBOT_TARGET_OVERHEAD_V1,
)


AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = AGENT_DIR / "config.yaml"


def _finite_vector(values: Sequence[Any], name: str, length: int) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def load_agent_settings(path: Path | None = None) -> dict[str, Any]:
    source = (path or DEFAULT_CONFIG_PATH).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Agent config must be a mapping: {source}")
    return value


def resolve_debug_camera(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the camera shared by Agent screenshots and view/probe tools."""

    debug = settings.get("debug", {})
    if not isinstance(debug, Mapping):
        raise ValueError("Agent config debug must be a mapping")
    resolution = [
        int(value)
        for value in _finite_vector(
            debug.get("topdown_resolution", [1280, 960]),
            "debug.topdown_resolution",
            2,
        )
    ]
    focal_length_mm = float(debug.get("topdown_focal_length_mm", 16.0))
    if min(resolution) <= 0 or not math.isfinite(focal_length_mm) or focal_length_mm <= 0.0:
        raise ValueError("debug camera resolution and focal length must be positive")
    raw_eye = debug.get("topdown_eye")
    raw_target = debug.get("topdown_target")
    if (raw_eye is None) != (raw_target is None):
        raise ValueError("debug.topdown_eye and topdown_target must be set together")
    template = str(debug.get("topdown_template", ROBOT_TARGET_OVERHEAD_V1))
    if template not in CAMERA_TEMPLATE_DEFAULTS:
        raise ValueError(f"unsupported debug camera template: {template}")
    raw_params = debug.get("topdown_template_params", {})
    if not isinstance(raw_params, Mapping):
        raise ValueError("debug.topdown_template_params must be a mapping")
    defaults = CAMERA_TEMPLATE_DEFAULTS[template]
    template_params = {**defaults, **dict(raw_params)}
    unknown = sorted(set(template_params) - set(defaults))
    if unknown:
        raise ValueError(f"unknown debug camera template parameters: {unknown}")
    template_params = {key: float(value) for key, value in template_params.items()}
    return {
        "template": template,
        "template_params": template_params,
        "eye": (
            _finite_vector(raw_eye, "debug.topdown_eye", 3)
            if raw_eye is not None
            else None
        ),
        "target": (
            _finite_vector(raw_target, "debug.topdown_target", 3)
            if raw_target is not None
            else None
        ),
        "resolution": resolution,
        "focal_length_mm": focal_length_mm,
    }


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



SCENE_INGEST_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "output_root": "output/agent_scene_ingest",
    "out_dir_rel": "assets/basic",
    "target_class_task_object": "RigidObject",
    "task_id_source": "scene_name",
    "envmap_lib": "environment/envmaps",
    "max_episode_length": 10000,
    "support_fixture": "central_work_table",
}


def resolve_scene_ingest(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the effective scene-ingest settings merged over defaults."""
    section = dict(settings or {}).get("scene_ingest", {})
    if not isinstance(section, Mapping):
        raise ValueError("Agent config scene_ingest must be a mapping")
    return merge_mappings(SCENE_INGEST_DEFAULTS, section)
