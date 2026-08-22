"""Simulator-independent planning configuration contracts.

The public SimBox planner is deliberately small: structured manipulator
planning is backed by the Physics-schema world, while a Skill may explicitly
choose the controller's execution-only ``dummy_forward`` interface for direct
joint actions.  This module is kept free of Isaac/CuRobo imports so task
compilation and configuration tests can run on an ordinary Python install.

Older task files are still common in downloaded datasets.  Their planning
knobs are accepted as inert data and produce one ``FutureWarning`` per
``(config_path, field)``.  They must not select a second runtime world.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
import warnings
from typing import Any


PHYSICS_SCHEMA_MODE = "physics_schema"
PASSTHROUGH_MODE = "passthrough"
# Execution-only Skill mode.  This is deliberately not a collision-world
# selector: direct commands do not create or activate a planner world.
DIRECT_EXECUTION_MODE = "direct_execution"

PHYSICS_SCHEMA_SKILLS = {"pick", "place"}
VALIDATION_ONLY_SKILLS = {"pick_plan_probe"}
PHYSICS_SCHEMA_ONLY_SKILLS = PHYSICS_SCHEMA_SKILLS | VALIDATION_ONLY_SKILLS

# These Skills do not select a MotionPlanner world or batch capability.  They
# may run while the Physics world remains initialized, but they must never
# trigger planner/world switching.  Keep this list explicit: unknown Skill
# names remain operation Skills and therefore use the canonical Physics schema.
NON_MANIPULATION_SKILLS = {
    "navigate",
    "wait",
    "observe_hold",
    "scan",
    "track",
}


class DeprecatedPlanningParameterWarning(FutureWarning):
    """A deprecated planning parameter was supplied but has no runtime effect."""


_DEPRECATED_WARNING_KEYS: set[tuple[str, str]] = set()

# ``ignore_substring`` is intentionally listed by itself: it can occur under
# a robot, a Skill, or an old collision-world block, but warning de-duplication
# is by the field name requested by the configuration contract.
DEPRECATED_PLANNING_FIELDS = frozenset(
    {
        "ignore_substring",
        "test_mode",
        "use_batch",
        "reference_prim_path",
        "mode",
        "exact_exclusions",
        "collision_world_mode",
    }
)


def _normalise_config_path(config_path: Any | None) -> str:
    if config_path is None or str(config_path).strip() == "":
        return "<unknown>"
    try:
        return str(Path(config_path).expanduser().resolve())
    except (TypeError, ValueError, OSError):
        return str(config_path)


def reset_deprecated_planning_warnings() -> None:
    """Clear the process-local warning registry.

    This is primarily useful to test runners that parse the same temporary
    path more than once.  Normal application code should leave the registry
    intact so repeated task resets stay quiet.
    """

    _DEPRECATED_WARNING_KEYS.clear()


def warn_deprecated_planning_parameter(
    field: str,
    *,
    config_path: Any | None = None,
    message: str | None = None,
    stacklevel: int = 2,
) -> bool:
    """Warn once for one deprecated field and return whether it was sent."""

    field_name = str(field).strip()
    path = _normalise_config_path(config_path)
    key = (path, field_name)
    if key in _DEPRECATED_WARNING_KEYS:
        return False
    _DEPRECATED_WARNING_KEYS.add(key)
    detail = message or (
        f"planning parameter {field_name!r} in {path!r} is deprecated and ignored; "
        "use the Physics-schema MotionPlanner contract"
    )
    warnings.warn(
        detail,
        DeprecatedPlanningParameterWarning,
        stacklevel=stacklevel,
    )
    return True


def _task_config_path(task_cfg: Mapping[str, Any] | None, config_path: Any | None) -> Any:
    if config_path is not None:
        return config_path
    if task_cfg is None:
        return None
    metadata = task_cfg.get("metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("source_yaml", "config_path", "task_config_path"):
            if metadata.get(key):
                return metadata[key]
    for key in ("_config_path", "config_path", "task_cfg_path"):
        if task_cfg.get(key):
            return task_cfg[key]
    return None


def _walk_mappings(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield every mapping with its immediate key for deprecation scanning."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            yield key_text, child
            if isinstance(child, Mapping):
                yield from _walk_mappings(child)
            elif isinstance(child, list):
                yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def warn_for_deprecated_planning_parameters(
    task_cfg: Mapping[str, Any], *, config_path: Any | None = None
) -> tuple[str, ...]:
    """Scan a task config and emit deprecations once per path/field.

    The function intentionally does not rewrite the input.  Old values remain
    available for diagnostics, while all runtime consumers use the canonical
    helpers in this module.
    """

    path = _task_config_path(task_cfg, config_path)
    found: set[str] = set()
    for key, _mapping in _walk_mappings(task_cfg):
        # ``mode`` is valid for Skill payloads (e.g. heuristic home); its
        # deprecated meaning is only the task-level collision-world selector,
        # handled by the explicit planning block checks below.
        if key in DEPRECATED_PLANNING_FIELDS and key != "mode":
            found.add(key)
    # A task-level old mode is often represented as ``planning.collision_world
    # .mode`` and is found by the walk above.  The direct checks make malformed
    # OmegaConf containers and scalar values deterministic as well.
    planning = task_cfg.get("planning", {})
    if isinstance(planning, Mapping):
        world = planning.get("collision_world", {})
        if isinstance(world, Mapping):
            if "mode" in world:
                found.add("mode")
            if "exact_exclusions" in world:
                found.add("exact_exclusions")
    for field in sorted(found):
        warn_deprecated_planning_parameter(field, config_path=path, stacklevel=3)
    return tuple(sorted(found))


def _skill_entries(task_cfg: Mapping[str, Any]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    """Yield ``(robot, arm/controller, skill_cfg)`` from task DAG YAML."""

    for cfg_skill_dict in task_cfg.get("skills", []) or []:
        if not isinstance(cfg_skill_dict, Mapping):
            continue
        for robot_name, robot_skill_list in cfg_skill_dict.items():
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, Mapping):
                    continue
                for arm_name, arm_skill_list in lr_skill_dict.items():
                    if not isinstance(arm_skill_list, list):
                        continue
                    for skill_cfg in arm_skill_list:
                        if isinstance(skill_cfg, Mapping):
                            yield str(robot_name), str(arm_name), skill_cfg


def _skill_name(skill_cfg: Any) -> str:
    if isinstance(skill_cfg, Mapping):
        return str(skill_cfg.get("name", "")).strip().lower()
    return ""


def is_passthrough_skill(skill_cfg_or_name: Any) -> bool:
    name = (
        _skill_name(skill_cfg_or_name)
        if isinstance(skill_cfg_or_name, Mapping)
        else str(skill_cfg_or_name).strip().lower()
    )
    return name in NON_MANIPULATION_SKILLS


def _canonical_task_mode(requested_mode: str | None, *, config_path: Any | None = None) -> str:
    requested = PHYSICS_SCHEMA_MODE if requested_mode is None else str(requested_mode).strip().lower()
    if requested and requested != PHYSICS_SCHEMA_MODE:
        # Deprecated mode selectors are inert.  Warning (instead of silently
        # selecting a second world) is the compatibility boundary promised to
        # old task files.
        warn_deprecated_planning_parameter(
            "mode",
            config_path=config_path,
            message=(
                f"planning collision-world mode {requested!r} is deprecated and ignored; "
                "Physics schema is the only supported planning world"
            ),
            stacklevel=3,
        )
    return PHYSICS_SCHEMA_MODE


def resolve_skill_collision_world_mode(
    skill_name: str, requested_mode: str | None = None, *, config_path: Any | None = None
) -> str:
    """Resolve planner-world metadata; direct commands bypass it at execution.

    ``dummy_forward`` is a command-level execution interface rather than a
    skill-name allowlist or a second collision-world mode.  A Skill can use it
    without changing this resolver.
    """

    if is_passthrough_skill(skill_name):
        return PASSTHROUGH_MODE
    return _canonical_task_mode(requested_mode, config_path=config_path)


_EXCLUSION_PATTERN_CHARS = frozenset("*?[]{}^$()|+")


def _validate_entity_name(name: str) -> bool:
    """Return whether a value is one exact task-entity name.

    Entity names are resolved to their unique collider during compilation.
    Keep this boundary independent of USD: paths, globs, regex-like patterns,
    and whitespace are not entity names and must never reach the resolver.
    """

    return (
        bool(name)
        and name not in {".", ".."}
        and not name.startswith("/")
        and "/" not in name
        and "\\" not in name
        and not any(char.isspace() for char in name)
        and not any(char in _EXCLUSION_PATTERN_CHARS for char in name)
    )


def validate_planning_exclusions(values: Any | None) -> list[str]:
    """Validate exact task-entity names used for planning exclusions.

    The YAML contract is deliberately ``list[str]``.  Each string names one
    task entity; compilation resolves that name to its unique collider.  No
    mapping, Prim path, reason field, substring, glob, or regex form is
    accepted here.
    """

    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("planning_exclusions must be a YAML list of exact entity names")

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("planning_exclusions entries must be exact entity-name strings")
        name = value.strip()
        if value != name or not _validate_entity_name(name):
            raise ValueError(
                "planning exclusion must be one exact task entity name, not a "
                f"path, substring, or glob: {value!r}"
            )
        if name in seen:
            raise ValueError(f"duplicate planning exclusion: {name}")
        seen.add(name)
        result.append(name)
    return result


def canonicalize_planning_config(
    task_cfg: Mapping[str, Any], *, config_path: Any | None = None
) -> dict[str, Any]:
    """Return a copy with a canonical Physics-schema planning block."""

    result = deepcopy(dict(task_cfg))
    warn_for_deprecated_planning_parameters(result, config_path=config_path)
    planning_value = result.get("planning", {})
    planning = deepcopy(dict(planning_value)) if isinstance(planning_value, Mapping) else {}
    world_value = planning.get("collision_world", {})
    world = deepcopy(dict(world_value)) if isinstance(world_value, Mapping) else {}
    path = _task_config_path(result, config_path)

    _canonical_task_mode(world.get("mode"), config_path=path)
    world.pop("mode", None)
    # ``exact_exclusions`` was the old name and is intentionally ignored.  Do
    # not merge it into the new contract: an old substring/asset assumption
    # must not silently become a Physics exclusion.
    if "exact_exclusions" in world:
        warn_deprecated_planning_parameter("exact_exclusions", config_path=path, stacklevel=3)
        world.pop("exact_exclusions", None)

    # ``neglect_collision_names`` belongs to the simulator's collision-group
    # setup and deliberately does not select a planner exclusion.  Physics
    # planning exclusions must be supplied through the canonical typed field
    # above; importing the old substring semantics here would make planner
    # behavior depend on an untyped legacy task key.
    exclusions = planning.get("planning_exclusions", world.get("planning_exclusions", []))
    validated = validate_planning_exclusions(exclusions)
    # Keep one canonical location and pass entity names to the compiler.  The
    # compiler, not this simulator-independent boundary, resolves each name
    # to its unique collider.
    planning["planning_exclusions"] = validated
    world["mode"] = PHYSICS_SCHEMA_MODE
    world.pop("planning_exclusions", None)
    planning["collision_world"] = world
    result["planning"] = planning
    return result


def derive_batch_capabilities(task_cfg: Mapping[str, Any]) -> dict[tuple[str, str], bool]:
    """Derive candidate-batch capability from the Skill DAG.

    Robot-level ``use_batch`` flags are deprecated planner knobs and are ignored.
    A controller can host a batch planner when its DAG contains a Physics
    manipulation Skill; passthrough nodes do not grant or remove that
    capability.  The tuple key is ``(robot_name, arm/controller_name)``.
    """

    capabilities: dict[tuple[str, str], bool] = {}
    for robot_name, arm_name, skill_cfg in _skill_entries(task_cfg):
        if is_passthrough_skill(skill_cfg):
            capabilities.setdefault((robot_name, arm_name), False)
            continue
        # All operation Skills use the MotionPlanner API.  Candidate-bearing
        # Physics Skills are explicitly known; unknown operation Skills are
        # left false until their adapter declares batch support.
        if _skill_name(skill_cfg) in PHYSICS_SCHEMA_ONLY_SKILLS:
            capabilities[(robot_name, arm_name)] = True
        else:
            capabilities.setdefault((robot_name, arm_name), False)
    return capabilities


def derive_batch_capability(
    task_cfg: Mapping[str, Any],
    robot_name: str | None = None,
    arm_name: str | None = None,
) -> bool | dict[tuple[str, str], bool]:
    """Return one DAG-derived capability, or the complete capability map."""

    values = derive_batch_capabilities(task_cfg)
    if robot_name is None:
        return values
    if arm_name is None:
        return any(value for (robot, _arm), value in values.items() if robot == str(robot_name))
    return bool(values.get((str(robot_name), str(arm_name)), False))


def validate_planning_contract(
    task_cfg: Mapping[str, Any],
    collision_world_mode: str | None = None,
    *,
    config_path: Any | None = None,
) -> None:
    """Reject config that bypasses the canonical Physics-schema planner."""

    path = _task_config_path(task_cfg, config_path)
    if collision_world_mode is not None:
        # A supplied old selector is an inert deprecated field, not a second
        # execution path.  Canonicalize it here as well as at the parser
        # boundary so direct callers receive the same warning and result.
        _canonical_task_mode(collision_world_mode, config_path=path)

    planning = task_cfg.get("planning", {})
    if isinstance(planning, Mapping):
        world = planning.get("collision_world", {})
        if isinstance(world, Mapping):
            if "exact_exclusions" in world:
                warn_deprecated_planning_parameter("exact_exclusions", config_path=path, stacklevel=3)
            exclusions = planning.get(
                "planning_exclusions", world.get("planning_exclusions", [])
            )
        else:
            exclusions = planning.get("planning_exclusions", [])
        validate_planning_exclusions(exclusions)

    # Both arms may be active in one YAML phase.  Workflow DAG compilation
    # adds deterministic dependency edges so those operation nodes execute
    # sequentially; config validation must not reject the existing dual-arm
    # task inventory merely because both arms occur in the same phase.
    for stage_index, cfg_skill_dict in enumerate(task_cfg.get("skills", []) or []):
        if not isinstance(cfg_skill_dict, Mapping):
            continue
        for robot_name, robot_skill_list in cfg_skill_dict.items():
            if not isinstance(robot_skill_list, list):
                continue
            for phase_index, lr_skill_dict in enumerate(robot_skill_list):
                if not isinstance(lr_skill_dict, Mapping):
                    continue
                for arm in ("left", "right"):
                    arm_skills = lr_skill_dict.get(arm, [])
                    if not isinstance(arm_skills, list):
                        continue
                    for skill_cfg in arm_skills:
                        name = _skill_name(skill_cfg)
                        if is_passthrough_skill(skill_cfg):
                            continue
                        if name == "pick_plan_probe":
                            metadata = task_cfg.get("metadata", {})
                            if not isinstance(metadata, Mapping) or not isinstance(
                                metadata.get("workspace_probe"), Mapping
                            ):
                                raise ValueError(
                                    "validation-only Skill 'pick_plan_probe' requires "
                                    "metadata.workspace_probe"
                                )
                            if len(skill_cfg.get("objects", []) or []) != 1:
                                raise ValueError(
                                    "validation-only Skill 'pick_plan_probe' requires "
                                    "exactly 1 object identity"
                                )
                            continue
                        if name in PHYSICS_SCHEMA_SKILLS:
                            expected_count = 1 if name == "pick" else 2
                            object_count = len(skill_cfg.get("objects", []) or [])
                            if object_count != expected_count:
                                raise ValueError(
                                    f"physics_schema {name} requires exactly "
                                    f"{expected_count} object identities, got {object_count}"
                                )


def resolve_collision_world_mode(
    task_cfg: Mapping[str, Any], requested_mode: str | None = None
) -> tuple[str, str]:
    """Resolve a task to Physics schema; old mode selectors are inert."""

    planning = task_cfg.get("planning", {})
    configured = None
    if isinstance(planning, Mapping):
        world = planning.get("collision_world", {})
        if isinstance(world, Mapping):
            configured = world.get("mode")
    mode = _canonical_task_mode(
        requested_mode if requested_mode is not None else configured,
        config_path=_task_config_path(task_cfg, None),
    )
    validate_planning_contract(task_cfg, mode)
    if requested_mode is not None and str(requested_mode).strip().lower() != mode:
        return mode, "deprecated mode ignored; Physics schema is the only planning world"
    return mode, "Physics schema is the only planning world"
