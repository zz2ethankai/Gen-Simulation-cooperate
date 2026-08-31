"""Compile nested SimBox skill YAML into stable DAG node metadata.

The original task format predates DAG node IDs.  It stores skills in the
following nested shape::

    phase -> robot -> sequence -> controller -> [skill configs]

The runtime still uses that shape, but the scheduler and diagnostics need a
stable node identity.  This module is deliberately free of Isaac/CuRobo
imports so task parsing and compatibility tests can exercise the compiler in
an ordinary Python process.

Explicit ``id`` and ``depends_on`` values remain authoritative.  A missing
``id`` gets a readable ``legacy:`` ID containing every source-location field
(``robot``, ``controller``, ``phase``, ``sequence``, ``skill``) and the skill
name.  For those generated nodes only, the compiler reconstructs the old
phase/sequence barriers.  Candidate implicit edges are discarded when they
would create a cycle; an explicit cycle is still reported by the workflow's
normal topological sort.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


_GENERATED_ID_PREFIX = "legacy"


def _id_component(value: Any) -> str:
    """Encode one ID component without allowing delimiters to collide."""

    # Keep common YAML names readable while escaping the separators used by
    # the generated format.  ``quote`` is deterministic across processes and
    # does not depend on object identity or hash randomization.
    encoded = quote(str(value).strip(), safe="-._~")
    return encoded or "_"


def generated_skill_id(
    *,
    robot_name: Any,
    controller_name: Any,
    phase_index: int,
    sequence_index: int,
    skill_index: int,
    skill_name: Any,
) -> str:
    """Return the stable ID used for a legacy skill without an explicit ID."""

    return (
        f"{_GENERATED_ID_PREFIX}:{_id_component(robot_name)}:"
        f"{_id_component(controller_name)}:p{int(phase_index)}:"
        f"s{int(sequence_index)}:k{int(skill_index)}:"
        f"{_id_component(str(skill_name).strip().lower() or 'unnamed')}"
    )


@dataclass(frozen=True)
class CompiledSkillConfig:
    """One flattened skill config ready for typed runtime construction."""

    robot_name: str
    controller_name: str
    phase_index: int
    sequence_index: int
    skill_index: int
    skill_id: str
    depends_on: tuple[str, ...]
    generated_id: bool
    # This is an in-memory copy.  The source task/YAML is never rewritten.
    skill_cfg: dict[str, Any]


def _source_entries(task_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the historical nested skill shape in source order."""

    entries: list[dict[str, Any]] = []
    skills = task_cfg.get("skills", [])
    if skills is None:
        return entries
    if not isinstance(skills, list):
        raise TypeError("task skills must be a list")

    for phase_index, phase_cfg in enumerate(skills):
        if not isinstance(phase_cfg, Mapping):
            raise TypeError(f"skill phase {phase_index} must be a mapping")
        for robot_name, robot_sequences in phase_cfg.items():
            if not isinstance(robot_sequences, list):
                raise TypeError(
                    f"skill phase {phase_index} robot {robot_name!r} sequences must be a list"
                )
            for sequence_index, sequence_cfg in enumerate(robot_sequences):
                if not isinstance(sequence_cfg, Mapping):
                    raise TypeError(
                        f"skill phase {phase_index} robot {robot_name!r} "
                        f"sequence {sequence_index} must be a mapping"
                    )
                for controller_name, controller_skills in sequence_cfg.items():
                    if not isinstance(controller_skills, list):
                        raise TypeError(
                            f"skill phase {phase_index} robot {robot_name!r} "
                            f"sequence {sequence_index} controller {controller_name!r} "
                            "skills must be a list"
                        )
                    for skill_index, skill_cfg in enumerate(controller_skills):
                        if not isinstance(skill_cfg, Mapping):
                            raise TypeError(
                                f"skill config at phase={phase_index}, sequence={sequence_index}, "
                                f"skill={skill_index} must be a mapping"
                            )
                        entries.append(
                            {
                                "robot_name": str(robot_name),
                                "controller_name": str(controller_name),
                                "phase_index": phase_index,
                                "sequence_index": sequence_index,
                                "skill_index": skill_index,
                                "skill_cfg": skill_cfg,
                            }
                        )
    return entries


def _explicit_dependencies(skill_id: str, skill_cfg: Mapping[str, Any]) -> list[str]:
    depends_on = skill_cfg.get("depends_on", [])
    if depends_on is None:
        return []
    if not isinstance(depends_on, list):
        raise TypeError(f"Skill '{skill_id}' depends_on must be a list")
    # Keep list order and repeated references for compatibility with the
    # existing scheduler.  Implicit edges are appended only when absent.
    return [str(dep_id) for dep_id in depends_on]


def _entry_order(entry: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(entry["sequence_index"]),
        int(entry["skill_index"]),
        int(entry["source_index"]),
    )


def _tail_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one terminal entry per robot/controller in stable order."""

    tails: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry["robot_name"]), str(entry["controller_name"]))
        previous = tails.get(key)
        if previous is None or _entry_order(previous) < _entry_order(entry):
            tails[key] = entry
    return sorted(tails.values(), key=lambda item: int(item["source_index"]))


def _implicit_candidates(
    current: dict[str, Any], entries: list[dict[str, Any]]
) -> list[str]:
    """Return legacy ordering predecessors for one generated node.

    The historical executor ran each arm list in order, waited for all arms
    in a sequence before entering the next sequence, and waited for the
    previous phase before entering a new phase.  Reconstruct exactly those
    barriers without linking the two arms within one sequence to each other.
    """

    phase = int(current["phase_index"])
    sequence = int(current["sequence_index"])
    skill_index = int(current["skill_index"])
    robot = str(current["robot_name"])
    controller = str(current["controller_name"])
    current_source_index = int(current["source_index"])
    candidates: list[dict[str, Any]] = []

    # One arm's skill list is sequential, including across sequence entries
    # when the same controller appears in both entries.
    same_arm_prior = [
        entry
        for entry in entries
        if int(entry["source_index"]) < current_source_index
        and int(entry["phase_index"]) == phase
        and str(entry["robot_name"]) == robot
        and str(entry["controller_name"]) == controller
    ]
    if same_arm_prior:
        candidates.append(max(same_arm_prior, key=_entry_order))

    # A new sequence begins only after every arm in the preceding sequence
    # (or the nearest preceding sequence when a YAML sequence is empty) ends.
    if skill_index == 0:
        prior_same_robot = [
            entry
            for entry in entries
            if int(entry["source_index"]) < current_source_index
            and int(entry["phase_index"]) == phase
            and str(entry["robot_name"]) == robot
            and int(entry["sequence_index"]) < sequence
        ]
        if prior_same_robot:
            nearest_sequence = max(int(entry["sequence_index"]) for entry in prior_same_robot)
            candidates.extend(
                entry
                for entry in _tail_entries(
                    [
                        entry
                        for entry in prior_same_robot
                        if int(entry["sequence_index"]) == nearest_sequence
                    ]
                )
                if entry not in candidates
            )

    # A new phase is a barrier after the nearest non-empty prior phase.  The
    # barrier includes all robots/controllers that participated in that phase.
    if sequence == 0 and skill_index == 0:
        prior_phases = [
            entry
            for entry in entries
            if int(entry["source_index"]) < current_source_index
            and int(entry["phase_index"]) < phase
        ]
        if prior_phases:
            nearest_phase = max(int(entry["phase_index"]) for entry in prior_phases)
            candidates.extend(
                entry
                for entry in _tail_entries(
                    [
                        entry
                        for entry in prior_phases
                        if int(entry["phase_index"]) == nearest_phase
                    ]
                )
                if entry not in candidates
            )

    # Source order is stable and makes the generated dependency list stable
    # even when the input mapping happens to contain several controllers.
    candidates.sort(key=lambda item: int(item["source_index"]))
    return [str(entry["skill_id"]) for entry in candidates]


def _would_create_cycle(
    dependencies: Mapping[str, list[str]], source_id: str, target_id: str
) -> bool:
    """Return whether adding ``source_id -> target_id`` closes a cycle."""

    if source_id == target_id:
        return True
    # ``dependencies[node]`` points from node to its prerequisites.  A new
    # prerequisite ``source_id`` is cyclic when it already (transitively)
    # depends on ``target_id``.
    pending = [source_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for dependency in dependencies.get(current, []):
            if dependency == target_id:
                return True
            if dependency not in visited:
                pending.append(dependency)
    return False


def compile_skill_dag_configs(task_cfg: Mapping[str, Any]) -> list[CompiledSkillConfig]:
    """Compile nested skill configs into stable IDs and dependency metadata.

    The result is in source order.  No input mapping is modified.  Duplicate
    IDs and malformed ``depends_on`` values fail before any runtime Skill is
    constructed, so a bad config cannot partially initialize an episode.
    """

    source_entries = _source_entries(task_cfg)
    known_ids: set[str] = set()

    # Pass one assigns every ID before resolving dependencies.  This permits
    # explicit dependencies to refer to a generated ID later in source order.
    for source_index, entry in enumerate(source_entries):
        entry["source_index"] = source_index
        skill_cfg = entry["skill_cfg"]
        if "id" in skill_cfg:
            skill_id = str(skill_cfg["id"])
            generated_id = False
        else:
            skill_id = generated_skill_id(
                robot_name=entry["robot_name"],
                controller_name=entry["controller_name"],
                phase_index=entry["phase_index"],
                sequence_index=entry["sequence_index"],
                skill_index=entry["skill_index"],
                skill_name=skill_cfg.get("name", ""),
            )
            generated_id = True
        if skill_id in known_ids:
            raise ValueError(f"Duplicate skill id in DAG config: {skill_id}")
        known_ids.add(skill_id)
        entry["skill_id"] = skill_id
        entry["generated_id"] = generated_id
        entry["depends_on"] = _explicit_dependencies(skill_id, skill_cfg)

    dependencies = {
        str(entry["skill_id"]): list(entry["depends_on"])
        for entry in source_entries
    }

    # Pass two appends inferred edges only to generated nodes.  If an explicit
    # dependency points backwards, skip the conflicting inferred edge rather
    # than changing user-authored semantics or manufacturing a cycle.
    for entry in source_entries:
        skill_id = str(entry["skill_id"])
        if not bool(entry["generated_id"]):
            continue
        for dependency_id in _implicit_candidates(entry, source_entries):
            if dependency_id in dependencies[skill_id]:
                continue
            if _would_create_cycle(dependencies, dependency_id, skill_id):
                continue
            dependencies[skill_id].append(dependency_id)

    for entry in source_entries:
        skill_id = str(entry["skill_id"])
        unknown = [dep_id for dep_id in dependencies[skill_id] if dep_id not in known_ids]
        if unknown:
            raise ValueError(
                f"Skill '{skill_id}' depends on unknown skill '{unknown[0]}'"
            )

    compiled: list[CompiledSkillConfig] = []
    for entry in source_entries:
        skill_id = str(entry["skill_id"])
        depends_on = tuple(dependencies[skill_id])
        # Keep parser-specific mapping behavior (for example AttrDict's
        # attribute access) while ensuring the source config remains read-only
        # from the runtime compiler's point of view.
        skill_cfg = deepcopy(entry["skill_cfg"])
        # The in-memory copy is what runtime Skills and their debug artifacts
        # receive.  YAML remains unchanged, while all node-facing logs see
        # the same compiler-generated identity.
        skill_cfg["id"] = skill_id
        skill_cfg["depends_on"] = list(depends_on)
        compiled.append(
            CompiledSkillConfig(
                robot_name=str(entry["robot_name"]),
                controller_name=str(entry["controller_name"]),
                phase_index=int(entry["phase_index"]),
                sequence_index=int(entry["sequence_index"]),
                skill_index=int(entry["skill_index"]),
                skill_id=skill_id,
                depends_on=depends_on,
                generated_id=bool(entry["generated_id"]),
                skill_cfg=skill_cfg,
            )
        )
    return compiled


__all__ = ["CompiledSkillConfig", "compile_skill_dag_configs", "generated_skill_id"]
