"""Run measured layout evolution through the real compile and planning gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.compiler import (
    CompileError,
    compile_task_config,
    generate_workspace_manifest,
    select_task_workspace_candidate,
)
from agent.contracts import ExecutionVariant, SceneCapabilityManifest, TaskPlan
from agent.tools.feedback import failure_code_from_text

from .evolution import CandidateEvaluation, EvolutionSearch
from .planner import SceneLayoutBlocked, SceneLayoutPlanner


@dataclass(frozen=True)
class SceneLayoutSearchResult:
    candidate_id: str
    scene_revision: str
    scene_task_path: str
    scene_arena_path: str
    mutation_path: str
    workspace_paths: dict[str, str]
    workspace_selection_path: str
    workspace_candidate: dict[str, Any]
    search_dir: str
    generations_completed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> "SceneLayoutSearchResult":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"layout search result must be a mapping: {path}")
        return cls(**value)


def run_scene_layout_search(
    plan: TaskPlan,
    execution_variant: ExecutionVariant,
    manifest: SceneCapabilityManifest,
    source_task_path: Path,
    source_arena_path: Path,
    failure_code: str,
    output_dir: Path,
    gpu_ids: Sequence[int],
    conda_env: str,
    settings: Mapping[str, Any],
    *,
    subtask_id: str | None = None,
) -> SceneLayoutSearchResult:
    """Return the first layout robust across all frozen debug seeds."""

    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise SceneLayoutBlocked(
            "LAYOUT_GPU_QUEUES_INVALID",
            "SceneLayout v1 requires four distinct GPU queues",
            {"gpu_ids": list(gpu_ids)},
        )
    subtask = _select_subtask(plan, subtask_id)
    protected = {
        value
        for item in plan.subtasks
        for value in (item.target_object, item.manipulated_object)
        if value and value != subtask.manipulated_object
    }
    planner = SceneLayoutPlanner(
        source_task_path,
        source_arena_path,
        subtask.manipulated_object,
        failure_code,
        execution_variant.profile_id,
        protected_entities=protected,
    )
    output_dir = output_dir.resolve()
    search_dir = output_dir / "search"

    def evaluate(candidate, seed: int, worker_slot: int) -> CandidateEvaluation:
        candidate_dir = (
            output_dir
            / "candidates"
            / candidate.candidate_id
            / f"seed_{seed:03d}"
        )
        validation = planner.validate_candidate(candidate, candidate_dir / "scene")
        if not validation.hard_ok:
            return CandidateEvaluation(
                candidate.candidate_id,
                candidate.generation,
                seed,
                validation.hard_constraints,
                False,
                failure_code=validation.failure_code,
                artifact_refs=validation.artifact_refs,
            )
        assert validation.derived_task_path is not None
        assert validation.derived_arena_path is not None
        assert validation.mutation_path is not None
        derived_task = Path(validation.derived_task_path)
        base_task = candidate_dir / "base_task.yaml"
        workspace_paths: dict[str, str] = {}
        try:
            compile_task_config(
                plan,
                execution_variant,
                manifest,
                base_task,
                settings=settings,
                scene_task_path=derived_task,
            )
            for item in plan.subtasks:
                workspace_path = generate_workspace_manifest(
                    base_task,
                    item.manipulated_object,
                    execution_variant.arm_binding[item.subtask_id],
                    candidate_dir / "subtasks" / item.subtask_id / "workspace",
                    execution_variant.placement_family,
                )
                workspace_paths[item.subtask_id] = str(workspace_path)
            selection_dir = candidate_dir / "workspace_selection"
            selected = select_task_workspace_candidate(
                plan,
                execution_variant,
                workspace_paths,
                manifest,
                selection_dir,
                int(gpu_ids[worker_slot]),
                conda_env,
                settings,
                seed=seed,
            )
        except (CompileError, OSError, ValueError) as exc:
            failure = failure_code_from_text(exc, "LAYOUT_PLANNING_GATE_FAILED")
            failure_path = candidate_dir / "planning_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "failure_code": failure,
                        "failing_subtask_id": getattr(
                            exc, "failing_subtask_id", None
                        ),
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return CandidateEvaluation(
                candidate.candidate_id,
                candidate.generation,
                seed,
                validation.hard_constraints,
                False,
                failure_code=failure,
                artifact_refs=(*validation.artifact_refs, str(failure_path)),
            )
        result_path = candidate_dir / "layout_selection.json"
        result_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "scene_revision": validation.scene_revision,
                    "scene_task_path": validation.derived_task_path,
                    "scene_arena_path": validation.derived_arena_path,
                    "mutation_path": validation.mutation_path,
                    "workspace_paths": workspace_paths,
                    "workspace_selection_path": str(
                        selection_dir / "position_selection.json"
                    ),
                    "workspace_candidate": selected,
                    "seed": seed,
                    "gpu_id": int(gpu_ids[worker_slot]),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return CandidateEvaluation(
            candidate.candidate_id,
            candidate.generation,
            seed,
            validation.hard_constraints,
            True,
            path_length_m=_path_length(workspace_paths),
            diversity_score=_diversity(candidate),
            artifact_refs=(
                *validation.artifact_refs,
                str(base_task),
                str(selection_dir / "position_selection.json"),
                str(result_path),
            ),
        )

    outcome = EvolutionSearch().run(
        planner.initial_population(),
        evaluate,
        planner.evolve,
        search_dir,
    )
    if outcome.robust_winner is None:
        raise SceneLayoutBlocked(
            "LAYOUT_SEARCH_EXHAUSTED",
            "no K=8/G=5 layout candidate passed every debug seed",
            {"search_dir": str(search_dir)},
        )
    winner_path = (
        output_dir
        / "candidates"
        / outcome.robust_winner.candidate_id
        / "seed_000"
        / "layout_selection.json"
    )
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    result = SceneLayoutSearchResult(
        candidate_id=str(winner["candidate_id"]),
        scene_revision=str(winner["scene_revision"]),
        scene_task_path=str(winner["scene_task_path"]),
        scene_arena_path=str(winner["scene_arena_path"]),
        mutation_path=str(winner["mutation_path"]),
        workspace_paths={
            str(key): str(value) for key, value in winner["workspace_paths"].items()
        },
        workspace_selection_path=str(winner["workspace_selection_path"]),
        workspace_candidate=dict(winner["workspace_candidate"]),
        search_dir=str(search_dir),
        generations_completed=outcome.generations_completed,
    )
    result_path = output_dir / "scene_layout_search_result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _select_subtask(plan: TaskPlan, subtask_id: str | None):
    if subtask_id is None:
        if len(plan.subtasks) != 1:
            raise SceneLayoutBlocked(
                "LAYOUT_SUBTASK_AMBIGUOUS",
                "multi-object layout repair requires the failing subtask id",
            )
        return plan.subtasks[0]
    matches = [item for item in plan.subtasks if item.subtask_id == subtask_id]
    if len(matches) != 1:
        raise SceneLayoutBlocked(
            "LAYOUT_SUBTASK_UNKNOWN",
            f"layout repair subtask {subtask_id!r} is not in the semantic plan",
        )
    return matches[0]


def _path_length(workspace_paths: Mapping[str, str]) -> float:
    values = []
    for path in workspace_paths.values():
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = manifest.get("selected_candidate") or {}
        radius = selected.get("radius_m")
        if radius is not None:
            values.append(float(radius))
    return sum(values) if values else 0.0


def _diversity(candidate) -> float:
    return float(len({mutation.entity for mutation in candidate.mutations}))
