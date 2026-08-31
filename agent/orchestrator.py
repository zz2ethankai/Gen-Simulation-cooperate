"""Bounded, resumable orchestration over the existing SimBox runtime."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .compiler import (
    CompileError,
    compile_task_config,
    generate_workspace_manifest,
    select_task_workspace_candidate,
)
from .contracts import (
    Diagnosis,
    EvidenceBundle,
    ExecutionIdentity,
    ExecutionVariant,
    ResolutionMode,
    RunState,
    RunStatus,
    SceneCapabilityManifest,
    TaskPlan,
    dump_contract,
    load_contract,
)
from .evidence import classify_evidence, collect_evidence
from .inventory import DEFAULT_INDEX_PATH, load_or_build_inventory
from .resolver import AgentDecisionError, CodexBackend, OpenAIBackend, TaskResolver
from .retention import RetentionManager
from .settings import load_agent_settings, resolve_data_generation
from .tools.artifacts import write_variant_artifact_manifest
from .tools.feedback import RepairAction, classify_failure, failure_code_from_text
from .tools.scene_layout import (
    SceneLayoutBlocked,
    SceneLayoutSearchResult,
    run_scene_layout_search,
)
from .tools.scene_layout.models import load_scene_spec
from .tools.source_integrity import build_source_snapshot, write_source_snapshot
from .tools.trace import TraceContext, TraceEvent, TraceWriter
from workflows.simbox.core.robots.profile import load_robot_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "output" / "agent_runs"


def _stop_process_group(
    process: subprocess.Popen[Any], timeout_sec: float = 20.0
) -> int | None:
    if process.poll() is not None:
        return process.wait()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=timeout_sec)


def _execution_identity(
    run_id: str,
    variant: ExecutionVariant,
    seed: int,
    config_path: Path,
    source_snapshot_path: Path,
) -> ExecutionIdentity:
    source_snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    source_hash = source_snapshot.get("source_hash")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError("compiled task must contain exactly one task")
    agent_plan = (tasks[0].get("metadata") or {}).get("agent_plan")
    if not isinstance(agent_plan, dict):
        raise ValueError("compiled task has no metadata.agent_plan")
    return ExecutionIdentity(
        run_id=run_id,
        variant_id=variant.variant_id,
        seed=seed,
        profile_id=variant.profile_id,
        profile_hash=variant.profile_hash,
        source_hash=source_hash,
        scene_revision=agent_plan.get("scene_revision"),
    )

def _create_backend(settings: dict[str, Any], model: str | None) -> CodexBackend | OpenAIBackend:
    backend_cfg = settings.get("backend", {})
    backend_type = str(backend_cfg.get("type", "codex_cli")).lower()
    if backend_type == "openai_api":
        return OpenAIBackend(
            model=model or str(backend_cfg.get("model", "gpt-4o")),
            base_url=backend_cfg.get("base_url") or None,
            api_key=backend_cfg.get("api_key") or None,
        )
    return CodexBackend(model=model)

class AgentOrchestrator:
    def __init__(
        self,
        *,
        gpu: int = 0,
        max_revisions: int = 2,
        conda_env: str = "interndata-isaac6",
        simulator_backend: str | None = None,
        timeout_sec: int = 1800,
        model: str | None = None,
        inventory_path: Path = DEFAULT_INDEX_PATH,
        run_root: Path = DEFAULT_RUN_ROOT,
        retain_experience: bool = True,
        settings: dict[str, Any] | None = None,
    ):
        self.settings = dict(settings or load_agent_settings())
        self.gpu = gpu
        self.max_revisions = max_revisions
        self.conda_env = conda_env
        execution = dict(self.settings.get("execution", {}))
        self.settings["execution"] = execution
        self.simulator_backend = str(
            simulator_backend
            or execution.get("simulator_backend", "docker")
        ).lower()
        if self.simulator_backend not in {"docker", "conda"}:
            raise ValueError(
                "execution.simulator_backend must be 'docker' or 'conda'"
            )
        # Downstream workspace/layout validators receive the effective settings.
        # Reflect CLI overrides there so nested probes use the same launcher.
        execution["simulator_backend"] = self.simulator_backend
        execution["conda_env"] = self.conda_env
        self.timeout_sec = timeout_sec
        self.inventory_path = inventory_path
        self.run_root = run_root
        self.retain_experience = retain_experience
        generation = self.settings.get("generation", {})
        self.random_num = int(generation.get("random_num", 1))
        if self.random_num <= 0:
            raise ValueError("generation.random_num must be positive")
        self.seed = int(generation.get("seed", 0))
        if self.seed < 0:
            raise ValueError("generation.seed must be non-negative")
        self.backend = _create_backend(self.settings, model)
        self.resolver = TaskResolver(self.backend, settings=self.settings)
        self.retention = RetentionManager(self.backend)

    def _layout_gpu_ids(self) -> tuple[int, int, int, int]:
        execution = self.settings.get("execution", {})
        values = execution.get("layout_gpus", [0, 1, 2, 3])
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("execution.layout_gpus must contain exactly four GPU ids")
        gpu_ids = tuple(int(value) for value in values)
        if len(set(gpu_ids)) != 4 or any(value < 0 for value in gpu_ids):
            raise ValueError(
                "execution.layout_gpus must contain four distinct non-negative ids"
            )
        return gpu_ids

    def _search_layout(
        self,
        plan: TaskPlan,
        variant: ExecutionVariant,
        manifest: SceneCapabilityManifest,
        source_task: Path,
        source_arena: Path,
        failure_code: str,
        output_dir: Path,
        *,
        subtask_id: str | None = None,
    ) -> SceneLayoutSearchResult:
        subtask_id = self._resolve_layout_subtask_id(plan, subtask_id)
        result_path = output_dir / "scene_layout_search_result.json"
        if result_path.is_file():
            return SceneLayoutSearchResult.from_path(result_path)
        if output_dir.is_dir() and any(output_dir.iterdir()):
            raise SceneLayoutBlocked(
                "LAYOUT_SEARCH_INCOMPLETE",
                "an incomplete SceneLayout search cannot be resumed as a new search",
                {"output_dir": str(output_dir)},
            )
        return run_scene_layout_search(
            plan,
            variant,
            manifest,
            source_task,
            source_arena,
            failure_code,
            output_dir,
            self._layout_gpu_ids(),
            self.conda_env,
            self.settings,
            subtask_id=subtask_id,
        )

    @staticmethod
    def _resolve_layout_subtask_id(
        plan: TaskPlan, failing_subtask_id: str | None
    ) -> str:
        subtask_ids = {item.subtask_id for item in plan.subtasks}
        if failing_subtask_id is not None:
            if failing_subtask_id not in subtask_ids:
                raise SceneLayoutBlocked(
                    "LAYOUT_SUBTASK_UNKNOWN",
                    f"failure references unknown subtask {failing_subtask_id!r}",
                )
            return failing_subtask_id
        if len(subtask_ids) == 1:
            return next(iter(subtask_ids))
        raise SceneLayoutBlocked(
            "FAILURE_SUBTASK_UNATTRIBUTED",
            "multi-subtask layout repair requires a deterministically attributed failing_subtask_id",
        )

    @staticmethod
    def _layout_results_path(run_dir: Path) -> Path:
        return run_dir / "scene_layout_results.json"

    def _load_layout_results(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        path = self._layout_results_path(run_dir)
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(item, dict)
            for key, item in value.items()
        ):
            raise RuntimeError(f"invalid SceneLayout result index: {path}")
        return value

    def _store_layout_result(
        self,
        run_dir: Path,
        variant_id: str,
        result: SceneLayoutSearchResult,
    ) -> None:
        values = self._load_layout_results(run_dir)
        values[variant_id] = result.to_dict()
        self._layout_results_path(run_dir).write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _source_arena_path(run_dir: Path, source_task: Path) -> Path:
        scene_spec_path = run_dir / "scene_spec.json"
        if scene_spec_path.is_file():
            value = json.loads(scene_spec_path.read_text(encoding="utf-8"))
            source_arena = value.get("source_arena") if isinstance(value, dict) else None
            if isinstance(source_arena, str) and source_arena.strip():
                path = Path(source_arena).expanduser().resolve()
                if path.is_file():
                    return path
        return Path(load_scene_spec(source_task, REPO_ROOT).source_arena)

    @staticmethod
    def _run_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _state_path(run_dir: Path) -> Path:
        return run_dir / "state.json"

    def _save_state(self, state: RunState) -> None:
        dump_contract(state, self._state_path(Path(state.run_dir)))

    def _configured_scene_roots(self) -> list[Path] | None:
        values = self.settings.get("scene_roots")
        if values is None:
            return None
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError("Agent config scene_roots must be a list of non-empty paths")
        return [
            path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
            for path in (Path(value).expanduser() for value in values)
        ]

    def build_index(self, scene_roots: list[Path] | None = None) -> Path:
        from .inventory import build_inventory, write_inventory

        manifests = build_inventory(
            scene_roots if scene_roots is not None else self._configured_scene_roots(),
            settings=self.settings,
        )
        return write_inventory(manifests, self.inventory_path)

    def plan(self, prompt: str, run_id: str | None = None) -> RunState:
        run_id = run_id or self._run_id()
        run_dir = (self.run_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        state = RunState(
            run_id=run_id,
            prompt=prompt,
            status=RunStatus.CREATED,
            run_dir=str(run_dir),
            max_revisions=self.max_revisions,
        )
        self._save_state(state)
        (run_dir / "request.json").write_text(
            json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifests = load_or_build_inventory(
            self.inventory_path,
            self._configured_scene_roots(),
        )
        response, candidates = self.resolver.resolve(prompt, manifests, run_dir / "decisions")
        dump_contract(response, run_dir / "selection.json")
        
        if response.resolution.mode in {
            ResolutionMode.REUSE_EXISTING,
            ResolutionMode.REUSE_SCENE_NEW_TASK,
        }:
            source_manifest = self.resolver.select_source_manifest(
                response.resolution, candidates
            )
        if response.resolution.mode == ResolutionMode.REUSE_EXISTING:
            selected = source_manifest
        elif response.resolution.mode == ResolutionMode.REUSE_SCENE_NEW_TASK:
            try:
                selected = self.resolver.build_synthetic_manifest(
                    response.resolution, source_manifest
                )
            except AgentDecisionError as exc:
                state.status = RunStatus.BLOCKED
                state.message = str(exc)
                self._save_state(state)
                return state
        else:
            composition = self.resolver.composition_request(prompt, response.resolution, candidates)
            dump_contract(composition, run_dir / "scene_composition_request.json")
            state.status = RunStatus.BLOCKED
            state.message = composition.status
            self._save_state(state)
            return state

        selected_manifest_path = run_dir / "selected_manifest.json"
        dump_contract(selected, selected_manifest_path)
        state.selected_manifest_path = str(selected_manifest_path)
        self._save_state(state)
        task_plan = self.resolver.plan(prompt, response, selected, run_dir / "decisions")
        plan_path = run_dir / "semantic_task_plan.json"
        dump_contract(task_plan, plan_path)
        state.task_plan_path = str(plan_path)
        if task_plan.unresolved:
            state.status = RunStatus.BLOCKED
            state.message = "planning has unresolved requirements: " + "; ".join(task_plan.unresolved)
            self._save_state(state)
            return state

        execution_variants = self.resolver.execution_variants(task_plan)
        if not execution_variants:
            state.status = RunStatus.BLOCKED
            state.message = "no admitted execution variant satisfies the semantic plan"
            self._save_state(state)
            return state
        source_task = Path(task_plan.source_task)
        if not source_task.is_absolute():
            source_task = REPO_ROOT / source_task
        try:
            scene_spec = load_scene_spec(source_task, REPO_ROOT)
            scene_spec.write(run_dir / "scene_spec.json")
            source_snapshot = build_source_snapshot(
                scene_spec.source_task,
                scene_spec.source_arena,
                [variant.robot_config_file for variant in execution_variants],
                repo_root=REPO_ROOT,
            )
            source_hash = str(source_snapshot["source_hash"])
            write_source_snapshot(
                source_snapshot,
                run_dir / "source_snapshot.json",
            )
        except (OSError, ValueError) as exc:
            state.status = RunStatus.BLOCKED
            state.message = f"scene normalization failed: {exc}"
            self._save_state(state)
            return state

        TraceWriter(run_dir / "trace.jsonl").append(
            TraceEvent(
                TraceContext(
                    run_id=state.run_id,
                    source_hash=source_hash,
                ),
                stage="semantic_plan",
                status="complete",
                outputs={"semantic_task_plan": str(plan_path)},
                artifact_refs=(str(run_dir / "scene_spec.json"), str(plan_path)),
            )
        )
        workspace_paths: dict[str, dict[str, str]] = {}
        layout_results: dict[str, dict[str, Any]] = {}
        viable_variants: list[ExecutionVariant] = []
        planning_failures: list[dict[str, str]] = []
        for variant in execution_variants:
            variant_root = run_dir / "variants" / variant.variant_id
            base_task_path = variant_root / "base_task.yaml"
            profile = load_robot_profile(variant.robot_config_file)
            if profile.profile_hash != variant.profile_hash:
                raise RuntimeError(
                    f"execution variant profile hash drifted: {variant.variant_id}"
                )
            variant_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                profile.source_path,
                variant_root / "robot_profile.snapshot.yaml",
            )
            (variant_root / "parent.json").write_text(
                json.dumps(
                    {
                        "variant_id": variant.variant_id,
                        "parent_variant_id": None,
                        "source_task": scene_spec.source_task,
                        "source_task_hash": scene_spec.source_task_hash,
                        "source_arena": scene_spec.source_arena,
                        "source_arena_hash": scene_spec.source_arena_hash,
                        "source_hash": source_hash,
                        "scene_revision": "source",
                        "world_revision": None,
                        "profile_id": variant.profile_id,
                        "profile_hash": variant.profile_hash,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                compile_task_config(
                    task_plan,
                    variant,
                    selected,
                    base_task_path,
                    settings=self.settings,
                )
            except (CompileError, ValueError) as exc:
                failure = {
                    "variant_id": variant.variant_id,
                    "failure_code": "VARIANT_COMPILATION_FAILED",
                    "message": str(exc),
                }
                planning_failures.append(failure)
                (variant_root / "planning_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                continue

            variant_paths: dict[str, str] = {}
            preparation_error: Exception | None = None
            for subtask in task_plan.subtasks:
                workspace_dir = (
                    variant_root
                    / "subtasks"
                    / subtask.subtask_id
                    / "workspace"
                )
                try:
                    manifest_path = generate_workspace_manifest(
                        base_task_path,
                        subtask.manipulated_object,
                        variant.arm_binding[subtask.subtask_id],
                        workspace_dir,
                        variant.placement_family,
                    )
                    variant_paths[subtask.subtask_id] = str(manifest_path)
                except (CompileError, ValueError) as exc:
                    failure_code = failure_code_from_text(
                        exc, "VARIANT_PREPARATION_FAILED"
                    )
                    repair = classify_failure(failure_code, "workspace")
                    if repair.action != RepairAction.MUTATE_LAYOUT:
                        preparation_error = exc
                        break
                    try:
                        layout_result = self._search_layout(
                            task_plan,
                            variant,
                            selected,
                            source_task,
                            Path(scene_spec.source_arena),
                            failure_code,
                            variant_root / "scene_layout" / "initial",
                            subtask_id=subtask.subtask_id,
                        )
                    except (SceneLayoutBlocked, CompileError, OSError, ValueError) as layout_exc:
                        preparation_error = layout_exc
                        break
                    variant_paths = dict(layout_result.workspace_paths)
                    layout_results[variant.variant_id] = layout_result.to_dict()
                    break
            if preparation_error is None and set(variant_paths) == {
                subtask.subtask_id for subtask in task_plan.subtasks
            }:
                workspace_paths[variant.variant_id] = variant_paths
                viable_variants.append(variant)
            else:
                exc = preparation_error or RuntimeError(
                    "workspace preparation did not cover every semantic subtask"
                )
                failure = {
                    "variant_id": variant.variant_id,
                    "failure_code": failure_code_from_text(
                        exc, "VARIANT_PREPARATION_FAILED"
                    ),
                    "message": str(exc),
                }
                planning_failures.append(failure)
                (variant_root / "planning_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        if not viable_variants:
            state.status = RunStatus.BLOCKED
            state.message = "all execution variants failed deterministic preparation"
            self._save_state(state)
            return state
        (run_dir / "execution_variants.json").write_text(
            json.dumps(
                [variant.to_dict() for variant in viable_variants],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if planning_failures:
            (run_dir / "variant_planning_failures.json").write_text(
                json.dumps(planning_failures, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (run_dir / "workspace_manifests.json").write_text(
            json.dumps(workspace_paths, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if layout_results:
            self._layout_results_path(run_dir).write_text(
                json.dumps(layout_results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        first_variant_paths = next(iter(workspace_paths.values()), {})
        state.workspace_manifest_path = next(iter(first_variant_paths.values()), None)
        state.status = RunStatus.PLANNED
        state.message = "semantic plan and geometry workspace candidates are ready"
        self._save_state(state)
        return state

    def run(self, prompt: str) -> RunState:
        state = self.plan(prompt)
        if state.status != RunStatus.PLANNED:
            return state
        return self._execute_planned(state)

    def resume(self, run_id: str) -> RunState:
        run_dir = (self.run_root / run_id).resolve()
        state = load_contract(RunState, self._state_path(run_dir))
        if state.status == RunStatus.SUCCEEDED:
            return state
        if not state.task_plan_path or not Path(state.task_plan_path).is_file():
            raise RuntimeError(f"run has no resumable TaskPlan: {run_id}")
        if (
            not state.selected_manifest_path
            or not Path(state.selected_manifest_path).is_file()
        ):
            raise RuntimeError(f"run has no resumable selected manifest: {run_id}")
        if not (run_dir / "workspace_manifests.json").is_file():
            raise RuntimeError(f"run has no workspace manifests: {run_id}")
        return self._execute_planned(state)

    def _execute_planned(self, state: RunState) -> RunState:
        run_dir = Path(state.run_dir)
        plan = load_contract(TaskPlan, Path(str(state.task_plan_path)))
        if not state.selected_manifest_path:
            raise RuntimeError(f"run has no selected manifest: {state.run_id}")
        selected_manifest = load_contract(
            SceneCapabilityManifest, Path(state.selected_manifest_path)
        )
        self.resolver.validate_plan(plan, selected_manifest)
        execution_variants = [
            ExecutionVariant.from_dict(value)
            for value in json.loads(
                (run_dir / "execution_variants.json").read_text(encoding="utf-8")
            )
        ]
        if not execution_variants:
            raise RuntimeError(f"run has no execution variants: {state.run_id}")
        all_workspace_paths = json.loads(
            (run_dir / "workspace_manifests.json").read_text(encoding="utf-8")
        )
        layout_results = self._load_layout_results(run_dir)
        source_task = Path(plan.source_task)
        if not source_task.is_absolute():
            source_task = REPO_ROOT / source_task
        execution_variant = None
        selected_candidate = None
        workspace_paths: dict[str, str] = {}
        workspace_selection_dir = run_dir / "workspace_selection"
        workspace_selection_path = workspace_selection_dir / "position_selection.json"
        active_scene_task_path: Path | None = None
        variant_selection_failures: list[dict[str, str]] = []
        excluded_candidates: set[str] = set()
        for candidate_variant in execution_variants:
            candidate_paths = all_workspace_paths.get(candidate_variant.variant_id)
            if not isinstance(candidate_paths, dict):
                variant_selection_failures.append(
                    {
                        "variant_id": candidate_variant.variant_id,
                        "message": "workspace manifests are missing",
                    }
                )
                continue
            candidate_selection_dir = (
                run_dir / "variants" / candidate_variant.variant_id / "workspace_selection"
            )
            candidate_layout = layout_results.get(candidate_variant.variant_id)
            if candidate_layout is not None:
                try:
                    layout_result = SceneLayoutSearchResult(**candidate_layout)
                except TypeError as exc:
                    variant_selection_failures.append(
                        {
                            "variant_id": candidate_variant.variant_id,
                            "message": f"invalid SceneLayout result: {exc}",
                        }
                    )
                    continue
                selection_path = Path(layout_result.workspace_selection_path)
                scene_task_path = Path(layout_result.scene_task_path)
                if not selection_path.is_file() or not scene_task_path.is_file():
                    variant_selection_failures.append(
                        {
                            "variant_id": candidate_variant.variant_id,
                            "message": "SceneLayout result artifacts are missing",
                        }
                    )
                    continue
                execution_variant = candidate_variant
                workspace_paths = dict(layout_result.workspace_paths)
                workspace_selection_dir = selection_path.parent
                workspace_selection_path = selection_path
                active_scene_task_path = scene_task_path
                selected_candidate = dict(layout_result.workspace_candidate)
                break
            try:
                candidate = select_task_workspace_candidate(
                    plan,
                    candidate_variant,
                    candidate_paths,
                    selected_manifest,
                    candidate_selection_dir,
                    self.gpu,
                    self.conda_env,
                    self.settings,
                )
            except CompileError as exc:
                failure_code = failure_code_from_text(
                    exc, "NO_CUROBO_CANDIDATE"
                )
                repair = classify_failure(failure_code, "workspace")
                if repair.action in {
                    RepairAction.MUTATE_LAYOUT,
                    RepairAction.NEXT_CANDIDATE,
                }:
                    try:
                        layout_result = self._search_layout(
                            plan,
                            candidate_variant,
                            selected_manifest,
                            source_task,
                            self._source_arena_path(run_dir, source_task),
                            failure_code,
                            run_dir
                            / "variants"
                            / candidate_variant.variant_id
                            / "scene_layout"
                            / "selection_failure",
                            subtask_id=exc.failing_subtask_id,
                        )
                    except (SceneLayoutBlocked, CompileError, OSError, ValueError) as layout_exc:
                        variant_selection_failures.append(
                            {
                                "variant_id": candidate_variant.variant_id,
                                "message": str(layout_exc),
                            }
                        )
                        continue
                    self._store_layout_result(
                        run_dir, candidate_variant.variant_id, layout_result
                    )
                    execution_variant = candidate_variant
                    workspace_paths = dict(layout_result.workspace_paths)
                    workspace_selection_path = Path(
                        layout_result.workspace_selection_path
                    )
                    workspace_selection_dir = workspace_selection_path.parent
                    active_scene_task_path = Path(layout_result.scene_task_path)
                    selected_candidate = dict(layout_result.workspace_candidate)
                    break
                variant_selection_failures.append(
                    {"variant_id": candidate_variant.variant_id, "message": str(exc)}
                )
                continue
            execution_variant = candidate_variant
            workspace_paths = candidate_paths
            workspace_selection_dir = candidate_selection_dir
            workspace_selection_path = candidate_selection_dir / "position_selection.json"
            selected_candidate = candidate
            break
        if execution_variant is None or selected_candidate is None:
            state.status = RunStatus.BLOCKED
            state.message = "no execution variant has a CuRobo-feasible workspace candidate"
            state.workspace_manifest_path = str(workspace_selection_path)
            (run_dir / "variant_selection_failures.json").write_text(
                json.dumps(variant_selection_failures, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._save_state(state)
            self._write_report(state)
            return state

        state.status = RunStatus.WORKSPACE_READY
        state.workspace_manifest_path = str(workspace_selection_path)
        self._save_state(state)

        start_attempt = state.attempt_index
        data_generation = resolve_data_generation(
            plan.task_request.data_generation,
            self.settings,
        )
        variant_root = run_dir / "variants" / execution_variant.variant_id
        for attempt_index in range(start_attempt, state.max_revisions + 1):
            state.status = RunStatus.RUNNING
            state.attempt_index = attempt_index
            attempt_dir = variant_root / "attempts" / f"{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            trace = TraceWriter(attempt_dir / "trace.jsonl")
            dump_contract(plan, attempt_dir / "semantic_task_plan.json")
            config_path = attempt_dir / "task.yaml"
            try:
                compile_task_config(
                    plan,
                    execution_variant,
                    selected_manifest,
                    config_path,
                    workspace_candidate=selected_candidate,
                    settings=self.settings,
                    scene_task_path=active_scene_task_path,
                )
                expected_identity = _execution_identity(
                    state.run_id,
                    execution_variant,
                    self.seed,
                    config_path,
                    run_dir / "source_snapshot.json",
                )
            except (CompileError, ValueError) as exc:
                diagnosis = Diagnosis(
                    stage="configuration",
                    failure_code="INVALID_TASK_CONFIG",
                    category="configuration",
                    root_cause=str(exc),
                    retryable=False,
                    recommended_action="repair the generated deterministic config contract",
                )
                dump_contract(diagnosis, attempt_dir / "diagnosis.json")
                state.status = RunStatus.BLOCKED
                state.message = str(exc)
                state.last_diagnosis_path = str(attempt_dir / "diagnosis.json")
                self._save_state(state)
                self._write_report(state)
                return state

            state.config_path = str(config_path)
            self._save_state(state)
            return_code, timed_out, event_path, log_path = self._run_simbox(
                config_path,
                attempt_dir,
                expected_identity,
                data_generation=data_generation,
            )
            evidence = collect_evidence(
                f"task:{attempt_index:02d}",
                attempt_dir,
                event_path,
                log_path,
                return_code,
                timed_out,
                expected_identity=expected_identity,
                data_generation_required=data_generation,
                robot_profile_path=execution_variant.robot_config_file,
                compiled_task_path=config_path,
                source_snapshot_path=run_dir / "source_snapshot.json",
            )
            evidence_path = attempt_dir / "evidence.json"
            diagnosis = classify_evidence(evidence)
            diagnosis_path = attempt_dir / "diagnosis.json"
            dump_contract(diagnosis, diagnosis_path)
            state.last_evidence_path = str(evidence_path)
            state.last_diagnosis_path = str(diagnosis_path)
            self._save_state(state)
            trace.append(
                TraceEvent(
                    TraceContext(
                        run_id=state.run_id,
                        variant_id=execution_variant.variant_id,
                        attempt_id=attempt_dir.name,
                        parent_variant_id="",
                        seed=self.seed,
                        profile_id=execution_variant.profile_id,
                        profile_hash=execution_variant.profile_hash,
                        source_hash=expected_identity.source_hash,
                        scene_revision=expected_identity.scene_revision,
                        world_revision=(
                            evidence.identity.world_revision
                            if evidence.identity is not None
                            else None
                        ),
                    ),
                    stage="episode_evaluation",
                    status="success" if evidence.task_success else "failed",
                    failure_code="" if evidence.task_success else diagnosis.failure_code,
                    inputs={"config": str(config_path)},
                    outputs={
                        "strict_success": evidence.task_success,
                        "identity_errors": evidence.identity_errors,
                        "variant_signature": evidence.variant_signature,
                    },
                    artifact_refs=tuple(evidence.artifact_refs),
                )
            )
            should_repair = (
                not evidence.task_success
                and diagnosis.retryable
                and attempt_index < state.max_revisions
            )
            if should_repair:
                repair = classify_failure(
                    diagnosis.failure_code,
                    diagnosis.category,
                    failing_subtask_id=diagnosis.failing_subtask_id,
                )
                if repair.action == RepairAction.MUTATE_LAYOUT:
                    try:
                        diagnosis.failing_subtask_id = self._resolve_layout_subtask_id(
                            plan, diagnosis.failing_subtask_id
                        )
                    except SceneLayoutBlocked as exc:
                        diagnosis.failure_code = exc.failure_code
                        diagnosis.category = "data_integrity"
                        diagnosis.root_cause = str(exc)
                        diagnosis.retryable = False
                        diagnosis.recommended_action = (
                            "record the active compiled subtask in the failure event before layout repair"
                        )
                        diagnosis.workspace_action = "block"
                        repair = classify_failure(
                            diagnosis.failure_code,
                            diagnosis.category,
                        )
                    else:
                        repair = classify_failure(
                            diagnosis.failure_code,
                            diagnosis.category,
                            failing_subtask_id=diagnosis.failing_subtask_id,
                        )
                (attempt_dir / "repair.json").write_text(
                    json.dumps(repair.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                trace.append(
                    TraceEvent(
                        TraceContext(
                            run_id=state.run_id,
                            variant_id=execution_variant.variant_id,
                            attempt_id=attempt_dir.name,
                            seed=self.seed,
                            profile_id=execution_variant.profile_id,
                            profile_hash=execution_variant.profile_hash,
                            source_hash=expected_identity.source_hash,
                            scene_revision=expected_identity.scene_revision,
                            world_revision=(
                                evidence.identity.world_revision
                                if evidence.identity is not None
                                else None
                            ),
                        ),
                        stage="feedback",
                        status=repair.action.value,
                        failure_code=diagnosis.failure_code,
                        outputs=repair.to_dict(),
                        artifact_refs=(str(attempt_dir / "repair.json"),),
                    )
                )
                if repair.action == RepairAction.DIAGNOSE:
                    proposal = self.resolver.diagnose_unknown(
                        plan, evidence, attempt_dir / "decisions"
                    )
                    diagnosis.skill_updates = proposal.skill_updates
                    diagnosis.workspace_action = proposal.workspace_action
                    diagnosis.recommended_action = proposal.recommended_action
                elif repair.action == RepairAction.NEXT_CANDIDATE:
                    diagnosis.workspace_action = "replan"
                elif repair.action == RepairAction.MUTATE_LAYOUT:
                    diagnosis.workspace_action = "mutate_layout"
                elif repair.action == RepairAction.BLOCK:
                    diagnosis.workspace_action = "block"
                else:
                    diagnosis.workspace_action = "keep"
                dump_contract(diagnosis, diagnosis_path)

            artifact_manifest = write_variant_artifact_manifest(
                variant_root,
                attempt_dir,
                workspace_selection_path,
                data_required=data_generation,
            )
            if evidence.task_success and not artifact_manifest.complete:
                diagnosis = Diagnosis(
                    stage="artifact_contract",
                    failure_code="ARTIFACT_CONTRACT_FAILED",
                    category="data_integrity",
                    root_cause=", ".join(artifact_manifest.failure_codes),
                    retryable=False,
                    recommended_action=(
                        "produce every required structured artifact before qualification"
                    ),
                    evidence_refs=[str(attempt_dir / "artifact_manifest.json")],
                )
                dump_contract(diagnosis, diagnosis_path)
                state.status = RunStatus.FAILED
                state.message = (
                    "ARTIFACT_CONTRACT_FAILED: "
                    + ", ".join(artifact_manifest.failure_codes)
                )
                self._save_state(state)
                self._finish(state, plan)
                return state
            if evidence.task_success:
                state.status = RunStatus.SUCCEEDED
                state.message = "all object subtasks completed in one SimBox episode"
                state.current_subtask = len(plan.subtasks)
                state.attempt_index = 0
                dump_contract(plan, Path(str(state.task_plan_path)))
                self._save_state(state)
                self._finish(state, plan)
                return state
            if not should_repair:
                state.status = RunStatus.FAILED
                state.message = f"{diagnosis.failure_code}: {diagnosis.recommended_action}"
                self._save_state(state)
                self._finish(state, plan)
                return state

            changed = self._apply_skill_updates(plan, diagnosis, selected_manifest)
            if diagnosis.workspace_action == "mutate_layout":
                try:
                    layout_result = self._search_layout(
                        plan,
                        execution_variant,
                        selected_manifest,
                        source_task,
                        self._source_arena_path(run_dir, source_task),
                        diagnosis.failure_code,
                        variant_root
                        / "scene_layout"
                        / "repairs"
                        / f"attempt_{attempt_index + 1:02d}",
                        subtask_id=diagnosis.failing_subtask_id,
                    )
                except (SceneLayoutBlocked, CompileError, OSError, ValueError) as exc:
                    state.status = RunStatus.BLOCKED
                    state.message = str(exc)
                    self._save_state(state)
                    self._finish(state, plan)
                    return state
                self._store_layout_result(
                    run_dir, execution_variant.variant_id, layout_result
                )
                active_scene_task_path = Path(layout_result.scene_task_path)
                workspace_paths = dict(layout_result.workspace_paths)
                selected_candidate = dict(layout_result.workspace_candidate)
                workspace_selection_path = Path(
                    layout_result.workspace_selection_path
                )
                workspace_selection_dir = workspace_selection_path.parent
                excluded_candidates.clear()
                changed = True
            if diagnosis.workspace_action == "replan":
                excluded_candidates.add(str(selected_candidate["candidate_id"]))
                try:
                    replan_dir = workspace_selection_dir / f"replan_{attempt_index + 1:02d}"
                    selected_candidate = select_task_workspace_candidate(
                        plan,
                        execution_variant,
                        workspace_paths,
                        selected_manifest,
                        replan_dir,
                        self.gpu,
                        self.conda_env,
                        self.settings,
                        excluded_candidate_ids=excluded_candidates,
                    )
                    workspace_selection_dir = replan_dir
                    workspace_selection_path = replan_dir / "position_selection.json"
                    changed = True
                except CompileError as exc:
                    failure_code = failure_code_from_text(
                        exc, "NO_CUROBO_CANDIDATE"
                    )
                    try:
                        layout_result = self._search_layout(
                            plan,
                            execution_variant,
                            selected_manifest,
                            source_task,
                            self._source_arena_path(run_dir, source_task),
                            failure_code,
                            variant_root
                            / "scene_layout"
                            / "repairs"
                            / f"replan_{attempt_index + 1:02d}",
                            subtask_id=(
                                exc.failing_subtask_id
                                or diagnosis.failing_subtask_id
                            ),
                        )
                    except (SceneLayoutBlocked, CompileError, OSError, ValueError) as layout_exc:
                        state.status = RunStatus.BLOCKED
                        state.message = str(layout_exc)
                        self._save_state(state)
                        self._finish(state, plan)
                        return state
                    self._store_layout_result(
                        run_dir, execution_variant.variant_id, layout_result
                    )
                    active_scene_task_path = Path(layout_result.scene_task_path)
                    workspace_paths = dict(layout_result.workspace_paths)
                    selected_candidate = dict(layout_result.workspace_candidate)
                    workspace_selection_path = Path(
                        layout_result.workspace_selection_path
                    )
                    workspace_selection_dir = workspace_selection_path.parent
                    excluded_candidates.clear()
                    changed = True
            state.workspace_manifest_path = str(workspace_selection_path)
            self._save_state(state)
            if diagnosis.workspace_action == "block" or not changed:
                state.status = RunStatus.BLOCKED
                state.message = "diagnosis produced no safe, applicable single-cause revision"
                self._save_state(state)
                self._finish(state, plan)
                return state
            dump_contract(plan, Path(str(state.task_plan_path)))

        state.status = RunStatus.FAILED
        state.message = "task did not succeed within the revision budget"
        self._save_state(state)
        self._finish(state, plan)
        return state

    def _run_simbox(
        self,
        config_path: Path,
        attempt_dir: Path,
        identity: ExecutionIdentity,
        *,
        data_generation: bool,
    ) -> tuple[int | None, bool, Path, Path]:
        event_path = attempt_dir / "episode_events.jsonl"
        log_path = attempt_dir / "stdout.log"
        data_dir = attempt_dir / "data"
        env = os.environ.copy()
        debug_cfg = self.settings.get("debug", {})
        execution_cfg = self.settings.get("execution", {})
        docker_cfg = execution_cfg.get("docker", {})
        conda_cfg = execution_cfg.get("conda", {})
        launcher_cfg = (
            docker_cfg if self.simulator_backend == "docker" else conda_cfg
        )
        env.update(
            {
                "TASK_CONFIG": str(config_path),
                "GPU_ID": str(self.gpu),
                "RANDOM_NUM": str(self.random_num),
                "RANDOM_SEED": str(identity.seed),
                "RUN_NAME": f"agent/{identity.run_id}/{identity.variant_id}/seed_{identity.seed}",
                "OUTPUT_DIR": str(data_dir) if data_generation else "",
                "INTERNDATA_EPISODE_EVENT_PATH": str(event_path),
                "INTERNDATA_RUN_ID": identity.run_id,
                "INTERNDATA_VARIANT_ID": identity.variant_id,
                "INTERNDATA_RANDOM_SEED": str(identity.seed),
                "INTERNDATA_PROFILE_ID": identity.profile_id,
                "INTERNDATA_PROFILE_HASH": identity.profile_hash,
                "INTERNDATA_SOURCE_HASH": identity.source_hash,
                "INTERNDATA_SCENE_REVISION": identity.scene_revision,
                "INTERNDATA_GPU": str(self.gpu),
                "INTERNDATA_DEBUG_TOPDOWN": "1" if debug_cfg.get("topdown_check") else "0",
                "INTERNDATA_SCREENSHOT_DIR": str(
                    (attempt_dir / "screenshots").resolve()
                ),
                "INTERNDATA_TASK_PATH": str(config_path),
                "INTERNDATA_SIMULATOR_BACKEND": self.simulator_backend,
                "INTERNDATA_STACK_ID": (
                    f"agent-{identity.run_id}-{identity.variant_id}-seed-{identity.seed}"
                ),
                "LAUNCH_TEMPLATE": str(
                    launcher_cfg.get(
                        "launcher_config",
                        "configs/de_plan_with_render_template.yaml",
                    )
                ),
                "SIMBOX_DEBUG_OUTPUT_DIR": str(attempt_dir / "simbox_debug"),
                "CONDA_ENV": self.conda_env,
                "PYTHONUNBUFFERED": "1",
            }
        )
        if self.simulator_backend == "docker":
            env.update(
                {
                    "INTERNDATA_COMPOSE_FILE": str(
                        docker_cfg.get(
                            "compose_file", "docker/docker-compose.yml"
                        )
                    ),
                    "INTERNDATA_DOCKER_METADATA_PATH": str(
                        attempt_dir / "docker_runtime.json"
                    ),
                }
            )
            command = ["bash", "scripts/docker/up_simbox_isaac.sh"]
        else:
            command = ["bash", "scripts/simbox/run_simbox_task.sh"]
        (attempt_dir / "command.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "env": {
                        key: env[key]
                        for key in sorted(env)
                        if key.startswith("INTERNDATA_")
                        or key
                        in {
                            "TASK_CONFIG",
                            "GPU_ID",
                            "RANDOM_NUM",
                            "RANDOM_SEED",
                            "RUN_NAME",
                            "OUTPUT_DIR",
                            "LAUNCH_TEMPLATE",
                            "SIMBOX_DEBUG_OUTPUT_DIR",
                            "CONDA_ENV",
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        timed_out = False
        return_code: int | None
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_group(process)
                return_code = None
        return return_code, timed_out, event_path, log_path

    def _apply_skill_updates(
        self,
        plan: TaskPlan,
        diagnosis: Diagnosis,
        manifest: SceneCapabilityManifest,
    ) -> bool:
        if not diagnosis.skill_updates:
            return False
        data = plan.to_dict()
        changed = False
        contracts = self.resolver.skill_contracts
        for update in diagnosis.skill_updates:
            subtask = next(
                (item for item in data["subtasks"] if item["subtask_id"] == update.subtask_id),
                None,
            )
            if subtask is None:
                raise AgentDecisionError(f"revision references unknown subtask: {update.subtask_id}")
            stage = next(
                (item for item in subtask["stages"] if item["stage_id"] == update.stage_id),
                None,
            )
            if stage is None or not 0 <= update.skill_index < len(stage["skills"]):
                raise AgentDecisionError("revision references an unknown stage or skill index")
            skill = stage["skills"][update.skill_index]
            contract = contracts[skill["name"]]
            unknown = set(update.params) - set(contract.allowed_params)
            if unknown:
                raise AgentDecisionError(f"revision contains unsupported params: {sorted(unknown)}")
            for key, value in update.params.items():
                if skill.setdefault("params", {}).get(key) != value:
                    skill["params"][key] = value
                    changed = True
        if changed:
            revised = TaskPlan.from_dict(data)
            self.resolver.validate_plan(revised, manifest)
            plan.__dict__.update(revised.__dict__)
        return changed

    def diagnose_path(self, run_dir: Path) -> Diagnosis:
        evidence_paths = sorted(run_dir.glob("**/evidence.json"))
        if not evidence_paths:
            raise FileNotFoundError(f"no Agent evidence.json found under {run_dir}")
        evidence = load_contract(EvidenceBundle, evidence_paths[-1])
        diagnosis = classify_evidence(evidence)
        dump_contract(diagnosis, evidence_paths[-1].with_name("diagnosis.manual.json"))
        return diagnosis

    def _finish(self, state: RunState, plan: TaskPlan) -> None:
        report = self._write_report(state)
        if not self.retain_experience:
            return
        try:
            decision = self.retention.decide(report, Path(state.run_dir) / "decisions")
            dump_contract(decision, Path(state.run_dir) / "retention.json")
            self.retention.materialize(decision)
        except (AgentDecisionError, OSError, ValueError) as exc:
            (Path(state.run_dir) / "retention_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    def _write_report(self, state: RunState) -> dict[str, Any]:
        run_dir = Path(state.run_dir)
        attempts = []
        for path in sorted(run_dir.glob("variants/*/attempts/*/evidence.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            value["path"] = str(path)
            attempts.append(value)
        report = {
            "run_id": state.run_id,
            "prompt": state.prompt,
            "status": state.to_dict()["status"],
            "message": state.message,
            "completed_subtasks": state.current_subtask,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "state": str(self._state_path(run_dir)),
        }
        (run_dir / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            "# InternDataEngine Agent Run Report",
            "",
            f"- Run ID: `{state.run_id}`",
            f"- Status: `{state.to_dict()['status']}`",
            f"- Prompt: {state.prompt}",
            f"- Message: {state.message}",
            f"- Attempts: {len(attempts)}",
            "",
            "| Attempt | Status | Failure | Episode |",
            "|---|---|---|---|",
        ]
        for item in attempts:
            lines.append(
                f"| `{item.get('attempt_id')}` | `{item.get('status')}` | "
                f"`{item.get('failure_reason') or ''}` | `{item.get('episode_dir') or ''}` |"
            )
        (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report
