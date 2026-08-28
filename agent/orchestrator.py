"""Bounded, resumable orchestration over the existing SimBox runtime."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .compiler import (
    CompileError,
    compile_task_config,
    generate_workspace_manifest,
    select_task_workspace_candidate,
)
from .contracts import (
    Diagnosis,
    EvidenceBundle,
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
from .resolver import AgentDecisionError, CodexBackend, TaskResolver,OpenAIBackend
from .retention import RetentionManager
from .settings import load_agent_settings, resolve_data_generation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "output" / "agent_runs"


def _stop_process_group(process: subprocess.Popen[Any], timeout_sec: float = 20.0) -> int | None:
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
        timeout_sec: int = 1800,
        model: str | None = None,
        inventory_path: Path = DEFAULT_INDEX_PATH,
        run_root: Path = DEFAULT_RUN_ROOT,
        retain_experience: bool = True,
        settings: dict[str, Any] | None = None,
    ):
        self.settings = settings or load_agent_settings()
        self.gpu = gpu
        self.max_revisions = max_revisions
        self.timeout_sec = timeout_sec
        self.inventory_path = inventory_path
        self.run_root = run_root
        self.retain_experience = retain_experience
        generation = self.settings.get("generation", {})
        self.random_num = int(generation.get("random_num", 1))
        if self.random_num <= 0:
            raise ValueError("generation.random_num must be positive")
        self.backend = _create_backend(self.settings, model)
        self.resolver = TaskResolver(self.backend, settings=self.settings)
        self.retention = RetentionManager(self.backend)

    @staticmethod
    def _run_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _state_path(run_dir: Path) -> Path:
        return run_dir / "state.json"

    def _save_state(self, state: RunState) -> None:
        dump_contract(state, self._state_path(Path(state.run_dir)))

    def build_index(self, scene_roots: list[Path] | None = None) -> Path:
        from .inventory import build_inventory, write_inventory

        manifests = build_inventory(scene_roots)
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

        manifests = load_or_build_inventory(self.inventory_path)
        response, candidates = self.resolver.resolve(prompt, manifests, run_dir / "decisions")
        dump_contract(response, run_dir / "selection.json")

        if response.resolution.mode == ResolutionMode.REUSE_EXISTING:
            selected = next(
                item for item in manifests if item.task_id == response.resolution.selected_task_id
            )
        elif response.resolution.mode == ResolutionMode.REUSE_SCENE_NEW_TASK:
            try:
                selected = self.resolver.build_synthetic_manifest(
                    response.resolution, candidates
                )
            except AgentDecisionError as exc:
                state.status = RunStatus.BLOCKED
                state.message = str(exc)
                self._save_state(state)
                return state
            dump_contract(selected, run_dir / "synthetic_manifest.json")
        else:
            composition = self.resolver.composition_request(prompt, response.resolution, candidates)
            dump_contract(composition, run_dir / "scene_composition_request.json")
            state.status = RunStatus.BLOCKED
            state.message = composition.status
            self._save_state(state)
            return state

        # selected = next(
        #     item for item in manifests if item.task_id == response.resolution.selected_task_id
        # )
        task_plan = self.resolver.plan(prompt, response, selected, run_dir / "decisions")
        plan_path = run_dir / "task_plan.json"
        dump_contract(task_plan, plan_path)
        state.task_plan_path = str(plan_path)
        if task_plan.unresolved:
            state.status = RunStatus.BLOCKED
            state.message = "planning has unresolved requirements: " + "; ".join(task_plan.unresolved)
            self._save_state(state)
            return state

        workspace_paths: dict[str, str] = {}
        try:
            for subtask in task_plan.subtasks:
                workspace_dir = run_dir / "subtasks" / subtask.subtask_id / "workspace"
                manifest_path = generate_workspace_manifest(
                    Path(task_plan.source_task),
                    subtask.manipulated_object,
                    subtask.arm,
                    workspace_dir,
                    selected.robot_mounting,
                )
                workspace_paths[subtask.subtask_id] = str(manifest_path)
        except CompileError as exc:
            state.status = RunStatus.BLOCKED
            state.message = str(exc)
            self._save_state(state)
            return state
        (run_dir / "workspace_manifests.json").write_text(
            json.dumps(workspace_paths, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        state.workspace_manifest_path = next(iter(workspace_paths.values()), None)
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
        if not (run_dir / "workspace_manifests.json").is_file():
            raise RuntimeError(f"run has no workspace manifests: {run_id}")
        return self._execute_planned(state)

    def _execute_planned(self, state: RunState) -> RunState:
        debug_cfg = self.settings.get("debug", {})
        if debug_cfg.get("topdown_check"):
            os.environ["INTERNDATA_DEBUG_TOPDOWN"] = "1"
        else:
            os.environ.pop("INTERNDATA_DEBUG_TOPDOWN", None)
        run_dir = Path(state.run_dir)
        plan = load_contract(TaskPlan, Path(str(state.task_plan_path)))
        manifests = load_or_build_inventory(self.inventory_path)
        selected_manifest = next(item for item in manifests if item.task_id == plan.selected_task_id)
        workspace_paths = json.loads((run_dir / "workspace_manifests.json").read_text(encoding="utf-8"))
        workspace_selection_dir = run_dir / "workspace_selection"
        excluded_candidates: set[str] = set()
        try:
            selected_candidate = select_task_workspace_candidate(
                plan,
                workspace_paths,
                selected_manifest,
                workspace_selection_dir,
                self.gpu,
                self.settings,
            )
        except CompileError as exc:
            state.status = RunStatus.BLOCKED
            state.message = str(exc)
            state.workspace_manifest_path = str(workspace_selection_dir / "position_selection.json")
            self._save_state(state)
            self._write_report(state)
            return state

        state.status = RunStatus.WORKSPACE_READY
        state.workspace_manifest_path = str(workspace_selection_dir / "position_selection.json")
        self._save_state(state)

        start_attempt = state.attempt_index
        for attempt_index in range(start_attempt, state.max_revisions + 1):
            state.status = RunStatus.RUNNING
            state.attempt_index = attempt_index
            attempt_dir = run_dir / "attempts" / f"{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            dump_contract(plan, attempt_dir / "task_plan.json")
            config_path = attempt_dir / "task.yaml"
            try:
                compile_task_config(
                    plan,
                    selected_manifest,
                    config_path,
                    workspace_candidate=selected_candidate,
                    settings=self.settings,
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
                state.run_id,
                attempt_index,
                data_generation=resolve_data_generation(
                    plan.task_request.data_generation,
                    self.settings,
                ),
            )
            evidence = collect_evidence(
                f"task:{attempt_index:02d}",
                attempt_dir,
                event_path,
                log_path,
                return_code,
                timed_out,
            )
            evidence_path = attempt_dir / "evidence.json"
            diagnosis = classify_evidence(evidence)
            diagnosis_path = attempt_dir / "diagnosis.json"
            dump_contract(diagnosis, diagnosis_path)
            state.last_evidence_path = str(evidence_path)
            state.last_diagnosis_path = str(diagnosis_path)
            self._save_state(state)
            if evidence.task_success:
                state.status = RunStatus.SUCCEEDED
                state.message = "all object subtasks completed in one SimBox episode"
                state.current_subtask = len(plan.subtasks)
                state.attempt_index = 0
                dump_contract(plan, Path(str(state.task_plan_path)))
                self._save_state(state)
                self._finish(state, plan)
                return state
            if not diagnosis.retryable or attempt_index >= state.max_revisions:
                state.status = RunStatus.FAILED
                state.message = f"{diagnosis.failure_code}: {diagnosis.recommended_action}"
                self._save_state(state)
                self._finish(state, plan)
                return state

            proposal = self.resolver.diagnose_unknown(
                plan, evidence, attempt_dir / "decisions"
            )
            diagnosis.skill_updates = proposal.skill_updates
            if proposal.workspace_action in {"replan", "block"}:
                diagnosis.workspace_action = proposal.workspace_action
            diagnosis.recommended_action = proposal.recommended_action
            dump_contract(diagnosis, diagnosis_path)
            changed = self._apply_skill_updates(plan, diagnosis, selected_manifest)
            if diagnosis.workspace_action == "replan":
                excluded_candidates.add(str(selected_candidate["candidate_id"]))
                try:
                    selected_candidate = select_task_workspace_candidate(
                        plan,
                        workspace_paths,
                        selected_manifest,
                        workspace_selection_dir / f"replan_{attempt_index + 1:02d}",
                        self.gpu,
                        self.settings,
                        excluded_candidate_ids=excluded_candidates,
                    )
                    changed = True
                except CompileError as exc:
                    state.status = RunStatus.BLOCKED
                    state.message = str(exc)
                    self._save_state(state)
                    self._finish(state, plan)
                    return state
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
        run_id: str,
        attempt_index: int,
        *,
        data_generation: bool,
    ) -> tuple[int | None, bool, Path, Path]:
        event_path = attempt_dir / "episode_events.jsonl"
        log_path = attempt_dir / "stdout.log"
        data_dir = attempt_dir / "data"
        env = os.environ.copy()
        debug_cfg = self.settings.get("debug", {})
        docker_cfg = self.settings.get("execution", {}).get("docker", {})
        env.update(
            {
                "TASK_CONFIG": str(config_path),
                "GPU_ID": str(self.gpu),
                "RANDOM_NUM": str(self.random_num),
                "RANDOM_SEED": str(attempt_index),
                "RUN_NAME": f"agent/{run_id}/attempt_{attempt_index:02d}",
                "OUTPUT_DIR": str(data_dir) if data_generation else "",
                "INTERNDATA_EPISODE_EVENT_PATH": str(event_path),
                "INTERNDATA_RANDOM_SEED": str(attempt_index),
                "INTERNDATA_GPU": str(self.gpu),
                "INTERNDATA_DEBUG_TOPDOWN": "1" if debug_cfg.get("topdown_check") else "0",
                "INTERNDATA_TASK_PATH": str(config_path),
                "INTERNDATA_STACK_ID": f"agent-{run_id}-a{attempt_index:02d}",
                "INTERNDATA_COMPOSE_FILE": str(
                    docker_cfg.get("compose_file", "docker/docker-compose.yml")
                ),
                "INTERNDATA_DOCKER_METADATA_PATH": str(attempt_dir / "docker_runtime.json"),
                "LAUNCH_TEMPLATE": str(
                    docker_cfg.get("launcher_config", "configs/de_plan_with_render_template.yaml")
                ),
                "SIMBOX_DEBUG_OUTPUT_DIR": str(attempt_dir / "simbox_debug"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = ["bash", "scripts/docker/up_simbox_isaac.sh"]
        (attempt_dir / "command.json").write_text(
            json.dumps({"command": command, "env": {key: env[key] for key in sorted(env) if key.startswith("INTERNDATA_") or key in {"TASK_CONFIG", "GPU_ID", "RANDOM_NUM", "RANDOM_SEED", "RUN_NAME", "OUTPUT_DIR", "LAUNCH_TEMPLATE", "SIMBOX_DEBUG_OUTPUT_DIR"}}}, ensure_ascii=False, indent=2)
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
        for path in sorted(run_dir.glob("attempts/*/evidence.json")):
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
