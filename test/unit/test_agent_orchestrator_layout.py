"""Offline integration tests for SceneLayout orchestration state transitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

import agent.orchestrator as orchestrator_module
from agent.compiler import CompileError
from agent.contracts import (
    Diagnosis,
    EvidenceBundle,
    ExecutionVariant,
    ResolutionResponse,
    RunState,
    RunStatus,
    SceneCapabilityManifest,
    TaskPlan,
    dump_contract,
)
from agent.orchestrator import AgentOrchestrator
from agent.tools.scene_layout import SceneLayoutSearchResult
from workflows.simbox.core.robots.profile import load_robot_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG = REPO_ROOT / "workflows/simbox/core/configs/robots/split_aloha.yaml"


def _contracts(tmp_path: Path):
    source_task = tmp_path / "source_task.yaml"
    source_arena = tmp_path / "source_arena.yaml"
    source_task.write_text("tasks: []\n", encoding="utf-8")
    source_arena.write_text("fixtures: []\n", encoding="utf-8")
    profile = load_robot_profile(ROBOT_CONFIG)
    placement_family = str(
        getattr(profile.placement.family, "value", profile.placement.family)
    )
    manifest = SceneCapabilityManifest.from_dict(
        {
            "task_id": "cup_to_tray",
            "scene_id": "scene",
            "source_task": str(source_task),
            "task_class": "Banana",
            "robot_instances": [
                {
                    "instance_name": "robot",
                    "profile_id": profile.profile_id,
                    "robot_config_file": str(ROBOT_CONFIG),
                    "target_class": profile.target_class,
                    "placement_family": placement_family,
                    "available_arms": ["left"],
                    "capabilities": ["pick", "place"],
                    "collision_world_modes": ["physics_schema"],
                    "profile_hash": profile.profile_hash,
                }
            ],
            "objects": [
                {"name": "cup", "category": "cup"},
                {"name": "tray", "category": "tray"},
            ],
        }
    )
    plan = TaskPlan.from_dict(
        {
            "prompt": "put cup in tray",
            "selected_task_id": manifest.task_id,
            "source_task": str(source_task),
            "task_request": {
                "prompt": "put cup in tray",
                "relation": "inside",
                "data_generation": False,
            },
            "robot_requirement": {
                "required_capabilities": ["pick", "place"],
                "decision_basis": "pick and place",
            },
            "subtasks": [
                {
                    "subtask_id": "transfer",
                    "manipulated_object": "cup",
                    "target_object": "tray",
                    "relation": "inside",
                    "arm": "any_single_arm",
                    "stages": [
                        {
                            "stage_id": "pick_place",
                            "objective": "transfer cup",
                            "execution_mode": "single_arm_sequential",
                            "skills": [
                                {
                                    "name": "pick",
                                    "objects": ["cup"],
                                    "arm": "auto",
                                    "decision_basis": "pick cup",
                                },
                                {
                                    "name": "place",
                                    "objects": ["cup", "tray"],
                                    "arm": "auto",
                                    "decision_basis": "place cup",
                                },
                            ],
                        }
                    ],
                }
            ],
            "decision_basis": "single transfer",
        }
    )
    variant = ExecutionVariant(
        variant_id="split_layout",
        instance_name="robot",
        profile_id=profile.profile_id,
        robot_config_file=str(ROBOT_CONFIG),
        placement_family=placement_family,
        profile_hash=profile.profile_hash,
        collision_world_mode="physics_schema",
        arm_binding={"transfer": "left"},
    )
    return source_task, source_arena, manifest, plan, variant


def _layout_result(
    root: Path,
    *,
    revision: str,
    candidate_id: str = "layout_candidate",
    workspace_subtasks: tuple[str, ...] = ("transfer",),
) -> SceneLayoutSearchResult:
    root.mkdir(parents=True, exist_ok=True)
    derived_task = root / "derived_task.yaml"
    derived_arena = root / "derived_arena.yaml"
    mutation_path = root / "scene_mutations.json"
    selection_path = root / "workspace_selection" / "position_selection.json"
    derived_task.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "metadata": {
                            "agent_scene_layout": {"scene_revision": revision}
                        }
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    derived_arena.write_text("fixtures: []\n", encoding="utf-8")
    mutation_path.write_text("{}\n", encoding="utf-8")
    workspace_paths = {}
    for subtask_id in workspace_subtasks:
        workspace_path = root / f"workspace_{subtask_id}.json"
        workspace_path.write_text("{}\n", encoding="utf-8")
        workspace_paths[subtask_id] = str(workspace_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = {"candidate_id": candidate_id, "arm": "left", "seed": 0}
    selection_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    return SceneLayoutSearchResult(
        candidate_id=candidate_id,
        scene_revision=revision,
        scene_task_path=str(derived_task),
        scene_arena_path=str(derived_arena),
        mutation_path=str(mutation_path),
        workspace_paths=workspace_paths,
        workspace_selection_path=str(selection_path),
        workspace_candidate=candidate,
        search_dir=str(root / "search"),
        generations_completed=1,
    )


class _Resolver:
    def __init__(
        self,
        manifest: SceneCapabilityManifest,
        plan: TaskPlan,
        variant: ExecutionVariant,
    ):
        self.manifest = manifest
        self.task_plan = plan
        self.variant = variant
        self.skill_contracts = {}

    def resolve(self, _prompt, _manifests, _output_dir):
        return (
            ResolutionResponse.from_dict(
                {
                    "task_request": self.task_plan.task_request.to_dict(),
                    "resolution": {
                        "mode": "reuse_existing",
                        "selected_task_id": self.manifest.task_id,
                        "selected_source_task": self.manifest.source_task,
                        "selected_scene_id": self.manifest.scene_id,
                        "decision_basis": "test selection",
                    },
                }
            ),
            [self.manifest],
        )

    def select_source_manifest(self, _resolution, _candidates):
        return self.manifest

    def plan(self, _prompt, _response, _manifest, _output_dir):
        return self.task_plan

    def execution_variants(self, _plan):
        return [self.variant]

    def validate_plan(self, _plan, _manifest):
        return None


def _orchestrator(
    tmp_path: Path,
    manifest: SceneCapabilityManifest,
    plan: TaskPlan,
    variant: ExecutionVariant,
    *,
    max_revisions: int,
) -> AgentOrchestrator:
    orchestrator = object.__new__(AgentOrchestrator)
    orchestrator.settings = {
        "generation": {"enabled": False},
        "execution": {"layout_gpus": [0, 1, 2, 3]},
    }
    orchestrator.gpu = 0
    orchestrator.max_revisions = max_revisions
    orchestrator.conda_env = "interndata"
    orchestrator.timeout_sec = 30
    orchestrator.inventory_path = tmp_path / "inventory.json"
    orchestrator.run_root = tmp_path / "runs"
    orchestrator.retain_experience = False
    orchestrator.random_num = 1
    orchestrator.seed = 0
    orchestrator.resolver = _Resolver(manifest, plan, variant)
    orchestrator.retention = SimpleNamespace()
    return orchestrator


def _scene_spec(source_task: Path, source_arena: Path):
    return SimpleNamespace(
        source_task=str(source_task),
        source_arena=str(source_arena),
        source_task_hash="1" * 64,
        source_arena_hash="2" * 64,
        write=lambda path: path.write_text("{}\n", encoding="utf-8"),
    )


def _prepare_execution(
    tmp_path: Path,
    monkeypatch,
    *,
    max_revisions: int,
    persisted_layout: SceneLayoutSearchResult | None = None,
    multi_subtask: bool = False,
):
    source_task, source_arena, manifest, plan, variant = _contracts(tmp_path)
    if multi_subtask:
        manifest_data = manifest.to_dict()
        manifest_data["objects"].append({"name": "spoon", "category": "spoon"})
        manifest = SceneCapabilityManifest.from_dict(manifest_data)
        plan_data = plan.to_dict()
        second = json.loads(json.dumps(plan_data["subtasks"][0]))
        second["subtask_id"] = "spoon_transfer"
        second["manipulated_object"] = "spoon"
        second["stages"][0]["stage_id"] = "spoon_pick_place"
        second["stages"][0]["skills"][0]["objects"] = ["spoon"]
        second["stages"][0]["skills"][1]["objects"] = ["spoon", "tray"]
        plan_data["subtasks"].append(second)
        plan = TaskPlan.from_dict(plan_data)
        variant_data = variant.to_dict()
        variant_data["arm_binding"]["spoon_transfer"] = "left"
        variant = ExecutionVariant.from_dict(variant_data)
    orchestrator = _orchestrator(
        tmp_path,
        manifest,
        plan,
        variant,
        max_revisions=max_revisions,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    plan_path = run_dir / "semantic_task_plan.json"
    manifest_path = run_dir / "selected_manifest.json"
    dump_contract(plan, plan_path)
    dump_contract(manifest, manifest_path)
    (run_dir / "execution_variants.json").write_text(
        json.dumps([variant.to_dict()]) + "\n", encoding="utf-8"
    )
    source_workspace_paths = {}
    for subtask in plan.subtasks:
        workspace_path = run_dir / f"source_workspace_{subtask.subtask_id}.json"
        workspace_path.write_text("{}\n", encoding="utf-8")
        source_workspace_paths[subtask.subtask_id] = str(workspace_path)
    (run_dir / "workspace_manifests.json").write_text(
        json.dumps({variant.variant_id: source_workspace_paths}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "source_snapshot.json").write_text(
        json.dumps({"source_hash": "3" * 64}) + "\n", encoding="utf-8"
    )
    if persisted_layout is not None:
        (run_dir / "scene_layout_results.json").write_text(
            json.dumps({variant.variant_id: persisted_layout.to_dict()}) + "\n",
            encoding="utf-8",
        )
    state = RunState(
        run_id="run",
        prompt=plan.prompt,
        status=RunStatus.PLANNED,
        run_dir=str(run_dir),
        max_revisions=max_revisions,
        task_plan_path=str(plan_path),
        selected_manifest_path=str(manifest_path),
    )
    dump_contract(state, run_dir / "state.json")
    monkeypatch.setattr(
        orchestrator_module,
        "load_scene_spec",
        lambda *_args: _scene_spec(source_task, source_arena),
    )
    monkeypatch.setattr(orchestrator, "_finish", lambda *_args: None)
    return orchestrator, state, plan, variant, source_task, source_arena


def _install_episode_fakes(
    monkeypatch,
    orchestrator: AgentOrchestrator,
    diagnoses: list[Diagnosis],
    compile_calls: list[dict[str, Any]],
    artifact_selection_paths: list[Path],
):
    diagnosis_iter = iter(diagnoses)

    def fake_compile(
        _plan,
        variant,
        _manifest,
        output_path,
        *,
        workspace_candidate=None,
        settings=None,
        scene_task_path=None,
    ):
        revision = "source"
        if scene_task_path is not None:
            scene_document = yaml.safe_load(
                Path(scene_task_path).read_text(encoding="utf-8")
            )
            revision = scene_document["tasks"][0]["metadata"][
                "agent_scene_layout"
            ]["scene_revision"]
        compile_calls.append(
            {
                "candidate_id": workspace_candidate["candidate_id"],
                "scene_task_path": scene_task_path,
                "scene_revision": revision,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(
                {
                    "tasks": [
                        {
                            "metadata": {
                                "agent_plan": {
                                    "execution_variant_id": variant.variant_id,
                                    "robot_profile_id": variant.profile_id,
                                    "robot_profile_hash": variant.profile_hash,
                                    "scene_revision": revision,
                                }
                            }
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return output_path

    def fake_run(_config_path, attempt_dir, _identity, *, data_generation):
        assert data_generation is False
        event_path = attempt_dir / "episode_events.jsonl"
        log_path = attempt_dir / "stdout.log"
        event_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        return 1, False, event_path, log_path

    def fake_collect(attempt_id, attempt_dir, *_args, **_kwargs):
        evidence = EvidenceBundle(
            attempt_id=attempt_id,
            status="success" if len(compile_calls) == len(diagnoses) else "failed",
            task_success=len(compile_calls) == len(diagnoses),
        )
        dump_contract(evidence, attempt_dir / "evidence.json")
        return evidence

    def fake_artifact(_variant_root, _attempt_dir, selection_path, **_kwargs):
        artifact_selection_paths.append(Path(selection_path))
        return SimpleNamespace(complete=True, failure_codes=[])

    monkeypatch.setattr(orchestrator_module, "compile_task_config", fake_compile)
    monkeypatch.setattr(orchestrator_module, "collect_evidence", fake_collect)
    monkeypatch.setattr(
        orchestrator_module, "classify_evidence", lambda _evidence: next(diagnosis_iter)
    )
    monkeypatch.setattr(
        orchestrator_module, "write_variant_artifact_manifest", fake_artifact
    )
    monkeypatch.setattr(orchestrator, "_run_simbox", fake_run)


def _diagnosis(
    failure_code: str,
    *,
    retryable: bool,
    failing_subtask_id: str | None = None,
) -> Diagnosis:
    return Diagnosis(
        stage="workspace",
        failure_code=failure_code,
        failing_subtask_id=failing_subtask_id,
        category="workspace",
        root_cause=failure_code,
        retryable=retryable,
        recommended_action="test repair",
    )


def test_plan_routes_no_geometry_through_typed_layout_search(tmp_path: Path, monkeypatch):
    source_task, source_arena, manifest, plan, variant = _contracts(tmp_path)
    orchestrator = _orchestrator(
        tmp_path, manifest, plan, variant, max_revisions=1
    )
    result = _layout_result(tmp_path / "layout", revision="layout_initial")
    search_calls = []

    monkeypatch.setattr(orchestrator_module, "load_or_build_inventory", lambda *_args: [manifest])
    monkeypatch.setattr(
        orchestrator_module,
        "load_scene_spec",
        lambda *_args: _scene_spec(source_task, source_arena),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_source_snapshot",
        lambda *_args, **_kwargs: {"source_hash": "3" * 64},
    )
    monkeypatch.setattr(
        orchestrator_module,
        "write_source_snapshot",
        lambda value, path: path.write_text(json.dumps(value) + "\n", encoding="utf-8"),
    )

    def fake_compile(_plan, _variant, _manifest, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("tasks: []\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(orchestrator_module, "compile_task_config", fake_compile)
    monkeypatch.setattr(
        orchestrator_module,
        "generate_workspace_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CompileError("NO_GEOMETRY_CANDIDATE: no valid base pose")
        ),
    )

    def fake_search(*args, **kwargs):
        search_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(orchestrator, "_search_layout", fake_search)

    state = orchestrator.plan(plan.prompt, run_id="layout_plan")

    assert state.status == RunStatus.PLANNED
    assert len(search_calls) == 1
    assert search_calls[0][0][5] == "NO_GEOMETRY_CANDIDATE"
    assert search_calls[0][1] == {"subtask_id": "transfer"}
    persisted = json.loads(
        (Path(state.run_dir) / "scene_layout_results.json").read_text(encoding="utf-8")
    )
    assert persisted == {variant.variant_id: result.to_dict()}
    workspace_paths = json.loads(
        (Path(state.run_dir) / "workspace_manifests.json").read_text(encoding="utf-8")
    )
    assert workspace_paths[variant.variant_id] == result.workspace_paths


def test_execute_consumes_persisted_layout_paths_without_source_selection(
    tmp_path: Path,
    monkeypatch,
):
    layout = _layout_result(tmp_path / "persisted_layout", revision="layout_persisted")
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=0,
        persisted_layout=layout,
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [_diagnosis("NONE", retryable=False)],
        compile_calls,
        artifact_paths,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "select_task_workspace_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("persisted layout must bypass source workspace selection")
        ),
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert compile_calls == [
        {
            "candidate_id": layout.candidate_id,
            "scene_task_path": Path(layout.scene_task_path),
            "scene_revision": "layout_persisted",
        }
    ]
    assert artifact_paths == [Path(layout.workspace_selection_path)]
    assert result.workspace_manifest_path == layout.workspace_selection_path


def test_mutate_layout_feedback_advances_next_attempt_to_derived_revision(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
    )
    initial_candidate = {"candidate_id": "source_candidate", "arm": "left", "seed": 0}

    def fake_select(*args, **_kwargs):
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(initial_candidate) + "\n", encoding="utf-8"
        )
        return initial_candidate

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    layout = _layout_result(tmp_path / "feedback_layout", revision="layout_feedback")
    search_calls = []
    monkeypatch.setattr(
        orchestrator,
        "_search_layout",
        lambda *args, **kwargs: search_calls.append((args, kwargs)) or layout,
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis("NO_GEOMETRY_CANDIDATE", retryable=True),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert len(search_calls) == 1
    assert search_calls[0][0][5] == "NO_GEOMETRY_CANDIDATE"
    assert [item["scene_task_path"] for item in compile_calls] == [
        None,
        Path(layout.scene_task_path),
    ]
    assert [item["scene_revision"] for item in compile_calls] == [
        "source",
        "layout_feedback",
    ]
    assert artifact_paths[-1] == Path(layout.workspace_selection_path)
    persisted = json.loads(
        (Path(state.run_dir) / "scene_layout_results.json").read_text(encoding="utf-8")
    )
    assert persisted["split_layout"]["scene_revision"] == "layout_feedback"


def test_attempt_manifest_finalizes_an_immutable_attempt_local_trace(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
    )
    initial_candidate = {"candidate_id": "source_candidate", "arm": "left", "seed": 0}

    def fake_select(*args, **_kwargs):
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(initial_candidate) + "\n", encoding="utf-8"
        )
        return initial_candidate

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    layout = _layout_result(tmp_path / "feedback_layout", revision="layout_feedback")
    monkeypatch.setattr(orchestrator, "_search_layout", lambda *_args, **_kwargs: layout)
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis("NO_GEOMETRY_CANDIDATE", retryable=True),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )
    finalized: list[tuple[Path, str, list[dict[str, Any]]]] = []

    def capture_manifest(_variant_root, attempt_dir, _selection_path, **_kwargs):
        trace_path = Path(attempt_dir) / "trace.jsonl"
        trace_bytes = trace_path.read_bytes()
        records = [json.loads(line) for line in trace_bytes.splitlines()]
        finalized.append(
            (trace_path, hashlib.sha256(trace_bytes).hexdigest(), records)
        )
        return SimpleNamespace(complete=True, failure_codes=[])

    monkeypatch.setattr(
        orchestrator_module, "write_variant_artifact_manifest", capture_manifest
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert [[record["stage"] for record in records] for _, _, records in finalized] == [
        ["episode_evaluation", "feedback"],
        ["episode_evaluation"],
    ]
    assert [
        {record["attempt_id"] for record in records} for _, _, records in finalized
    ] == [{"00"}, {"01"}]
    for trace_path, frozen_sha256, _records in finalized:
        assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == frozen_sha256
    variant_root = Path(state.run_dir) / "variants" / "split_layout"
    assert not (variant_root / "trace.jsonl").exists()


def test_replan_updates_exact_artifact_selection_path(tmp_path: Path, monkeypatch):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
    )
    selection_dirs: list[Path] = []

    def fake_select(*args, **kwargs):
        output_dir = Path(args[4])
        selection_dirs.append(output_dir)
        candidate_id = "source_candidate" if len(selection_dirs) == 1 else "replanned_candidate"
        candidate = {"candidate_id": candidate_id, "arm": "left", "seed": 0}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(candidate) + "\n", encoding="utf-8"
        )
        if len(selection_dirs) == 2:
            assert kwargs["excluded_candidate_ids"] == {"source_candidate"}
        return candidate

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis("NO_CUROBO_CANDIDATE", retryable=True),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert [item["candidate_id"] for item in compile_calls] == [
        "source_candidate",
        "replanned_candidate",
    ]
    assert artifact_paths == [
        selection_dirs[0] / "position_selection.json",
        selection_dirs[1] / "position_selection.json",
    ]
    assert Path(result.workspace_manifest_path) == artifact_paths[-1]


def test_multi_subtask_feedback_mutates_only_the_attributed_failure(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
        multi_subtask=True,
    )
    initial_candidate = {
        "candidate_id": "common_000",
        "world_xy": [-0.6, 0.0],
        "yaw_deg": 0.0,
    }

    def fake_select(*args, **_kwargs):
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(initial_candidate) + "\n", encoding="utf-8"
        )
        return initial_candidate

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    layout = _layout_result(
        tmp_path / "multi_feedback_layout",
        revision="layout_spoon",
        workspace_subtasks=("transfer", "spoon_transfer"),
    )
    search_calls = []
    monkeypatch.setattr(
        orchestrator,
        "_search_layout",
        lambda *args, **kwargs: search_calls.append((args, kwargs)) or layout,
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis(
                "NO_GEOMETRY_CANDIDATE",
                retryable=True,
                failing_subtask_id="spoon_transfer",
            ),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert len(search_calls) == 1
    assert search_calls[0][1] == {"subtask_id": "spoon_transfer"}
    repair = json.loads(
        (
            Path(state.run_dir)
            / "variants"
            / "split_layout"
            / "attempts"
            / "00"
            / "repair.json"
        ).read_text(encoding="utf-8")
    )
    assert repair["failing_subtask_id"] == "spoon_transfer"


def test_multi_subtask_feedback_blocks_without_failure_attribution(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
        multi_subtask=True,
    )
    initial_candidate = {
        "candidate_id": "common_000",
        "world_xy": [-0.6, 0.0],
        "yaw_deg": 0.0,
    }

    def fake_select(*args, **_kwargs):
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "position_selection.json").write_text(
            json.dumps(initial_candidate) + "\n", encoding="utf-8"
        )
        return initial_candidate

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    monkeypatch.setattr(
        orchestrator,
        "_search_layout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unattributed multi-subtask failure must not mutate layout")
        ),
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis("NO_GEOMETRY_CANDIDATE", retryable=True),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.BLOCKED
    attempt = (
        Path(state.run_dir)
        / "variants"
        / "split_layout"
        / "attempts"
        / "00"
    )
    diagnosis = json.loads((attempt / "diagnosis.json").read_text(encoding="utf-8"))
    repair = json.loads((attempt / "repair.json").read_text(encoding="utf-8"))
    assert diagnosis["failure_code"] == "FAILURE_SUBTASK_UNATTRIBUTED"
    assert diagnosis["workspace_action"] == "block"
    assert repair["action"] == "block"


def test_multi_subtask_selection_failure_routes_typed_probe_attribution(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=0,
        multi_subtask=True,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "select_task_workspace_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CompileError(
                "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE: spoon probe failed",
                failing_subtask_id="spoon_transfer",
            )
        ),
    )
    layout = _layout_result(
        tmp_path / "multi_selection_layout",
        revision="layout_selection_spoon",
        workspace_subtasks=("transfer", "spoon_transfer"),
    )
    search_calls = []
    monkeypatch.setattr(
        orchestrator,
        "_search_layout",
        lambda *args, **kwargs: search_calls.append((args, kwargs)) or layout,
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [_diagnosis("NONE", retryable=False)],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert len(search_calls) == 1
    assert search_calls[0][1] == {"subtask_id": "spoon_transfer"}


def test_multi_subtask_replan_failure_routes_typed_probe_attribution(
    tmp_path: Path,
    monkeypatch,
):
    orchestrator, state, _plan, _variant, _source_task, _source_arena = _prepare_execution(
        tmp_path,
        monkeypatch,
        max_revisions=1,
        multi_subtask=True,
    )
    initial_candidate = {
        "candidate_id": "common_000",
        "world_xy": [-0.6, 0.0],
        "yaw_deg": 0.0,
    }
    selection_calls = 0

    def fake_select(*args, **_kwargs):
        nonlocal selection_calls
        selection_calls += 1
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        if selection_calls == 1:
            (output_dir / "position_selection.json").write_text(
                json.dumps(initial_candidate) + "\n", encoding="utf-8"
            )
            return initial_candidate
        raise CompileError(
            "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE: spoon probe failed",
            failing_subtask_id="spoon_transfer",
        )

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    layout = _layout_result(
        tmp_path / "multi_replan_layout",
        revision="layout_replan_spoon",
        workspace_subtasks=("transfer", "spoon_transfer"),
    )
    search_calls = []
    monkeypatch.setattr(
        orchestrator,
        "_search_layout",
        lambda *args, **kwargs: search_calls.append((args, kwargs)) or layout,
    )
    compile_calls: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    _install_episode_fakes(
        monkeypatch,
        orchestrator,
        [
            _diagnosis("NO_CUROBO_CANDIDATE", retryable=True),
            _diagnosis("NONE", retryable=False),
        ],
        compile_calls,
        artifact_paths,
    )

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert selection_calls == 2
    assert len(search_calls) == 1
    assert search_calls[0][1] == {"subtask_id": "spoon_transfer"}
