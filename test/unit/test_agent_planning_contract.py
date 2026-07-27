"""Offline tests for Agent planning modes, Skill values and YAML compilation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import agent.compiler as compiler_module
import agent.orchestrator as orchestrator_module
from agent.compiler import (
    CompileError,
    compile_task_config,
    rank_common_workspace_candidates,
    select_task_workspace_candidate,
)
from agent.contracts import RunState, RunStatus, SceneCapabilityManifest, TaskPlan, dump_contract
from agent.orchestrator import AgentOrchestrator
from agent.resolver import AgentDecisionError, TaskResolver, load_skill_contracts
from agent.settings import load_agent_settings


def _source_task(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "robots": [{"name": "split_aloha"}],
                        "skills": [],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _manifest(source: Path) -> SceneCapabilityManifest:
    return SceneCapabilityManifest.from_dict(
        {
            "task_id": "cup_to_tray",
            "scene_id": "kitchen",
            "source_task": str(source),
            "task_class": "Banana",
            "robots": ["split_aloha"],
            "objects": [
                {"name": "cup", "category": "cup"},
                {"name": "tray", "category": "tray"},
            ],
            "container_regions": [
                {"object": "tray", "region": "tray_interior", "can_receive_objects": True}
            ],
        }
    )


def _plan(source: Path, *, params=None, execution_mode="single_arm_sequential") -> TaskPlan:
    return TaskPlan.from_dict(
        {
            "prompt": "把杯子放进托盘",
            "selected_task_id": "cup_to_tray",
            "source_task": str(source),
            "task_request": {
                "prompt": "把杯子放进托盘",
                "relation": "inside",
            },
            "robot": {
                "robot_type": "split_aloha",
                "robot_profile": "split_aloha_tabletop_v1",
                "decision_basis": "selected task provides split_aloha",
            },
            "subtasks": [
                {
                    "subtask_id": "cup_transfer",
                    "manipulated_object": "cup",
                    "target_object": "tray",
                    "relation": "inside",
                    "arm": "left",
                    "stages": [
                        {
                            "stage_id": "pick_place",
                            "objective": "transfer cup",
                            "execution_mode": execution_mode,
                            "skills": [
                                {
                                    "name": "pick",
                                    "objects": ["cup"],
                                    "arm": "left",
                                    "params": {},
                                    "decision_basis": "cup is manipulated",
                                },
                                {
                                    "name": "place",
                                    "objects": ["cup", "tray"],
                                    "arm": "left",
                                    "params": params or {},
                                    "decision_basis": "tray is target",
                                },
                            ],
                        }
                    ],
                }
            ],
            "decision_basis": "existing task matches",
        }
    )


def _resolver() -> TaskResolver:
    return TaskResolver(backend=object(), skill_contracts=load_skill_contracts())


class _CaptureBackend:
    def __init__(self, result):
        self.result = result
        self.prompt = ""

    def generate(self, _model, prompt, _artifact_dir, _stem):
        self.prompt = prompt
        return self.result


def test_valid_pick_place_plan_and_deterministic_defaults(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    plan = _plan(source)

    _resolver().validate_plan(plan, manifest)
    output = compile_task_config(plan, manifest, tmp_path / "compiled.yaml")
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]
    phase = task["skills"][0]["split_aloha"][0]

    assert [skill["name"] for skill in phase["left"]] == ["pick", "place"]
    assert phase["right"] == []
    assert phase["left"][1]["position_constraint"] == "object"
    assert phase["left"][1]["success_mode"] == "xybbox"
    assert phase["left"][1]["test_mode"] == "forward"


def test_skill_defaults_are_loaded_from_agent_config_and_agent_params_override_them(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    plan = _plan(source, params={"place_z_offset": 0.07})
    settings = load_agent_settings()
    settings["skill_defaults"]["pick"]["pre_grasp_offset"] = 0.23

    output = compile_task_config(
        plan,
        manifest,
        tmp_path / "compiled.yaml",
        settings=settings,
    )
    phase = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]["skills"][0]["split_aloha"][0]

    assert phase["left"][0]["pre_grasp_offset"] == pytest.approx(0.23)
    assert phase["left"][1]["place_z_offset"] == pytest.approx(0.07)


def test_data_generation_uses_config_default_and_keeps_explicit_user_override(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    default_plan = _plan(source)

    default_output = compile_task_config(default_plan, manifest, tmp_path / "default.yaml")
    default_metadata = yaml.safe_load(default_output.read_text(encoding="utf-8"))["tasks"][0]["metadata"]
    assert default_plan.task_request.data_generation is None
    assert default_metadata["agent_plan"]["data_generation"] is True

    override_data = default_plan.to_dict()
    override_data["task_request"]["data_generation"] = False
    override_plan = TaskPlan.from_dict(override_data)
    override_output = compile_task_config(override_plan, manifest, tmp_path / "override.yaml")
    override_metadata = yaml.safe_load(override_output.read_text(encoding="utf-8"))["tasks"][0]["metadata"]
    assert override_metadata["agent_plan"]["data_generation"] is False


def test_planning_agent_receives_the_stage_spec_and_typed_skill_contracts(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    backend = _CaptureBackend(_plan(source))
    resolver = TaskResolver(backend=backend, skill_contracts=load_skill_contracts())

    resolver.plan("把杯子放进托盘", None, manifest, tmp_path / "decisions")

    assert "Agent 任务规划与 Skill 编排规范" in backend.prompt
    assert "Agent 中心物品选择与机器人初始位姿生成规范" in backend.prompt
    assert '"enabled": true' in backend.prompt
    assert '"default_arm": "right"' in backend.prompt
    assert "single_arm_sequential" in backend.prompt
    assert '"allowed_values"' in backend.prompt
    assert "inside 不是合法值" in backend.prompt


def test_planning_spec_lists_every_exposed_pick_and_place_parameter():
    spec = (
        Path(__file__).resolve().parents[2]
        / "agent"
        / "prompts"
        / "Agent任务规划与Skill编排规范.md"
    ).read_text(encoding="utf-8")
    for contract in load_skill_contracts().values():
        for parameter_name in contract.parameters:
            assert f"`{parameter_name}`" in spec


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"success_mode": "inside"}, "invalid value for place.success_mode"),
        (
            {"position_constraint": "tray_interior"},
            "invalid value for place.position_constraint",
        ),
        ({"test_mode": "forward"}, "owned by compiler"),
        ({"x_ratio_range": [0.8, 0.2]}, "must be ordered"),
    ],
)
def test_invalid_place_parameter_values_are_rejected(tmp_path, params, message):
    source = _source_task(tmp_path / "source.yaml")
    with pytest.raises(AgentDecisionError, match=message):
        _resolver().validate_plan(_plan(source, params=params), _manifest(source))


def test_single_arm_mode_must_match_skill_count(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    with pytest.raises(AgentDecisionError, match="requires exactly one Skill"):
        _resolver().validate_plan(
            _plan(source, execution_mode="single_arm_single_skill"),
            _manifest(source),
        )


def test_dual_arm_simultaneous_keeps_a_disabled_capability_slot(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    stage = plan_data["subtasks"][0]["stages"][0]
    plan_data["subtasks"][0]["arm"] = "both"
    stage["execution_mode"] = "dual_arm_simultaneous"
    stage["skills"][1]["arm"] = "right"
    plan = TaskPlan.from_dict(plan_data)

    with pytest.raises(CompileError, match="not enabled"):
        compile_task_config(plan, _manifest(source), tmp_path / "compiled.yaml")


def test_both_arm_subtask_must_be_marked_unresolved_before_workspace(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"][0]["arm"] = "both"
    stage = plan_data["subtasks"][0]["stages"][0]
    stage["execution_mode"] = "dual_arm_simultaneous"
    stage["skills"][0]["arm"] = "left"
    stage["skills"][1]["arm"] = "right"

    with pytest.raises(AgentDecisionError, match="record it as unresolved"):
        _resolver().validate_plan(TaskPlan.from_dict(plan_data), _manifest(source))


def test_subtask_arm_is_decided_before_workspace_and_skills_must_match(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"][0]["arm"] = "right"
    plan = TaskPlan.from_dict(plan_data)

    with pytest.raises(AgentDecisionError, match="two ordered Skills on one arm"):
        _resolver().validate_plan(plan, _manifest(source))


def test_multiple_center_objects_compile_into_one_yaml_in_subtask_order(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"].append(
        {
            "subtask_id": "spoon_transfer",
            "manipulated_object": "spoon",
            "target_object": "tray",
            "relation": "inside",
            "arm": "right",
            "stages": [
                {
                    "stage_id": "pick_place_spoon",
                    "objective": "transfer spoon",
                    "execution_mode": "single_arm_sequential",
                    "skills": [
                        {
                            "name": "pick",
                            "objects": ["spoon"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "spoon is manipulated",
                        },
                        {
                            "name": "place",
                            "objects": ["spoon", "tray"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "tray is target",
                        },
                    ],
                }
            ],
        }
    )
    manifest_data = _manifest(source).to_dict()
    manifest_data["objects"].append({"name": "spoon", "category": "spoon"})
    manifest = SceneCapabilityManifest.from_dict(manifest_data)
    plan = TaskPlan.from_dict(plan_data)

    _resolver().validate_plan(plan, manifest)
    output = compile_task_config(plan, manifest, tmp_path / "compiled.yaml")
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]

    assert len(task["skills"]) == 2
    cup_phase = task["skills"][0]["split_aloha"][0]
    spoon_phase = task["skills"][1]["split_aloha"][0]
    assert [item["objects"] for item in cup_phase["left"]] == [["cup"], ["cup", "tray"]]
    assert cup_phase["right"] == []
    assert spoon_phase["left"] == []
    assert [item["objects"] for item in spoon_phase["right"]] == [
        ["spoon"],
        ["spoon", "tray"],
    ]


def test_common_workspace_candidates_cross_score_the_same_base_pose(tmp_path):
    paths = {}
    for subtask_id, target_xy, arm in (
        ("cup_transfer", [0.0, 0.0], "left"),
        ("spoon_transfer", [0.2, 0.0], "right"),
    ):
        path = tmp_path / subtask_id / "candidates.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "target": {"name": subtask_id, "world_xyz": [*target_xy, 0.8]},
                    "required_arm": arm,
                    "sampling": {
                        "min_radius_m": 0.4,
                        "max_radius_m": 1.1,
                        "preferred_radius_m": 0.7,
                    },
                    "geometry_candidates": [
                        {
                            "candidate_id": "annulus_000",
                            "world_xy": [-0.6, 0.0],
                            "yaw_deg": 0.0,
                            "radius_m": 0.6,
                            "angle_deg": 180.0,
                            "geometry_feasible": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths[subtask_id] = path

    candidates = rank_common_workspace_candidates(paths, max_heading_error_deg=20.0, limit=4)

    assert len(candidates) == 1
    assert candidates[0]["world_xy"] == [-0.6, 0.0]
    assert candidates[0]["common_metrics"]["cup_transfer"]["required_arm"] == "left"
    assert candidates[0]["common_metrics"]["spoon_transfer"]["required_arm"] == "right"


def test_common_workspace_selection_probes_same_pose_with_each_preselected_arm(
    monkeypatch,
    tmp_path,
):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"].append(
        {
            "subtask_id": "spoon_transfer",
            "manipulated_object": "spoon",
            "target_object": "tray",
            "relation": "inside",
            "arm": "right",
            "stages": [
                {
                    "stage_id": "pick_place_spoon",
                    "objective": "transfer spoon",
                    "execution_mode": "single_arm_sequential",
                    "skills": [
                        {
                            "name": "pick",
                            "objects": ["spoon"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "spoon is manipulated",
                        },
                        {
                            "name": "place",
                            "objects": ["spoon", "tray"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "tray is target",
                        },
                    ],
                }
            ],
        }
    )
    plan = TaskPlan.from_dict(plan_data)
    manifest_data = _manifest(source).to_dict()
    manifest_data["objects"][0]["attach_prim_path_children"] = ["/World/cup/collision"]
    manifest_data["objects"].append(
        {
            "name": "spoon",
            "category": "spoon",
            "attach_prim_path_children": ["/World/spoon/collision"],
        }
    )
    manifest = SceneCapabilityManifest.from_dict(manifest_data)
    workspace_paths = {}
    for subtask_id, target_name, target_xy, arm in (
        ("cup_transfer", "cup", [0.0, 0.0], "left"),
        ("spoon_transfer", "spoon", [0.2, 0.0], "right"),
    ):
        path = tmp_path / subtask_id / "candidates.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "target": {"name": target_name, "world_xyz": [*target_xy, 0.8]},
                    "required_arm": arm,
                    "sampling": {
                        "min_radius_m": 0.4,
                        "max_radius_m": 1.1,
                        "preferred_radius_m": 0.7,
                    },
                    "geometry_candidates": [
                        {
                            "candidate_id": "annulus_000",
                            "world_xy": [-0.6, 0.0],
                            "yaw_deg": 0.0,
                            "geometry_feasible": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        workspace_paths[subtask_id] = path

    probes = []

    def fake_validate(path, _gpu, _conda_env, **kwargs):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        local = value["geometry_candidates"][0]
        probes.append(
            {
                "arm": kwargs["arm"],
                "target": value["target"]["name"],
                "world_xy": local["world_xy"],
                "attach": kwargs["attach_prim_path_children"],
            }
        )
        return {**local, "arm": kwargs["arm"]}

    monkeypatch.setattr(compiler_module, "validate_workspace_manifest", fake_validate)
    selected = select_task_workspace_candidate(
        plan,
        workspace_paths,
        manifest,
        tmp_path / "selection",
        gpu=0,
    )

    assert selected["candidate_id"] == "common_000"
    assert [(item["target"], item["arm"]) for item in probes] == [
        ("cup", "left"),
        ("spoon", "right"),
    ]
    assert {tuple(item["world_xy"]) for item in probes} == {(-0.6, 0.0)}
    assert probes[0]["attach"] == ["/World/cup/collision"]
    assert probes[1]["attach"] == ["/World/spoon/collision"]
    selection = json.loads((tmp_path / "selection" / "position_selection.json").read_text())
    assert selection["mode"] == "common"


def test_orchestrator_compiles_and_runs_all_subtasks_once(monkeypatch, tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest_data = _manifest(source).to_dict()
    manifest_data["objects"].append({"name": "spoon", "category": "spoon"})
    manifest = SceneCapabilityManifest.from_dict(manifest_data)
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"].append(
        {
            "subtask_id": "spoon_transfer",
            "manipulated_object": "spoon",
            "target_object": "tray",
            "relation": "inside",
            "arm": "right",
            "stages": [
                {
                    "stage_id": "pick_place_spoon",
                    "objective": "transfer spoon",
                    "execution_mode": "single_arm_sequential",
                    "skills": [
                        {
                            "name": "pick",
                            "objects": ["spoon"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "spoon is manipulated",
                        },
                        {
                            "name": "place",
                            "objects": ["spoon", "tray"],
                            "arm": "auto",
                            "params": {},
                            "decision_basis": "tray is target",
                        },
                    ],
                }
            ],
        }
    )
    plan = TaskPlan.from_dict(plan_data)
    run_dir = tmp_path / "runs" / "multi"
    run_dir.mkdir(parents=True)
    plan_path = run_dir / "task_plan.json"
    dump_contract(plan, plan_path)
    (run_dir / "workspace_manifests.json").write_text(
        json.dumps(
            {
                "cup_transfer": str(run_dir / "cup.json"),
                "spoon_transfer": str(run_dir / "spoon.json"),
            }
        ),
        encoding="utf-8",
    )
    state = RunState(
        run_id="multi",
        prompt=plan.prompt,
        status=RunStatus.PLANNED,
        run_dir=str(run_dir),
        task_plan_path=str(plan_path),
    )
    calls = {"select": 0, "compile": 0, "run": 0}

    monkeypatch.setattr(orchestrator_module, "load_or_build_inventory", lambda _path: [manifest])

    def fake_select(*_args, **_kwargs):
        calls["select"] += 1
        return {"candidate_id": "common_000", "world_xy": [-0.6, 0.0], "yaw_deg": 0.0}

    def fake_compile(compiled_plan, _manifest, output_path, **_kwargs):
        calls["compile"] += 1
        assert [item.subtask_id for item in compiled_plan.subtasks] == [
            "cup_transfer",
            "spoon_transfer",
        ]
        output_path.write_text("tasks: []\n", encoding="utf-8")
        return output_path

    def fake_run(_config_path, attempt_dir, _run_id, _attempt_index, *, data_generation):
        calls["run"] += 1
        assert data_generation is True
        event_path = attempt_dir / "episode_events.jsonl"
        log_path = attempt_dir / "stdout.log"
        event_path.write_text('{"status":"success"}\n', encoding="utf-8")
        log_path.write_text("Task is successful\n", encoding="utf-8")
        return 0, False, event_path, log_path

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    monkeypatch.setattr(orchestrator_module, "compile_task_config", fake_compile)
    orchestrator = AgentOrchestrator(
        run_root=tmp_path / "runs",
        inventory_path=tmp_path / "inventory.json",
        retain_experience=False,
        settings=load_agent_settings(),
    )
    monkeypatch.setattr(orchestrator, "_run_simbox", fake_run)

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert result.current_subtask == 2
    assert calls == {"select": 1, "compile": 1, "run": 1}
    assert (run_dir / "attempts" / "00" / "task.yaml").is_file()
