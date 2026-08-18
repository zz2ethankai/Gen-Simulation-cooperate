"""Offline tests for Agent planning modes, Skill values and YAML compilation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import agent.compiler as compiler_module
import agent.orchestrator as orchestrator_module
from agent.compiler import (
    CompileError,
    _compile_profile_cameras,
    compile_task_config,
    rank_common_workspace_candidates,
    select_task_workspace_candidate,
    validate_workspace_manifest,
)
from agent.contracts import (
    ExecutionVariant,
    ResolutionDecision,
    RunState,
    RunStatus,
    SceneCapabilityManifest,
    TaskPlan,
    dump_contract,
)
from agent.orchestrator import AgentOrchestrator
from agent.resolver import AgentDecisionError, TaskResolver, load_skill_contracts
from agent.settings import load_agent_settings
from agent.tools.source_integrity import (
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
    SourceMember,
    canonical_source_hash,
)
from workflows.simbox.core.robots.profile import load_robot_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def _robot_instance(profile_name: str, instance_name: str) -> dict:
    config_file = f"workflows/simbox/core/configs/robots/{profile_name}.yaml"
    profile = load_robot_profile(REPO_ROOT / config_file)
    return {
        "instance_name": instance_name,
        "profile_id": profile.profile_id,
        "robot_config_file": config_file,
        "target_class": profile.target_class,
        "placement_family": str(
            getattr(profile.placement.family, "value", profile.placement.family)
        ),
        "available_arms": sorted(profile.arms),
        "capabilities": sorted(profile.capabilities),
        "collision_world_modes": sorted(profile.collision_world_modes),
        "profile_hash": profile.profile_hash,
    }


def _variant(
    profile_name: str = "split_aloha",
    instance_name: str = "split_aloha",
    arm_binding: dict[str, str] | None = None,
) -> ExecutionVariant:
    config_file = f"workflows/simbox/core/configs/robots/{profile_name}.yaml"
    profile = load_robot_profile(REPO_ROOT / config_file)
    return ExecutionVariant(
        variant_id=f"{profile.profile_id}__test",
        instance_name=instance_name,
        profile_id=profile.profile_id,
        robot_config_file=config_file,
        placement_family=str(
            getattr(profile.placement.family, "value", profile.placement.family)
        ),
        profile_hash=profile.profile_hash,
        collision_world_mode="physics_schema",
        arm_binding=arm_binding or {"cup_transfer": "left"},
    )


def _source_task(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "robots": [
                            {
                                "name": "split_aloha",
                                "robot_config_file": (
                                    "workflows/simbox/core/configs/robots/split_aloha.yaml"
                                ),
                            }
                        ],
                        "skills": [],
                        "container_regions": [
                            {
                                "name": "tray_interior",
                                "object": "tray",
                                "can_receive_objects": True,
                                "center": [0.25, -0.1],
                                "inner_size": [0.4, 0.3],
                                "interior_support_z": 0.7,
                            }
                        ],
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
            "robot_instances": [_robot_instance("split_aloha", "split_aloha")],
            "objects": [
                {"name": "cup", "category": "cup"},
                {"name": "tray", "category": "tray"},
            ],
            "container_regions": [
                {
                    "name": "tray_interior",
                    "object": "tray",
                    "can_receive_objects": True,
                    "center": [0.25, -0.1],
                    "inner_size": [0.4, 0.3],
                    "interior_support_z": 0.7,
                }
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
            "robot_requirement": {
                "required_capabilities": ["pick", "place"],
                "preferred_profile_ids": [],
                "decision_basis": "task requires pick and place",
            },
            "subtasks": [
                {
                    "subtask_id": "cup_transfer",
                    "manipulated_object": "cup",
                    "target_object": "tray",
                    "relation": "inside",
                    "arm": "any_single_arm",
                    "stages": [
                        {
                            "stage_id": "pick_place",
                            "objective": "transfer cup",
                            "execution_mode": execution_mode,
                            "skills": [
                                {
                                    "name": "pick",
                                    "objects": ["cup"],
                                    "arm": "auto",
                                    "params": {},
                                    "decision_basis": "cup is manipulated",
                                },
                                {
                                    "name": "place",
                                    "objects": ["cup", "tray"],
                                    "arm": "auto",
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
    output = compile_task_config(plan, _variant(), manifest, tmp_path / "compiled.yaml")
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]
    phase = task["skills"][0]["split_aloha"][0]

    assert [skill["name"] for skill in phase["left"]] == ["pick", "place"]
    assert [skill["agent_subtask_id"] for skill in phase["left"]] == [
        "cup_transfer",
        "cup_transfer",
    ]
    assert phase["right"] == []
    assert phase["left"][1]["position_constraint"] == "object"
    assert phase["left"][1]["success_mode"] == "relation_predicate"
    assert phase["left"][1]["semantic_relation"] == "inside"
    assert phase["left"][1]["relation_predicate"] == {
        "relation": "inside",
        "geometry_tolerance_m": 0.002,
        "support_gap_tolerance_m": 0.006,
        "minimum_support_contact_n": 0.0,
        "max_unexpected_contact_n": 5.0,
        "container_region": {
            "name": "tray_interior",
            "center": [0.25, -0.1],
            "inner_size": [0.4, 0.3],
            "interior_support_z": 0.7,
        },
    }
    assert phase["left"][1]["test_mode"] == "forward"
    assert task["metadata"]["agent_plan"]["subtasks"] == [
        {
            "subtask_id": "cup_transfer",
            "center_object": "cup",
            "target_object": "tray",
            "relation": "inside",
            "arm_constraint": "any_single_arm",
            "arm": "left",
        }
    ]


@pytest.mark.parametrize("relation", ["on", "inside", "insert"])
def test_strict_relation_requires_complete_pick_place_chain(tmp_path, relation):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = relation
    subtask = plan_data["subtasks"][0]
    subtask["relation"] = relation
    stage = subtask["stages"][0]
    stage["execution_mode"] = "single_arm_single_skill"
    stage["skills"] = stage["skills"][:1]
    plan_data["robot_requirement"]["required_capabilities"] = ["pick"]

    with pytest.raises(
        AgentDecisionError,
        match="requires one same-arm Pick followed by Place",
    ):
        _resolver().validate_plan(TaskPlan.from_dict(plan_data), _manifest(source))


def test_resolver_rejects_insert_without_an_executable_insertion_contract(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = "insert"
    plan_data["subtasks"][0]["relation"] = "insert"
    plan_data["subtasks"][0]["stages"][0]["skills"][1]["params"] = {
        "place_direction": "horizontal",
        "align_place_obj_axis": [0.0, 0.0, 1.0],
        "offset_place_obj_axis": [1.0, 0.0, 0.0],
    }

    with pytest.raises(
        AgentDecisionError,
        match="RELATION_INSERT_NOT_ADMITTED.*insertion axis, minimum depth",
    ):
        _resolver().validate_plan(TaskPlan.from_dict(plan_data), _manifest(source))


@pytest.mark.parametrize("relation", ["left_of", "right_of", "next_to", "hang", "none"])
def test_resolver_rejects_relations_without_terminal_predicates(tmp_path, relation):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = relation
    plan_data["subtasks"][0]["relation"] = relation
    if relation == "hang":
        plan_data["subtasks"][0]["stages"][0]["skills"][1]["params"] = {
            "place_direction": "horizontal",
            "align_place_obj_axis": [0.0, 0.0, 1.0],
            "offset_place_obj_axis": [1.0, 0.0, 0.0],
        }

    with pytest.raises(
        AgentDecisionError,
        match="RELATION_NOT_ADMITTED.*only 'on' and 'inside'",
    ):
        _resolver().validate_plan(TaskPlan.from_dict(plan_data), _manifest(source))


def test_compiler_rejects_insert_when_resolver_is_bypassed(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = "insert"
    plan_data["subtasks"][0]["relation"] = "insert"

    with pytest.raises(CompileError, match="RELATION_INSERT_NOT_ADMITTED") as caught:
        compile_task_config(
            TaskPlan.from_dict(plan_data),
            _variant(),
            _manifest(source),
            tmp_path / "compiled.yaml",
        )

    assert caught.value.failing_subtask_id == "cup_transfer"


def test_compiler_rejects_unproven_relation_when_resolver_is_bypassed(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = "left_of"
    plan_data["subtasks"][0]["relation"] = "left_of"

    with pytest.raises(CompileError, match="RELATION_NOT_ADMITTED") as caught:
        compile_task_config(
            TaskPlan.from_dict(plan_data),
            _variant(),
            _manifest(source),
            tmp_path / "compiled.yaml",
        )

    assert caught.value.failing_subtask_id == "cup_transfer"


@pytest.mark.parametrize(
    ("pos_range", "yaw_rotation", "randomized_field"),
    [
        (
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [0.0, 0.0],
            "pos_range",
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [-10.0, 10.0],
            "yaw_rotation",
        ),
    ],
)
def test_compiler_rejects_world_container_geometry_for_randomized_target_pose(
    tmp_path,
    pos_range,
    yaw_rotation,
    randomized_field,
):
    source = _source_task(tmp_path / "source.yaml")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["tasks"][0]["regions"] = [
        {
            "object": "tray",
            "target": "table",
            "random_type": "A_on_B_region_sampler",
            "random_config": {
                "pos_range": pos_range,
                "yaw_rotation": yaw_rotation,
            },
        }
    ]
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        CompileError,
        match=f"CONTAINER_REGION_WORLD_FRAME_RANDOMIZED.*{randomized_field}",
    ) as caught:
        compile_task_config(
            _plan(source),
            _variant(),
            _manifest(source),
            tmp_path / "compiled.yaml",
        )

    assert caught.value.failing_subtask_id == "cup_transfer"


def test_compiler_rejects_inside_without_measured_container_geometry(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["tasks"][0]["container_regions"][0].pop("inner_size")
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(CompileError, match="requires center and inner_size"):
        compile_task_config(
            _plan(source),
            _variant(),
            _manifest(source),
            tmp_path / "compiled.yaml",
        )


def test_compiler_generates_on_predicate_without_container_geometry(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["task_request"]["relation"] = "on"
    plan_data["subtasks"][0]["relation"] = "on"

    output = compile_task_config(
        TaskPlan.from_dict(plan_data),
        _variant(),
        _manifest(source),
        tmp_path / "compiled.yaml",
    )
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]
    place = task["skills"][0]["split_aloha"][0]["left"][1]

    assert place["relation_predicate"] == {
        "relation": "on",
        "geometry_tolerance_m": 0.002,
        "support_gap_tolerance_m": 0.006,
        "minimum_support_contact_n": 0.0,
        "max_unexpected_contact_n": 5.0,
    }


def test_compiler_selects_split_aloha_cameras_with_explicit_owner(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    source_doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    source_doc["tasks"][0]["cameras"] = [
        {
            "name": "navigate_global",
            "translation": [9.0, 8.0, 7.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "camera_axes": "usd",
            "camera_file": "room_camera.yaml",
            "parent": "",
        },
        {
            "name": "franka_hand",
            "parent": "franka/fr3/panda_hand",
        },
    ]
    source.write_text(yaml.safe_dump(source_doc, sort_keys=False), encoding="utf-8")

    output = compile_task_config(
        _plan(source), _variant(), _manifest(source), tmp_path / "compiled.yaml"
    )
    cameras = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]["cameras"]

    assert [camera["name"] for camera in cameras] == [
        "navigate_global",
        "split_aloha_hand_left",
        "split_aloha_hand_right",
        "split_aloha_head",
    ]
    assert cameras[0]["translation"] == [9.0, 8.0, 7.0]
    assert cameras[0]["parent"] == ""
    assert all(camera["record_to"] == "split_aloha" for camera in cameras)
    assert [camera["save_name"] for camera in cameras] == [
        "global",
        "hand_left",
        "hand_right",
        "head",
    ]
    assert not any("franka" in camera["name"] for camera in cameras)


def test_compiler_replaces_inherited_split_aloha_cameras_for_franka(tmp_path):
    source = _source_task(tmp_path / "source_franka.yaml")
    source_doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    task = source_doc["tasks"][0]
    task["robots"] = [
        {
            "name": "franka",
            "robot_config_file": "workflows/simbox/core/configs/robots/fr3.yaml",
        }
    ]
    task["cameras"] = [
        {
            "name": "navigate_global",
            "translation": [2.0, 1.5, 6.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "camera_axes": "usd",
            "camera_file": "room_camera.yaml",
            "parent": "",
        },
        {
            "name": "split_aloha_head",
            "parent": "split_aloha/top_camera_link",
        },
    ]
    source.write_text(yaml.safe_dump(source_doc, sort_keys=False), encoding="utf-8")

    profile = load_robot_profile(
        REPO_ROOT / "workflows/simbox/core/configs/robots/fr3.yaml"
    )
    cameras = _compile_profile_cameras(task, "franka", profile)

    assert [camera["name"] for camera in cameras] == [
        "navigate_global",
        "franka_hand",
        "franka_head",
    ]
    assert all(camera["record_to"] == "franka" for camera in cameras)
    assert [camera["save_name"] for camera in cameras] == ["global", "hand", "head"]
    assert cameras[0]["parent"] == ""
    assert cameras[1]["parent"] == "franka/fr3/panda_hand"
    assert cameras[2]["parent"] == "franka"
    assert not any("split_aloha" in camera["parent"] for camera in cameras)


@pytest.mark.parametrize(
    ("profile_name", "instance_name", "expected_family", "camera_names"),
    [
        (
            "split_aloha",
            "split_aloha",
            "floor_standing",
            ["split_aloha_hand_left", "split_aloha_hand_right", "split_aloha_head"],
        ),
        (
            "lift2",
            "lift2",
            "floor_standing",
            ["lift2_hand_left", "lift2_hand_right", "lift2_head"],
        ),
        (
            "fr3",
            "fr3",
            "support_mounted",
            ["fr3_hand", "fr3_head"],
        ),
    ],
)
def test_same_semantic_plan_compiles_for_three_robot_profiles(
    tmp_path,
    profile_name,
    instance_name,
    expected_family,
    camera_names,
):
    source = _source_task(tmp_path / f"source_{profile_name}.yaml")
    manifest = _manifest(source)
    plan = _plan(source)

    output = compile_task_config(
        plan,
        _variant(profile_name, instance_name),
        manifest,
        tmp_path / f"compiled_{profile_name}.yaml",
        admission_states=frozenset({"implemented", "admitted", "qualified"}),
    )
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]

    assert task["robots"] == [
        {
            "name": instance_name,
            "robot_config_file": (
                f"workflows/simbox/core/configs/robots/{profile_name}.yaml"
            ),
            "euler": [0.0, 0.0, 0.0],
            "use_batch": True,
            "collision_activation_distance": 0.05,
        }
    ]
    assert list(task["skills"][0]) == [instance_name]
    assert [camera["name"] for camera in task["cameras"]] == camera_names
    assert task["metadata"]["agent_plan"]["placement_family"] == expected_family


@pytest.mark.parametrize(
    ("profile_name", "instance_name", "expected_support"),
    [
        ("lift2", "lift2", "floor"),
        ("fr3", "fr3", "central_work_table"),
    ],
)
def test_compiler_routes_robot_region_through_profile_placement_family(
    tmp_path,
    profile_name,
    instance_name,
    expected_support,
):
    arena = tmp_path / "arena.yaml"
    arena.write_text(
        yaml.safe_dump(
            {
                "fixtures": [
                    {"name": "floor", "translation": [0.0, 0.0, 0.0]},
                    {
                        "name": "central_work_table",
                        "translation": [0.0, 0.0, 0.75],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = _source_task(tmp_path / f"source_{profile_name}.yaml")
    source_document = yaml.safe_load(source.read_text(encoding="utf-8"))
    source_task = source_document["tasks"][0]
    source_task.update(
        {
            "arena_file": str(arena),
            "delivery_active_objects": ["cup"],
            "regions": [
                {
                    "object": "cup",
                    "target": "central_work_table",
                    "B": "central_work_table",
                },
                {
                    "object": "split_aloha",
                    "target": "central_work_table",
                    "B": "central_work_table",
                },
            ],
            "source_regions": [
                {
                    "name": "robot_initial_region",
                    "A": "split_aloha",
                    "B": "central_work_table",
                }
            ],
        }
    )
    source.write_text(
        yaml.safe_dump(source_document, sort_keys=False),
        encoding="utf-8",
    )

    output = compile_task_config(
        _plan(source),
        _variant(profile_name, instance_name),
        _manifest(source),
        tmp_path / f"compiled_region_{profile_name}.yaml",
        workspace_candidate={
            "candidate_id": "placement_000",
            "world_xy": [0.25, -0.4],
            "yaw_deg": 20.0,
            "mount_support": expected_support,
        },
        admission_states=frozenset({"implemented", "admitted", "qualified"}),
    )
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]
    robot_region = next(
        region for region in task["regions"] if region["object"] == instance_name
    )
    source_region = next(
        region
        for region in task["source_regions"]
        if region["name"] == "robot_initial_region"
    )

    assert robot_region["target"] == expected_support
    assert robot_region["B"] == expected_support
    assert source_region["B"] == expected_support
    assert "split_aloha" not in output.read_text(encoding="utf-8")


def test_compiler_resolves_canonical_profile_independently_of_process_cwd(
    monkeypatch,
    tmp_path,
):
    source = _source_task(tmp_path / "source.yaml")
    monkeypatch.chdir(tmp_path)

    output = compile_task_config(
        _plan(source),
        _variant(),
        _manifest(source),
        tmp_path / "compiled.yaml",
    )

    assert output.is_file()


def test_compiler_never_overwrites_source_task_or_arena(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    arena = tmp_path / "arena.yaml"
    arena.write_text("fixtures: []\n", encoding="utf-8")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["tasks"][0]["arena_file"] = str(arena)
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    plan = _plan(source)
    manifest = _manifest(source)

    with pytest.raises(CompileError, match="must not overwrite a source task or arena"):
        compile_task_config(plan, _variant(), manifest, source)
    with pytest.raises(CompileError, match="must not overwrite a source task or arena"):
        compile_task_config(plan, _variant(), manifest, arena)

    assert yaml.safe_load(source.read_text(encoding="utf-8"))["tasks"][0][
        "arena_file"
    ] == str(arena)
    assert arena.read_text(encoding="utf-8") == "fixtures: []\n"

    source_asset = tmp_path / "source_asset.usd"
    source_asset.write_bytes(b"source USD bytes")
    with pytest.raises(CompileError, match="output must be a YAML file"):
        compile_task_config(plan, _variant(), manifest, source_asset)
    assert source_asset.read_bytes() == b"source USD bytes"


def test_skill_defaults_are_loaded_from_agent_config_and_agent_params_override_them(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    plan = _plan(source, params={"place_z_offset": 0.07})
    settings = load_agent_settings()
    settings["skill_defaults"]["pick"]["pre_grasp_offset"] = 0.23

    output = compile_task_config(
        plan,
        _variant(),
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

    default_output = compile_task_config(
        default_plan, _variant(), manifest, tmp_path / "default.yaml"
    )
    default_metadata = yaml.safe_load(default_output.read_text(encoding="utf-8"))["tasks"][0]["metadata"]
    assert default_plan.task_request.data_generation is None
    assert default_metadata["agent_plan"]["data_generation"] is True

    override_data = default_plan.to_dict()
    override_data["task_request"]["data_generation"] = False
    override_plan = TaskPlan.from_dict(override_data)
    override_output = compile_task_config(
        override_plan, _variant(), manifest, tmp_path / "override.yaml"
    )
    override_metadata = yaml.safe_load(override_output.read_text(encoding="utf-8"))["tasks"][0]["metadata"]
    assert override_metadata["agent_plan"]["data_generation"] is False


def test_planning_agent_receives_the_stage_spec_and_typed_skill_contracts(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    manifest = _manifest(source)
    backend = _CaptureBackend(_plan(source))
    resolver = TaskResolver(backend=backend, skill_contracts=load_skill_contracts())

    resolver.plan("把杯子放进托盘", None, manifest, tmp_path / "decisions")

    assert "任务规划与机器人 Skill 编排策略" in backend.prompt
    assert "操作对象与目标对象识别策略" in backend.prompt
    assert '"enabled": true' in backend.prompt
    assert '"semantic_arm_constraint": "any_single_arm"' in backend.prompt
    assert "sequential Pick then Place" in backend.prompt
    assert '"allowed_values"' in backend.prompt
    assert "on/inside 的严格谓词由编译器生成" in backend.prompt
    assert "insert 在 v1 中不准入" in backend.prompt


def test_planning_policy_points_to_the_machine_readable_skill_contract():
    spec = (
        Path(__file__).resolve().parents[2]
        / "agent"
        / "workflow"
        / "task_planning_policy.md"
    ).read_text(encoding="utf-8")

    assert "agent/robot_skills/contracts.yaml" in spec


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"success_mode": "inside"}, "invalid value for place.success_mode"),
        ({"success_mode": "xybbox"}, "compiler-owned for relation='inside'"),
        ({"semantic_relation": "inside"}, "owned by compiler"),
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
    stage["skills"][0]["arm"] = "left"
    stage["skills"][1]["arm"] = "right"
    plan = TaskPlan.from_dict(plan_data)

    with pytest.raises(CompileError, match="not enabled"):
        compile_task_config(
            plan, _variant(), _manifest(source), tmp_path / "compiled.yaml"
        )


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


def test_explicit_subtask_arm_constrains_execution_variants(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan_data = _plan(source).to_dict()
    plan_data["subtasks"][0]["arm"] = "right"
    plan = TaskPlan.from_dict(plan_data)

    resolver = _resolver()
    resolver.validate_plan(plan, _manifest(source))
    variants = resolver.execution_variants(plan)

    assert variants
    assert {variant.arm_binding["cup_transfer"] for variant in variants} == {"right"}


def test_any_single_arm_emits_one_execution_variant_per_available_arm(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    plan = _plan(source)

    variants = _resolver().execution_variants(plan)

    assert {variant.arm_binding["cup_transfer"] for variant in variants} == {
        "left",
        "right",
    }


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
    output = compile_task_config(
        plan,
        _variant(arm_binding={"cup_transfer": "left", "spoon_transfer": "right"}),
        manifest,
        tmp_path / "compiled.yaml",
    )
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


def test_compiler_accepts_only_typed_scene_revision_with_matching_source(tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    derived_document = yaml.safe_load(source.read_text(encoding="utf-8"))
    derived_document["tasks"][0].setdefault("metadata", {})[
        "agent_scene_layout"
    ] = {
        "scene_revision": "revision_a",
        "source_task": str(source.resolve()),
        "source_task_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    derived_document["tasks"][0]["container_regions"][0]["center"] = [0.6, 0.2]
    derived = tmp_path / "derived.yaml"
    derived.write_text(
        yaml.safe_dump(derived_document, sort_keys=False), encoding="utf-8"
    )

    output = compile_task_config(
        _plan(source),
        _variant(),
        _manifest(source),
        tmp_path / "compiled.yaml",
        scene_task_path=derived,
    )
    task = yaml.safe_load(output.read_text(encoding="utf-8"))["tasks"][0]
    assert task["metadata"]["agent_plan"]["scene_revision"] == "revision_a"
    place = task["skills"][0]["split_aloha"][0]["left"][1]
    assert place["relation_predicate"]["container_region"]["center"] == [0.6, 0.2]

    untyped = tmp_path / "untyped.yaml"
    untyped.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(CompileError, match="typed SceneLayout revision"):
        compile_task_config(
            _plan(source),
            _variant(),
            _manifest(source),
            tmp_path / "rejected.yaml",
            scene_task_path=untyped,
        )


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


def test_agent_workspace_validation_requires_pick_and_place_gate(monkeypatch, tmp_path):
    manifest_path = tmp_path / "candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "required_arm": "left",
                "geometry_candidates": [
                    {"candidate_id": "candidate_0", "geometry_feasible": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "planning_success",
                "selected_candidate": {
                    "candidate_id": "candidate_0",
                    "arm": "left",
                },
                "planning_probe_artifacts": {
                    "pick": "pick.json",
                    "place": "place.json",
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(compiler_module.subprocess, "run", fake_run)
    selected = validate_workspace_manifest(
        manifest_path,
        gpu=0,
        arm="left",
        attach_prim_path_children=["Aligned/collisions"],
        seed=4,
    )

    gate_index = captured["command"].index("--planning-gate")
    assert captured["command"][gate_index + 1] == "pick-place"
    seed_index = captured["command"].index("--seed")
    assert captured["command"][seed_index + 1] == "4"
    assert selected == {"candidate_id": "candidate_0", "arm": "left"}


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

    def fake_validate(path, _gpu, **kwargs):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        local = value["geometry_candidates"][0]
        probes.append(
            {
                "arm": kwargs["arm"],
                "target": value["target"]["name"],
                "world_xy": local["world_xy"],
                "attach": kwargs["attach_prim_path_children"],
                "diagnostic_paths": kwargs["diagnostic_disable_curobo_obstacle_paths"],
            }
        )
        return {**local, "arm": kwargs["arm"]}

    monkeypatch.setattr(compiler_module, "validate_workspace_manifest", fake_validate)
    selected = select_task_workspace_candidate(
        plan,
        _variant(
            arm_binding={"cup_transfer": "left", "spoon_transfer": "right"}
        ),
        workspace_paths,
        manifest,
        tmp_path / "selection",
        gpu=0,
        settings={
            "debug": {
                "workspace_probe": {
                    "disable_curobo_obstacle_paths": [
                        "/World/task_0/wall_south/collision_volume"
                    ]
                }
            },
            "workspace": {},
        },
    )

    assert selected["candidate_id"] == "common_000"
    assert [(item["target"], item["arm"]) for item in probes] == [
        ("cup", "left"),
        ("spoon", "right"),
    ]
    assert {tuple(item["world_xy"]) for item in probes} == {(-0.6, 0.0)}
    assert probes[0]["attach"] == ["/World/cup/collision"]
    assert probes[1]["attach"] == ["/World/spoon/collision"]
    assert all(
        item["diagnostic_paths"] == ["/World/task_0/wall_south/collision_volume"]
        for item in probes
    )
    selection = json.loads((tmp_path / "selection" / "position_selection.json").read_text())
    assert selection["mode"] == "common"

    def fail_spoon_probe(path, _gpu, _conda_env, **kwargs):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["target"]["name"] == "spoon":
            raise CompileError("NO_JOINT_GRASP_PLAN: spoon is unreachable")
        return {**value["geometry_candidates"][0], "arm": kwargs["arm"]}

    monkeypatch.setattr(
        compiler_module, "validate_workspace_manifest", fail_spoon_probe
    )
    with pytest.raises(CompileError) as failure:
        select_task_workspace_candidate(
            plan,
            _variant(
                arm_binding={"cup_transfer": "left", "spoon_transfer": "right"}
            ),
            workspace_paths,
            manifest,
            tmp_path / "failed_selection",
            gpu=0,
            settings={"workspace": {}},
        )
    assert failure.value.failing_subtask_id == "spoon_transfer"
    failed_selection = json.loads(
        (tmp_path / "failed_selection" / "position_selection.json").read_text()
    )
    assert failed_selection["failing_subtask_id"] == "spoon_transfer"


def test_orchestrator_compiles_and_runs_all_subtasks_once(monkeypatch, tmp_path):
    source = _source_task(tmp_path / "source.yaml")
    source_manifest = _manifest(source)
    manifest_data = source_manifest.to_dict()
    manifest_data["objects"].append({"name": "spoon", "category": "spoon"})
    source_manifest = SceneCapabilityManifest.from_dict(manifest_data)
    resolution = ResolutionDecision.from_dict(
        {
            "mode": "reuse_scene_new_task",
            "selected_task_id": source_manifest.task_id,
            "selected_source_task": source_manifest.source_task,
            "selected_scene_id": source_manifest.scene_id,
            "object_role_overrides": {"cup": "manipulated", "tray": "target"},
            "decision_basis": "reuse the persisted scene source",
        }
    )
    manifest = _resolver().build_synthetic_manifest(resolution, source_manifest)
    plan_data = _plan(source).to_dict()
    plan_data["selected_task_id"] = manifest.task_id
    plan_data["task_request"]["data_generation"] = False
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
    source_arena = tmp_path / "arena.yaml"
    source_arena.write_text("fixtures: []\n", encoding="utf-8")
    source_task_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_arena_hash = hashlib.sha256(source_arena.read_bytes()).hexdigest()
    source_members = [
        SourceMember(str(source.resolve()), "source_task", source_task_hash),
        SourceMember(str(source_arena.resolve()), "source_arena", source_arena_hash),
    ]
    source_hash = canonical_source_hash(source_members)
    (run_dir / "source_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
                "source_task": str(source.resolve()),
                "source_arena": str(source_arena.resolve()),
                "source_task_hash": source_task_hash,
                "source_arena_hash": source_arena_hash,
                "members": [member.to_dict() for member in source_members],
                "source_hash": source_hash,
            }
        ),
        encoding="utf-8",
    )
    plan_path = run_dir / "task_plan.json"
    dump_contract(plan, plan_path)
    selected_manifest_path = run_dir / "selected_manifest.json"
    dump_contract(manifest, selected_manifest_path)
    variant = _variant(
        arm_binding={"cup_transfer": "left", "spoon_transfer": "right"}
    )
    (run_dir / "execution_variants.json").write_text(
        json.dumps([variant.to_dict()]), encoding="utf-8"
    )
    (run_dir / "workspace_manifests.json").write_text(
        json.dumps(
            {
                variant.variant_id: {
                    "cup_transfer": str(run_dir / "cup.json"),
                    "spoon_transfer": str(run_dir / "spoon.json"),
                }
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
        selected_manifest_path=str(selected_manifest_path),
    )
    calls = {"select": 0, "compile": 0, "run": 0}

    def inventory_must_not_be_read(*_args, **_kwargs):
        raise AssertionError("planned and resumed runs must use selected_manifest.json")

    monkeypatch.setattr(
        orchestrator_module,
        "load_or_build_inventory",
        inventory_must_not_be_read,
    )

    def fake_select(*_args, **_kwargs):
        calls["select"] += 1
        return {"candidate_id": "common_000", "world_xy": [-0.6, 0.0], "yaw_deg": 0.0}

    def fake_compile(compiled_plan, compiled_variant, _manifest, output_path, **_kwargs):
        calls["compile"] += 1
        assert compiled_variant == variant
        assert [item.subtask_id for item in compiled_plan.subtasks] == [
            "cup_transfer",
            "spoon_transfer",
        ]
        output_path.write_text(
            yaml.safe_dump(
                {
                    "tasks": [
                        {
                            "robots": [
                                {
                                    "name": variant.instance_name,
                                    "robot_config_file": variant.robot_config_file,
                                }
                            ],
                            "metadata": {
                                "agent_plan": {
                                    "execution_variant_id": variant.variant_id,
                                    "robot_profile_id": variant.profile_id,
                                    "robot_profile_hash": variant.profile_hash,
                                    "scene_revision": "source",
                                    "subtasks": [
                                        {
                                            "subtask_id": item.subtask_id,
                                            "center_object": item.manipulated_object,
                                            "target_object": item.target_object,
                                            "relation": str(
                                                getattr(item.relation, "value", item.relation)
                                            ),
                                            "arm": variant.arm_binding[item.subtask_id],
                                        }
                                        for item in compiled_plan.subtasks
                                    ],
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

    def fake_run(_config_path, attempt_dir, identity, *, data_generation):
        calls["run"] += 1
        assert data_generation is False
        assert identity.run_id == "multi"
        assert identity.variant_id == variant.variant_id
        assert identity.seed == 0
        episode_dir = attempt_dir / "data" / "episode"
        (episode_dir / "images.rgb.global").mkdir(parents=True)
        (episode_dir / "lmdb").mkdir()
        (episode_dir / "images.rgb.global" / "demo.mp4").write_bytes(b"video")
        (episode_dir / "lmdb" / "data.mdb").write_bytes(b"data")
        (episode_dir / "meta_info.pkl").write_bytes(b"metadata")
        (episode_dir / "collision_world_audit.json").write_text(
            json.dumps(
                {
                    "world_revision": 2,
                    "physics_curobo_difference": {
                        "split_aloha/left": {
                            "missing_in_curobo": [],
                            "unexpected_in_curobo": [],
                        },
                        "split_aloha/right": {
                            "missing_in_curobo": [],
                            "unexpected_in_curobo": [],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (episode_dir / "safety_events.jsonl").write_text("", encoding="utf-8")
        event_path = attempt_dir / "episode_events.jsonl"
        log_path = attempt_dir / "stdout.log"
        event_path.write_text(
            json.dumps(
                {
                    "event": "episode_saved",
                    "finalized": True,
                    "status": "success",
                    "task_predicate_success": True,
                    "predicate_results": [
                        {
                            "predicate_id": f"relation_{index:02d}",
                            "subtask_id": item.subtask_id,
                            "skill": "place",
                            "objects": [item.manipulated_object, item.target_object],
                            "relation": str(getattr(item.relation, "value", item.relation)),
                            "terminal_success": True,
                            "success": True,
                            "checks": {"support_gap_ok": True},
                            "measurements": {"support_gap_m": 0.0},
                            "thresholds": {"support_gap_tolerance_m": 0.006},
                        }
                        for index, item in enumerate(plan.subtasks)
                    ],
                    "primary_episode_dir": str(episode_dir),
                    "num_steps": 10,
                    "video_stream_count": 1,
                    **identity.to_dict(),
                    "world_revision": 2,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        log_path.write_text("Task is successful\n", encoding="utf-8")
        return 0, False, event_path, log_path

    monkeypatch.setattr(orchestrator_module, "select_task_workspace_candidate", fake_select)
    monkeypatch.setattr(orchestrator_module, "compile_task_config", fake_compile)
    monkeypatch.setattr(
        orchestrator_module,
        "write_variant_artifact_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(complete=True, failure_codes=[]),
    )
    orchestrator = AgentOrchestrator(
        run_root=tmp_path / "runs",
        inventory_path=tmp_path / "inventory.json",
        retain_experience=False,
        settings={
            **load_agent_settings(),
            "backend": {"type": "codex_cli", "model": None},
        },
    )
    monkeypatch.setattr(orchestrator, "_run_simbox", fake_run)

    result = orchestrator._execute_planned(state)

    assert result.status == RunStatus.SUCCEEDED
    assert result.current_subtask == 2
    assert calls == {"select": 1, "compile": 1, "run": 1}
    assert (
        run_dir / "variants" / variant.variant_id / "attempts" / "00" / "task.yaml"
    ).is_file()
