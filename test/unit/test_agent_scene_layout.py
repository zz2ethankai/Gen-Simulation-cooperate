from __future__ import annotations

import hashlib
import json
import pickle
import sys
import threading
from pathlib import Path

import lmdb
import imageio.v2 as imageio
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.evidence import _predicate_payload_error, classify_evidence, collect_evidence
from agent.contracts import ExecutionIdentity
from agent.inventory import build_inventory
from agent.tools.scene_ingest import build_task_document
from agent.settings import SCENE_INGEST_DEFAULTS
from agent.tools.feedback import RepairAction, classify_failure
from agent.tools.scene_layout import (
    CandidateAggregate,
    CandidateEvaluation,
    CandidateGenome,
    EvolutionSearch,
    MoveEntityOnSupport,
    RotateEntityOnSupport,
    SceneLayoutBlocked,
    SceneLayoutCompiler,
    SceneLayoutPlanner,
    SceneMutationError,
    SceneSpec,
    SetRobotPlacement,
    SetSupportHeight,
)
from agent.tools.trace import TraceContext, TraceEvent, TraceWriter
from agent.tools.scene_layout.models import SceneSpecError
from agent.tools.source_integrity import (
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
    SourceMember,
    canonical_source_hash,
)
from workflows.simbox.core.robots.profile import load_robot_profile_for_task, project_runtime_config


def _write_yaml(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _source_documents(tmp_path: Path) -> tuple[Path, Path]:
    arena = {
        "fixtures": [
            {"name": "floor", "translation": [0.0, 0.0, 0.0], "support_surface_z": 0.0},
            {
                "name": "table",
                "target_class": "PlaneObject",
                "translation": [1.0, 2.0, 0.4],
                "size": [1.2, 1.0],
                "support_surface_z": 0.8,
            },
            {"name": "fixture_on_table", "translation": [1.0, 2.0, 0.9], "parent_fixture": "table"},
        ],
        "regions": [
            {
                "name": "arena_apple",
                "object": "apple",
                "B": "table",
                "support_surface_z": 0.8,
                "random_config": {"support_surface_z": 0.8},
            }
        ],
    }
    task = {
        "tasks": [
            {
                "name": "layout_test",
                "arena_file": str(tmp_path / "arena.yaml"),
                "objects": [{"name": "apple"}, {"name": "tool"}],
                "robots": [{"name": "robot_a"}],
                "regions": [
                    {
                        "name": "apple_region",
                        "object": "apple",
                        "target": "table",
                        "B": "table",
                        "center": [1.1, 2.2],
                        "size": [0.2, 0.3],
                        "runtime_placement": {
                            "frame": "parent_world_xy_offset",
                            "offset_xy": [0.1, 0.2],
                        },
                        "support_surface_z": 0.8,
                        "random_type": "A_on_B_region_sampler",
                        "random_config": {
                            "pos_range": [[0.1, 0.2, 0.0], [0.1, 0.2, 0.0]],
                            "yaw_rotation": [0.0, 0.0],
                            "support_surface_z": 0.8,
                        },
                    },
                    {
                        "name": "tool_region",
                        "object": "tool",
                        "target": "fixture_on_table",
                        "B": "fixture_on_table",
                        "center": [1.0, 2.0],
                        "size": [0.08, 0.08],
                        "runtime_placement": {
                            "frame": "parent_world_xy_offset",
                            "offset_xy": [0.0, 0.0],
                        },
                        "support_surface_z": 0.9,
                        "random_type": "A_on_B_region_sampler",
                        "random_config": {
                            "pos_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            "yaw_rotation": [0.0, 0.0],
                            "support_surface_z": 0.9,
                        },
                    },
                    {
                        "name": "robot_region",
                        "object": "robot_a",
                        "target": "table",
                        "B": "table",
                        "support_surface_z": 0.8,
                        "random_type": "A_on_B_region_sampler",
                        "random_config": {
                            "pos_range": [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                            "yaw_rotation": [0.0, 0.0],
                            "support_surface_z": 0.8,
                        },
                    },
                ],
                "source_regions": [
                    {
                        "name": "robot_initial_region",
                        "A": "robot",
                        "B": "table",
                        "robot_base": "robot_a",
                        "center": [0.0, -1.0],
                        "center_xyz": [0.0, 0.0, -1.0],
                        "yaw_range": [0.0, 0.0],
                        "support_surface_z": 0.8,
                    }
                ],
            }
        ]
    }
    return _write_yaml(tmp_path / "task.yaml", task), _write_yaml(tmp_path / "arena.yaml", arena)


def _place_tool_on_table(
    task_path: Path,
    *,
    center_xy: tuple[float, float],
) -> None:
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    tool = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "tool"
    )
    offset = [center_xy[0] - 1.0, center_xy[1] - 2.0]
    tool.update(
        {
            "target": "table",
            "B": "table",
            "center": list(center_xy),
            "size": [0.08, 0.08],
            "runtime_placement": {
                "frame": "parent_world_xy_offset",
                "offset_xy": offset,
            },
            "support_surface_z": 0.8,
        }
    )
    tool["random_config"].update(
        {
            "pos_range": [[*offset, 0.0], [*offset, 0.0]],
            "yaw_rotation": [0.0, 0.0],
            "support_surface_z": 0.8,
        }
    )
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")


def test_scene_ingest_preserves_region_geometry_and_robot_identity(tmp_path: Path):
    scene_dir = tmp_path / "scene"
    object_path = scene_dir / "assets" / "apple.usd"
    object_path.parent.mkdir(parents=True)
    object_path.write_text("#usda 1.0\n", encoding="utf-8")
    interdata = {
        "objects": [{"name": "apple", "usd_path": "assets/apple.usd"}],
        "regions": [
            {
                "name": "apple_region",
                "object": "apple",
                "A": "apple",
                "B": "table__support_plane",
                "parent_fixture": "table",
                "center": [1.25, 2.5],
                "size": [0.2, 0.3],
                "runtime_placement": {
                    "frame": "parent_world_xy_offset",
                    "offset_xy": [0.25, 0.5],
                },
                "support_surface_z": 0.9,
                "yaw_range": [10.0, 20.0],
                "random_config": {
                    "pos_range": [[-0.01, -0.02, 0.0], [0.01, 0.02, 0.0]],
                    "support_surface_z": 0.9,
                },
            },
            {
                "name": "robot_start_region",
                "object": "floor_bot",
                "A": "floor_bot",
                "B": "floor",
                "center": [0.0, -1.5],
                "size": [0.1, 0.1],
                "yaw_range": [90.0, 90.0],
                "random_config": {
                    "pos_range": [[0.0, -1.5, 0.0], [0.0, -1.5, 0.0]],
                    "yaw_rotation": [0.0, 0.0],
                },
            },
        ],
        "robot": {
            "name": "floor_bot",
            "robot_config_file": "workflows/simbox/core/configs/robots/lift2.yaml",
        },
        "cameras": [],
    }
    arena = {
        "fixtures": [
            {"name": "floor", "translation": [0.0, 0.0, 0.0]},
            {"name": "table", "translation": [1.0, 2.0, 0.5]},
            {
                "name": "table__support_plane",
                "translation": [1.0, 2.0, 0.9],
                "parent_fixture": "table",
            },
        ]
    }
    cfg = dict(SCENE_INGEST_DEFAULTS)
    cfg["support_fixture"] = "table"
    document, _ = build_task_document(
        interdata,
        arena,
        scene_dir,
        cfg,
        tmp_path / "simbox_task.yaml",
        tmp_path / "simbox_arena.yaml",
    )
    apple = next(region for region in document["regions"] if region["object"] == "apple")
    assert apple["center"] == [1.25, 2.5]
    assert apple["size"] == [0.2, 0.3]
    assert apple["runtime_placement"]["offset_xy"] == [0.25, 0.5]
    assert apple["random_config"]["support_surface_z"] == 0.9
    assert apple["target"] == "table__support_plane"
    assert apple["parent_fixture"] == "table"
    robot_region = next(region for region in document["regions"] if region["object"] == "floor_bot")
    assert robot_region["random_config"]["pos_range"] == [
        [0.0, -1.5, 0.0],
        [0.0, -1.5, 0.0],
    ]
    robot_document = document["robots"][0]
    assert set(robot_document) == {
        "name",
        "robot_config_file",
        "use_batch",
        "collision_activation_distance",
        "ignore_substring",
    }
    profile = load_robot_profile_for_task(robot_document, tmp_path / "simbox_task.yaml")
    runtime_config = project_runtime_config(profile, robot_document)
    assert runtime_config["profile_id"] == "lift2_floor_standing_v1"
    assert {camera["save_name"] for camera in document["cameras"]} == {
        "hand_left",
        "hand_right",
        "head",
    }
    assert list(document["skills"][0]) == ["floor_bot"]


def test_scene_layout_compiles_atomic_derived_revision(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    source_task_hash = hashlib.sha256(task_path.read_bytes()).hexdigest()
    source_arena_hash = hashlib.sha256(arena_path.read_bytes()).hexdigest()
    result = SceneLayoutCompiler().compile(
        task_path,
        arena_path,
        [
            MoveEntityOnSupport("apple", "table", (0.3, -0.1)),
            RotateEntityOnSupport("apple", 45.0),
            SetSupportHeight("table", 1.0),
            SetRobotPlacement("robot_a", "table", (0.2, -0.4), 90.0),
        ],
        tmp_path / "derived" / "candidate_00",
    )
    assert hashlib.sha256(task_path.read_bytes()).hexdigest() == source_task_hash
    assert hashlib.sha256(arena_path.read_bytes()).hexdigest() == source_arena_hash
    derived_task = yaml.safe_load(result.task_path.read_text(encoding="utf-8"))["tasks"][0]
    derived_arena = yaml.safe_load(result.arena_path.read_text(encoding="utf-8"))
    apple = next(region for region in derived_task["regions"] if region["object"] == "apple")
    assert apple["random_config"]["pos_range"] == [
        [0.4, 0.1, 0.0],
        [0.4, 0.1, 0.0],
    ]
    assert apple["random_config"]["yaw_rotation"] == [45.0, 45.0]
    assert apple["support_surface_z"] == 1.0
    table = next(fixture for fixture in derived_arena["fixtures"] if fixture["name"] == "table")
    child = next(
        fixture for fixture in derived_arena["fixtures"] if fixture["name"] == "fixture_on_table"
    )
    assert table["translation"][2] == pytest.approx(0.6)
    assert table["support_surface_z"] == 1.0
    assert child["translation"][2] == pytest.approx(1.1)
    tool = next(region for region in derived_task["regions"] if region["object"] == "tool")
    assert tool["support_surface_z"] == pytest.approx(1.1)
    assert tool["random_config"]["support_surface_z"] == pytest.approx(1.1)
    robot = next(region for region in derived_task["regions"] if region["object"] == "robot_a")
    assert robot["random_config"]["pos_range"][0] == [0.2, -1.4, 0.0]
    source_region = derived_task["source_regions"][0]
    assert source_region["center"] == [0.2, -1.4]
    assert source_region["center_xyz"] == [0.2, 0.0, -1.4]
    assert source_region["support_surface_z"] == pytest.approx(1.0)
    mutation_record = json.loads(result.mutation_path.read_text(encoding="utf-8"))
    assert mutation_record["scene_revision"] == result.scene_revision
    assert result.scene_spec.support_graph.entities_on("table") == (
        "apple",
        "fixture_on_table",
        "robot_a",
    )


def test_move_entity_freezes_runtime_position_range(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple["random_config"]["pos_range"] = [
        [0.08, 0.17, 0.0],
        [0.12, 0.23, 0.0],
    ]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    result = SceneLayoutCompiler().compile(
        task_path,
        arena_path,
        [MoveEntityOnSupport("apple", "table", (0.2, -0.1))],
        tmp_path / "derived" / "fixed_runtime_pose",
    )

    derived_task = yaml.safe_load(result.task_path.read_text(encoding="utf-8"))["tasks"][0]
    derived_apple = next(
        region for region in derived_task["regions"] if region["object"] == "apple"
    )
    assert derived_apple["random_config"]["pos_range"][0] == pytest.approx(
        [0.3, 0.1, 0.0]
    )
    assert derived_apple["random_config"]["pos_range"][1] == pytest.approx(
        [0.3, 0.1, 0.0]
    )
    assert derived_apple["center"] == pytest.approx([1.3, 2.1])
    assert derived_apple["runtime_placement"]["offset_xy"] == pytest.approx(
        [0.3, 0.1]
    )


def test_container_region_follows_exact_entity_translation_and_quarter_turn(
    tmp_path: Path,
):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    task["tasks"][0]["container_regions"] = [
        {
            "name": "apple_interior",
            "object": "apple",
            "can_receive_objects": True,
            "center": [1.15, 2.2],
            "inner_size": [0.12, 0.08],
            "interior_support_z": 0.8,
        }
    ]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    raised = SceneLayoutCompiler().compile(
        task_path,
        arena_path,
        [SetSupportHeight("table", 1.0)],
        tmp_path / "derived" / "raised_container",
    )
    raised_task = yaml.safe_load(raised.task_path.read_text(encoding="utf-8"))[
        "tasks"
    ][0]
    assert raised_task["container_regions"][0]["interior_support_z"] == pytest.approx(
        1.0
    )

    translated = SceneLayoutCompiler().compile(
        task_path,
        arena_path,
        [MoveEntityOnSupport("apple", "table", (0.3, -0.1))],
        tmp_path / "derived" / "translated_container",
    )
    translated_task = yaml.safe_load(
        translated.task_path.read_text(encoding="utf-8")
    )["tasks"][0]
    assert translated_task["container_regions"][0]["center"] == pytest.approx(
        [1.45, 2.1]
    )

    rotated = SceneLayoutCompiler().compile(
        task_path,
        arena_path,
        [RotateEntityOnSupport("apple", 90.0)],
        tmp_path / "derived" / "rotated_container",
    )
    rotated_task = yaml.safe_load(rotated.task_path.read_text(encoding="utf-8"))[
        "tasks"
    ][0]
    assert rotated_task["container_regions"][0]["center"] == pytest.approx(
        [1.1, 2.25]
    )
    assert rotated_task["container_regions"][0]["inner_size"] == pytest.approx(
        [0.08, 0.12]
    )


def test_container_region_rejects_stale_or_unrepresentable_geometry(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    task["tasks"][0]["container_regions"] = [
        {
            "name": "apple_interior",
            "object": "apple",
            "can_receive_objects": True,
            "inner_size": [0.12, 0.08],
            "interior_support_z": 0.8,
        }
    ]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    missing_center_output = tmp_path / "derived" / "missing_center"

    with pytest.raises(SceneMutationError, match="center must contain exactly two"):
        SceneLayoutCompiler().compile(
            task_path,
            arena_path,
            [MoveEntityOnSupport("apple", "table", (0.2, 0.0))],
            missing_center_output,
        )
    assert not missing_center_output.exists()

    task["tasks"][0]["container_regions"][0]["center"] = [1.1, 2.2]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    non_axis_aligned_output = tmp_path / "derived" / "non_axis_aligned"
    with pytest.raises(SceneMutationError, match="axis-aligned container-region schema"):
        SceneLayoutCompiler().compile(
            task_path,
            arena_path,
            [RotateEntityOnSupport("apple", 45.0)],
            non_axis_aligned_output,
        )
    assert not non_axis_aligned_output.exists()


def test_scene_spec_keeps_center_size_and_runtime_placement(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    arena = yaml.safe_load(arena_path.read_text(encoding="utf-8"))
    spec = SceneSpec.from_documents(task, arena, source_task=task_path, source_arena=arena_path)
    apple = next(region for region in spec.regions if region.entity == "apple")
    assert apple.center_xy == (1.1, 2.2)
    assert apple.size_xy == (0.2, 0.3)
    assert apple.runtime_placement == {
        "frame": "parent_world_xy_offset",
        "offset_xy": [0.1, 0.2],
    }


def test_scene_spec_merges_source_geometry_through_entity_aliases(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple_object = next(
        item for item in task["tasks"][0]["objects"] if item["name"] == "apple"
    )
    apple_object["source_name"] = "apple__source"
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple.pop("center")
    apple.pop("size")
    task["tasks"][0]["source_regions"].append(
        {
            "name": "apple_source_region",
            "A": "apple__source",
            "B": "table",
            "center": [1.1, 2.2],
            "size": [0.27, 0.126],
        }
    )

    spec = SceneSpec.from_documents(
        task,
        yaml.safe_load(arena_path.read_text(encoding="utf-8")),
        source_task=task_path,
        source_arena=arena_path,
    )

    apple_spec = next(region for region in spec.regions if region.entity == "apple")
    assert apple_spec.center_xy == (1.1, 2.2)
    assert apple_spec.size_xy == (0.27, 0.126)
    assert spec.support_graph.entities_on("table").count("apple") == 1


def test_scene_spec_separates_semantic_and_runtime_supports(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    arena = yaml.safe_load(arena_path.read_text(encoding="utf-8"))
    plane = "table__support_plane_z0800"
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple.update(
        {
            "target": plane,
            "B": plane,
            "support_collision_plane": plane,
            "parent_fixture": "table",
            "support_target_fixture": "table",
        }
    )
    task["tasks"][0]["source_regions"].append(
        {
            "name": "apple_source",
            "object": "apple",
            "A": "apple",
            "B": plane,
            "target": plane,
            "parent_fixture": "table",
            "support_target_fixture": "table",
            "support_collision_plane": plane,
        }
    )

    spec = SceneSpec.from_documents(
        task,
        arena,
        source_task=task_path,
        source_arena=arena_path,
    )

    apple_spec = next(region for region in spec.regions if region.entity == "apple")
    assert apple_spec.support == "table"
    assert apple_spec.runtime_support == plane
    assert "apple" in spec.support_graph.entities_on("table")
    assert spec.support_graph.runtime_targets_for("apple") == (plane,)


def test_scene_spec_rejects_genuine_semantic_support_conflict(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    arena = yaml.safe_load(arena_path.read_text(encoding="utf-8"))
    task["tasks"][0]["source_regions"].append(
        {"name": "apple_source", "object": "apple", "A": "apple", "B": "shelf"}
    )

    with pytest.raises(SceneSpecError, match="conflicting semantic supports"):
        SceneSpec.from_documents(
            task,
            arena,
            source_task=task_path,
            source_arena=arena_path,
        )


def test_inventory_ingest_writes_only_to_owned_output_root(tmp_path: Path):
    scene_root = tmp_path / "scenes"
    scene_dir = scene_root / "scene_a"
    source_task = _write_yaml(
        scene_dir / "interdata" / "task.yaml",
        {
            "objects": [
                {
                    "name": "apple",
                    "usd_path": "assets/apple.usd",
                    "rigidbody": True,
                    "collision_enabled": True,
                }
            ],
            "regions": [
                {
                    "name": "apple_region",
                    "object": "apple",
                    "A": "apple",
                    "B": "table",
                    "center": [0.0, 0.0],
                    "size": [0.2, 0.2],
                    "support_surface_z": 0.8,
                },
                {
                    "name": "robot_start_region",
                    "object": "floor_bot",
                    "A": "floor_bot",
                    "B": "floor",
                    "center": [0.0, -1.0],
                    "yaw_range": [0.0, 0.0],
                    "support_surface_z": 0.0,
                },
            ],
            "robot": {
                "name": "floor_bot",
                "robot_config_file": "workflows/simbox/core/configs/robots/lift2.yaml",
            },
            "tasks": [{"task_name": "place apple", "task_description": "place apple"}],
        },
    )
    source_arena = _write_yaml(
        scene_dir / "interdata" / "arena.yaml",
        {
            "fixtures": [
                {"name": "floor", "translation": [0.0, 0.0, 0.0]},
                {
                    "name": "table",
                    "translation": [0.0, 0.0, 0.4],
                    "support_surface_z": 0.8,
                },
            ]
        },
    )
    asset = scene_dir / "assets" / "apple.usd"
    asset.parent.mkdir(parents=True)
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    source_state = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (source_task, source_arena, asset)
    }
    output_root = tmp_path / "inventory_output"
    settings = {
        "scene_ingest": {
            "enabled": True,
            "output_root": str(output_root),
            "support_fixture": "table",
        }
    }

    manifests = build_inventory([scene_root], settings=settings)

    assert len(manifests) == 1
    derived_task = Path(manifests[0].source_task)
    assert derived_task.is_relative_to(output_root)
    assert derived_task.is_file()
    assert (derived_task.parent / "simbox_arena.yaml").is_file()
    assert not (scene_dir / "assets" / "basic").exists()
    assert {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (source_task, source_arena, asset)
    } == source_state


def test_support_height_rejects_untracked_link_without_partial_output(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    robot = next(region for region in task["tasks"][0]["regions"] if region["object"] == "robot_a")
    robot.pop("support_surface_z")
    robot["random_config"].pop("support_surface_z")
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "derived" / "invalid"
    with pytest.raises(ValueError, match="no explicit support height"):
        SceneLayoutCompiler().compile(
            task_path,
            arena_path,
            [SetSupportHeight("table", 1.0)],
            output_dir,
        )
    assert not output_dir.exists()


def test_evolution_uses_fixed_budget_all_debug_seeds_and_four_slots(tmp_path: Path):
    population = tuple(
        CandidateGenome(f"c{index}", 0, "source", "robot_profile")
        for index in range(8)
    )
    observed: list[tuple[str, int, int]] = []
    lock = threading.Lock()

    def evaluator(genome: CandidateGenome, seed: int, slot: int) -> CandidateEvaluation:
        with lock:
            observed.append((genome.candidate_id, seed, slot))
        winner = genome.candidate_id == "c0"
        return CandidateEvaluation(
            genome.candidate_id,
            genome.generation,
            seed,
            {"schema": True, "support": True, "collision": winner},
            planning_success=winner,
            collision_margin_m=0.1 if winner else 0.0,
            path_length_m=1.0,
        )

    def should_not_evolve(*_args):
        raise AssertionError("a robust generation-zero winner must stop search")

    result = EvolutionSearch().run(population, evaluator, should_not_evolve, tmp_path / "search")
    assert result.robust_winner == population[0]
    assert len(observed) == 8 * 5
    assert {seed for _, seed, _ in observed} == set(range(5))
    assert {slot for _, _, slot in observed} <= set(range(4))
    assert len((tmp_path / "search" / "generation_00.jsonl").read_text().splitlines()) == 40


def test_layout_planner_generates_measured_population_and_static_artifacts(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple["size"] = [0.12, 0.08]
    apple["center"] = [1.1, 2.2]
    apple["runtime_placement"]["offset_xy"] = [0.1, 0.2]
    apple["random_config"]["pos_range"] = [
        [0.08, 0.18, 0.0],
        [0.12, 0.22, 0.0],
    ]
    tool = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "tool"
    )
    tool.update(
        {
            "target": "table",
            "B": "table",
            "center": [1.38, 2.0],
            "runtime_placement": {
                "frame": "parent_world_xy_offset",
                "offset_xy": [0.38, 0.0],
            },
        }
    )
    tool["random_config"]["pos_range"] = [
        [0.38, 0.0, 0.0],
        [0.38, 0.0, 0.0],
    ]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    planner = SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "SPAWN_COLLISION",
        "profile_a",
    )
    population = planner.initial_population()

    assert len(population) == 8
    assert len({candidate.candidate_id for candidate in population}) == 8
    assert len({candidate.scene_revision for candidate in population}) == 8
    assert all(
        isinstance(candidate.mutations[0], MoveEntityOnSupport)
        and set(type(mutation) for mutation in candidate.mutations)
        <= {MoveEntityOnSupport, RotateEntityOnSupport}
        for candidate in population
    )
    assert population == SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "SPAWN_COLLISION",
        "profile_a",
    ).initial_population()

    results = [
        planner.validate_candidate(candidate, tmp_path / "search" / candidate.candidate_id)
        for candidate in population
    ]
    assert any(result.hard_ok for result in results)
    assert all(result.hard_constraints["schema"] for result in results)
    assert all(result.hard_constraints["support"] for result in results)
    assert all(result.hard_constraints["containment"] for result in results)
    assert all(result.derived_task_path for result in results)
    assert all(
        Path(result.artifact_refs[-1]).name == "static_validation.json"
        for result in results
    )
    derived_task = yaml.safe_load(
        Path(results[0].derived_task_path).read_text(encoding="utf-8")
    )["tasks"][0]
    derived_apple = next(
        region for region in derived_task["regions"] if region["object"] == "apple"
    )
    assert derived_apple["random_config"]["pos_range"][0] == derived_apple[
        "random_config"
    ]["pos_range"][1]


def test_layout_planner_moves_measured_blocker_and_evolves_from_peer_parent(
    tmp_path: Path,
):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple["size"] = [0.12, 0.08]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    _place_tool_on_table(task_path, center_xy=(1.14, 2.2))
    planner = SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "SPAWN_COLLISION",
        "profile_a",
    )

    population = planner.initial_population()
    peer_candidates = tuple(
        candidate
        for candidate in population
        if isinstance(candidate.mutations[0], MoveEntityOnSupport)
        and candidate.mutations[0].entity == "tool"
    )
    assert peer_candidates
    validations = [
        planner.validate_candidate(
            candidate,
            tmp_path / "peer_search" / candidate.candidate_id,
        )
        for candidate in peer_candidates
    ]
    assert any(validation.hard_ok for validation in validations)

    primary = peer_candidates[0]
    ranked = (
        CandidateAggregate(primary, ()),
        *(
            CandidateAggregate(candidate, ())
            for candidate in population
            if candidate != primary
        ),
    )
    generation_one = tuple(planner.evolve(ranked, 1, 8))
    assert {candidate.parent_id for candidate in generation_one} == {
        primary.candidate_id
    }
    assert {
        candidate.mutations[0].entity for candidate in generation_one
    } == {"tool"}


def test_layout_planner_never_moves_protected_target(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    _place_tool_on_table(task_path, center_xy=(1.14, 2.2))
    unprotected = SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "SPAWN_COLLISION",
        "profile_a",
    )
    target_candidate = next(
        candidate
        for candidate in unprotected.initial_population()
        if candidate.mutations[0].entity == "tool"
    )
    protected = SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "SPAWN_COLLISION",
        "profile_a",
        protected_entities={"tool"},
    )

    assert {
        candidate.mutations[0].entity
        for candidate in protected.initial_population()
    } == {"apple"}
    validation = protected.validate_candidate(
        target_candidate,
        tmp_path / "protected_target",
    )
    assert validation.failure_code == "LAYOUT_PROTECTED_ENTITY"
    assert not (tmp_path / "protected_target").exists()


def test_layout_evolver_shrinks_around_ranked_parent_with_unique_signatures(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple["size"] = [0.12, 0.08]
    apple["runtime_placement"]["offset_xy"] = [0.1, 0.2]
    apple["random_config"]["pos_range"] = [
        [0.08, 0.18, 0.0],
        [0.12, 0.22, 0.0],
    ]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    planner = SceneLayoutPlanner(
        task_path,
        arena_path,
        "apple",
        "NO_CUROBO_CANDIDATE",
        "profile_a",
    )
    initial = planner.initial_population()
    ranked = tuple(
        CandidateAggregate(candidate, ()) for candidate in initial
    )

    generation_one = tuple(planner.evolve(ranked, 1, 8))
    generation_four = tuple(
        planner.evolve(
            tuple(CandidateAggregate(candidate, ()) for candidate in generation_one),
            4,
            8,
        )
    )

    assert len(generation_one) == len(generation_four) == 8
    assert len({candidate.scene_revision for candidate in generation_one}) == 8
    assert len({candidate.scene_revision for candidate in generation_four}) == 8
    assert {candidate.parent_id for candidate in generation_one} == {
        initial[0].candidate_id
    }
    generation_one_moves = [
        candidate.mutations[0].delta_xy_m for candidate in generation_one
    ]
    generation_four_moves = [
        candidate.mutations[0].delta_xy_m for candidate in generation_four
    ]
    assert max(x for x, _ in generation_four_moves) - min(
        x for x, _ in generation_four_moves
    ) < max(x for x, _ in generation_one_moves) - min(
        x for x, _ in generation_one_moves
    )


def test_layout_planner_blocks_when_measurements_conflict(tmp_path: Path):
    task_path, arena_path = _source_documents(tmp_path)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    apple = next(
        region for region in task["tasks"][0]["regions"] if region["object"] == "apple"
    )
    apple["runtime_placement"]["offset_xy"] = [0.4, 0.2]
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    with pytest.raises(SceneLayoutBlocked) as raised:
        SceneLayoutPlanner(
            task_path,
            arena_path,
            "apple",
            "SPAWN_COLLISION",
            "profile_a",
        )

    assert raised.value.action == "block"
    assert raised.value.failure_code == "LAYOUT_RUNTIME_OFFSET_CONFLICT"
    assert "runtime_offset_xy" in raised.value.details


def test_feedback_and_trace_are_structured(tmp_path: Path):
    layout = classify_failure("SPAWN_COLLISION", "workspace")
    assert layout.action == RepairAction.MUTATE_LAYOUT
    assert "move_entity_on_support" in layout.allowed_scene_mutations
    assert classify_failure("PROBE_SPAWN_UNSTABLE", "workspace").action == RepairAction.MUTATE_LAYOUT
    assert classify_failure("NO_JOINT_GRASP_PLAN", "pick_planning").action == RepairAction.NEXT_CANDIDATE
    assert classify_failure("GRASP_CONTACT_MISSING", "pick_execution").action == RepairAction.MUTATE_SKILL
    assert classify_failure("INVALID_TASK_CONFIG", "configuration").action == RepairAction.BLOCK
    assert classify_failure("RELATION_INSERT_NOT_ADMITTED", "workspace").action == RepairAction.BLOCK
    assert classify_failure("CONTAINER_REGION_WORLD_FRAME_RANDOMIZED", "workspace").action == RepairAction.BLOCK
    assert classify_failure("PLACE_PREDICATE_FAILED", "place").action == RepairAction.MUTATE_SKILL
    assert classify_failure("mystery", "unknown").action == RepairAction.DIAGNOSE
    assert classify_failure("NONE", "success").action == RepairAction.KEEP

    trace_path = tmp_path / "trace.jsonl"
    TraceWriter(trace_path).append(
        TraceEvent(
            TraceContext(
                run_id="run_1",
                variant_id="candidate_0",
                attempt_id="00",
                seed=3,
                profile_id="profile_a",
                scene_revision="scene_1",
                world_revision=2,
            ),
            stage="collision_audit",
            status="failed",
            failure_code="SPAWN_COLLISION",
            artifact_refs=("collision_world_audit.json",),
        )
    )
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["run_id"] == "run_1"
    assert record["variant_id"] == "candidate_0"
    assert record["attempt_id"] == "00"
    assert record["failure_code"] == "SPAWN_COLLISION"


def test_predicate_payload_requires_exact_compiled_subtask_coverage():
    document = {
        "tasks": [
            {
                "metadata": {
                    "agent_plan": {
                        "subtasks": [
                            {
                                "subtask_id": "cup_transfer",
                                "center_object": "cup",
                                "target_object": "tray",
                                "relation": "inside",
                            },
                            {
                                "subtask_id": "spoon_transfer",
                                "center_object": "spoon",
                                "target_object": "tray",
                                "relation": "inside",
                            },
                        ]
                    }
                }
            }
        ]
    }

    def result(subtask_id: str, manipulated: str) -> dict:
        return {
            "predicate_id": f"relation_{subtask_id}",
            "subtask_id": subtask_id,
            "skill": "place",
            "objects": [manipulated, "tray"],
            "relation": "inside",
            "terminal_success": True,
            "success": True,
            "checks": {"support_gap_ok": True},
            "measurements": {"support_gap_m": 0.0},
            "thresholds": {"support_gap_tolerance_m": 0.006},
        }

    cup = result("cup_transfer", "cup")
    spoon = result("spoon_transfer", "spoon")
    assert _predicate_payload_error([cup, spoon], document) is None
    assert "cover every compiled subtask" in str(
        _predicate_payload_error([cup], document)
    )
    assert "unknown or duplicate" in str(
        _predicate_payload_error([cup, cup], document)
    )
    assert "no predicate_id" in str(
        _predicate_payload_error([{**cup, "predicate_id": ""}, spoon], document)
    )
    assert "invalid measurements" in str(
        _predicate_payload_error([{**cup, "measurements": {}}, spoon], document)
    )


def test_strict_evidence_requires_collision_audit_and_data(tmp_path: Path):
    attempt = tmp_path / "attempt"
    episode = attempt / "data" / "episode"
    profile_path = ROOT / "workflows/simbox/core/configs/robots/fr3.yaml"
    profile = load_robot_profile_for_task(
        {"robot_config_file": str(profile_path)}, tmp_path / "task.yaml"
    )
    num_steps = 3
    camera_keys = [
        *(f"images.rgb.{camera.save_name}" for camera in profile.camera_rig),
        "images.rgb.global",
    ]
    for camera_key in camera_keys:
        (episode / camera_key).mkdir(parents=True)
        imageio.mimsave(
            episode / camera_key / "demo.mp4",
            [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(num_steps)],
            fps=15,
        )
    proprio_keys = [
        "states.joint.position",
        "states.gripper.position",
        "states.gripper.pose",
    ]
    action_keys = [
        "master_actions.joint.position",
        "master_actions.gripper.position",
        "master_actions.gripper.openness",
        "master_actions.gripper.pose",
        "actions.joint.position",
        "actions.gripper.position",
        "actions.gripper.pose",
        "actions.gripper.openness",
    ]
    lmdb_path = episode / "lmdb"
    lmdb_path.mkdir()
    environment = lmdb.open(str(lmdb_path), map_size=4 * 1024 * 1024)
    encoded_image = imageio.imwrite(
        "<bytes>",
        np.zeros((16, 16, 3), dtype=np.uint8),
        format="jpg",
    )
    with environment.begin(write=True) as transaction:
        for key in [*proprio_keys, *action_keys]:
            if "joint.position" in key:
                sample = np.zeros(7, dtype=np.float32)
            elif "gripper.pose" in key:
                sample = np.zeros(6, dtype=np.float32)
            elif "gripper.position" in key:
                sample = np.zeros(1, dtype=np.float32)
            else:
                sample = 0.0
            samples = [
                sample.copy() if hasattr(sample, "copy") else sample
                for _ in range(num_steps)
            ]
            transaction.put(
                key.encode("utf-8"),
                pickle.dumps(samples),
            )
        for camera_key in camera_keys:
            for frame_id in range(num_steps):
                transaction.put(
                    f"{camera_key}/{frame_id:04d}".encode("utf-8"),
                    pickle.dumps(encoded_image),
                )
    environment.close()
    with (episode / "meta_info.pkl").open("wb") as stream:
        pickle.dump(
            {
                "num_steps": num_steps,
                "keys": {
                    "proprio_data": [key.encode("utf-8") for key in proprio_keys],
                    "action_data": [key.encode("utf-8") for key in action_keys],
                    **{
                        camera_key: [
                            f"{camera_key}/{frame_id:04d}".encode("utf-8")
                            for frame_id in range(num_steps)
                        ]
                        for camera_key in camera_keys
                    },
                },
                "image_valid_step_ids": {
                    camera_key: list(range(num_steps)) for camera_key in camera_keys
                },
            },
            stream,
        )
    (episode / "collision_world_audit.json").write_text(
        json.dumps(
            {
                "world_revision": 7,
                "physics_curobo_difference": {
                    "fr3/left": {"missing_in_curobo": [], "unexpected_in_curobo": []}
                }
            }
        ),
        encoding="utf-8",
    )
    (episode / "safety_events.jsonl").write_text("", encoding="utf-8")
    event_path = attempt / "episode_events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "event": "episode_saved",
                "finalized": True,
                "status": "success",
                "task_predicate_success": True,
                "predicate_results": [
                    {
                        "predicate_id": "relation_00",
                        "subtask_id": "transfer",
                        "skill": "place",
                        "objects": ["cup", "tray"],
                        "relation": "inside",
                        "terminal_success": True,
                        "success": True,
                        "checks": {"support_gap_ok": True},
                        "measurements": {"support_gap_m": 0.0},
                        "thresholds": {"support_gap_tolerance_m": 0.006},
                    }
                ],
                "primary_episode_dir": str(episode),
                "num_steps": num_steps,
                "video_stream_count": len(camera_keys),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = attempt / "stdout.log"
    log_path.write_text("Task is successful\n", encoding="utf-8")
    compiled_task_path = attempt / "task.yaml"
    compiled_task_path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "robots": [
                            {
                                "name": "fr3",
                                "robot_config_file": str(profile_path),
                            }
                        ],
                        "metadata": {
                            "agent_plan": {
                                "execution_variant_id": "fr3__left",
                                "robot_profile_id": profile.profile_id,
                                "robot_profile_hash": profile.profile_hash,
                                "scene_revision": "source",
                                "subtasks": [
                                    {
                                        "subtask_id": "transfer",
                                        "center_object": "cup",
                                        "target_object": "tray",
                                        "arm": "left",
                                        "relation": "inside",
                                    }
                                ],
                            }
                        },
                        "cameras": [
                            {"save_name": camera_key.removeprefix("images.rgb.")}
                            for camera_key in camera_keys
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_task = tmp_path / "source_task.yaml"
    source_arena = tmp_path / "source_arena.yaml"
    source_task.write_text("tasks: []\n", encoding="utf-8")
    source_arena.write_text("fixtures: []\n", encoding="utf-8")
    source_snapshot_path = tmp_path / "source_snapshot.json"
    source_task_hash = hashlib.sha256(source_task.read_bytes()).hexdigest()
    source_arena_hash = hashlib.sha256(source_arena.read_bytes()).hexdigest()
    source_members = [
        SourceMember(str(source_task.resolve()), "source_task", source_task_hash),
        SourceMember(str(source_arena.resolve()), "source_arena", source_arena_hash),
    ]
    source_hash = canonical_source_hash(source_members)
    source_snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
                "source_task": str(source_task.resolve()),
                "source_arena": str(source_arena.resolve()),
                "source_task_hash": source_task_hash,
                "source_arena_hash": source_arena_hash,
                "members": [member.to_dict() for member in source_members],
                "source_hash": source_hash,
            }
        ),
        encoding="utf-8",
    )
    expected_identity = ExecutionIdentity(
        run_id="evidence_run",
        variant_id="fr3__left",
        seed=0,
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        source_hash=source_hash,
        scene_revision="source",
    )
    episode_event = json.loads(event_path.read_text(encoding="utf-8"))
    episode_event.update(expected_identity.to_dict())
    episode_event["world_revision"] = 7
    event_path.write_text(json.dumps(episode_event) + "\n", encoding="utf-8")
    evidence = collect_evidence(
        "candidate:0",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert evidence.task_success is True
    assert classify_evidence(evidence).failure_code == "NONE"

    (episode / "collision_world_audit.json").write_text(
        json.dumps(
            {
                "world_revision": 7,
                "physics_curobo_difference": {
                    "fr3/right": {
                        "missing_in_curobo": [],
                        "unexpected_in_curobo": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    wrong_controller_audit = collect_evidence(
        "candidate:wrong_controller_audit",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert wrong_controller_audit.task_success is False
    assert classify_evidence(wrong_controller_audit).failure_code == (
        "PHYSICS_CUROBO_WORLD_MISMATCH"
    )
    (episode / "collision_world_audit.json").write_text(
        json.dumps(
            {
                "world_revision": 7,
                "physics_curobo_difference": {
                    "fr3/left": {
                        "missing_in_curobo": [],
                        "unexpected_in_curobo": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    event_path.write_text(
        json.dumps(episode_event) + "\n{malformed\n",
        encoding="utf-8",
    )
    malformed_event = collect_evidence(
        "candidate:malformed_event",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert malformed_event.task_success is False
    assert classify_evidence(malformed_event).failure_code == "EVENT_ARTIFACT_INVALID"
    event_path.write_text(json.dumps(episode_event) + "\n", encoding="utf-8")

    (episode / "safety_events.jsonl").write_text("{malformed\n", encoding="utf-8")
    malformed_safety = collect_evidence(
        "candidate:malformed_safety",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert malformed_safety.task_success is False
    assert classify_evidence(malformed_safety).failure_code == "SAFETY_ARTIFACT_INVALID"
    (episode / "safety_events.jsonl").write_text("", encoding="utf-8")

    environment = lmdb.open(str(lmdb_path), map_size=4 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(
            b"states.joint.position",
            pickle.dumps([np.full(7, np.nan) for _ in range(num_steps)]),
        )
    environment.close()
    nonfinite_data = collect_evidence(
        "candidate:nonfinite_data",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert nonfinite_data.task_success is False
    assert classify_evidence(nonfinite_data).failure_code == "DATA_INTEGRITY_FAILED"
    environment = lmdb.open(str(lmdb_path), map_size=4 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(
            b"states.joint.position",
            pickle.dumps([np.zeros(7) for _ in range(num_steps)]),
        )
        transaction.put(
            f"{camera_keys[0]}/0000".encode("utf-8"),
            pickle.dumps(b"not-an-image"),
        )
    environment.close()
    corrupt_image = collect_evidence(
        "candidate:corrupt_image",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert corrupt_image.task_success is False
    assert classify_evidence(corrupt_image).failure_code == "DATA_INTEGRITY_FAILED"
    environment = lmdb.open(str(lmdb_path), map_size=4 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(
            f"{camera_keys[0]}/0000".encode("utf-8"),
            pickle.dumps(encoded_image),
        )
    environment.close()

    attributed_event = dict(episode_event)
    attributed_event["status"] = "failed"
    attributed_event["failure_reason"] = "SPAWN_COLLISION"
    attributed_event["failing_subtask_id"] = "transfer"
    event_path.write_text(json.dumps(attributed_event) + "\n", encoding="utf-8")
    attributed = collect_evidence(
        "candidate:attributed_failure",
        attempt,
        event_path,
        log_path,
        1,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    attributed_diagnosis = classify_evidence(attributed)
    assert attributed.failing_subtask_id == "transfer"
    assert attributed_diagnosis.failing_subtask_id == "transfer"
    event_path.write_text(json.dumps(episode_event) + "\n", encoding="utf-8")

    (episode / "safety_events.jsonl").unlink()
    missing_safety = collect_evidence(
        "candidate:missing_safety",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert missing_safety.task_success is False
    assert missing_safety.failure_reason == "SAFETY_ARTIFACT_MISSING"
    assert classify_evidence(missing_safety).retryable is False
    (episode / "safety_events.jsonl").write_text("", encoding="utf-8")

    nested_lmdb = episode / "nested" / "lmdb"
    nested_lmdb.parent.mkdir()
    lmdb_path.rename(nested_lmdb)
    nested_attempt = tmp_path / "attempt_nested_lmdb"
    nested_attempt.mkdir()
    nested_event_path = nested_attempt / "episode_events.jsonl"
    nested_event_path.write_text(event_path.read_text(encoding="utf-8"), encoding="utf-8")
    nested_log_path = nested_attempt / "stdout.log"
    nested_log_path.write_text("Task is successful\n", encoding="utf-8")
    nested_evidence = collect_evidence(
        "candidate:nested_lmdb",
        attempt,
        nested_event_path,
        nested_log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert nested_evidence.task_success is False
    assert nested_evidence.failure_reason == "DATA_INTEGRITY_FAILED"
    nested_lmdb.rename(lmdb_path)

    forged_event = json.loads(event_path.read_text(encoding="utf-8"))
    forged_event["variant_id"] = "forged_variant"
    forged_attempt = tmp_path / "attempt_forged_identity"
    forged_attempt.mkdir()
    forged_event_path = forged_attempt / "episode_events.jsonl"
    forged_event_path.write_text(json.dumps(forged_event) + "\n", encoding="utf-8")
    forged_log_path = forged_attempt / "stdout.log"
    forged_log_path.write_text("Task is successful\n", encoding="utf-8")
    forged_evidence = collect_evidence(
        "candidate:forged_identity",
        attempt,
        forged_event_path,
        forged_log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert forged_evidence.task_success is False
    assert forged_evidence.failure_reason == "IDENTITY_MISMATCH"
    assert forged_evidence.identity is None
    assert classify_evidence(forged_evidence).retryable is False

    predicate_failed_event = json.loads(event_path.read_text(encoding="utf-8"))
    predicate_failed_event["task_predicate_success"] = False
    predicate_failed_event["predicate_results"][0]["success"] = False
    predicate_failed_event["failing_subtask_id"] = "transfer"
    predicate_attempt = tmp_path / "attempt_predicate_failed"
    predicate_attempt.mkdir()
    predicate_event_path = predicate_attempt / "episode_events.jsonl"
    predicate_event_path.write_text(
        json.dumps(predicate_failed_event) + "\n", encoding="utf-8"
    )
    predicate_log_path = predicate_attempt / "stdout.log"
    predicate_log_path.write_text("Task is successful\n", encoding="utf-8")
    predicate_failed = collect_evidence(
        "candidate:predicate_failed",
        attempt,
        predicate_event_path,
        predicate_log_path,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert predicate_failed.task_success is False
    assert predicate_failed.failure_reason == "PLACE_PREDICATE_FAILED"
    assert predicate_failed.failing_subtask_id == "transfer"

    (episode / "collision_world_audit.json").unlink()
    second = tmp_path / "attempt_missing_audit"
    second.mkdir()
    second_event = second / "episode_events.jsonl"
    second_event.write_text(event_path.read_text(encoding="utf-8"), encoding="utf-8")
    second_log = second / "stdout.log"
    second_log.write_text("Task is successful\n", encoding="utf-8")
    failed = collect_evidence(
        "candidate:1",
        attempt,
        second_event,
        second_log,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert failed.task_success is False
    assert failed.failure_reason == "COLLISION_WORLD_AUDIT_MISSING"
    assert classify_evidence(failed).retryable is False

    (episode / "collision_world_audit.json").write_text(
        json.dumps(
            {
                "world_revision": 7,
                "physics_curobo_difference": {
                    "fr3/left": {
                        "missing_in_curobo": [],
                        "unexpected_in_curobo": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    corrupt_camera = episode / "images.rgb.global" / "demo.mp4"
    corrupt_camera.write_bytes(b"not an mp4")
    corrupt_attempt = tmp_path / "attempt_corrupt_video"
    corrupt_attempt.mkdir()
    corrupt_event = corrupt_attempt / "episode_events.jsonl"
    corrupt_event.write_text(event_path.read_text(encoding="utf-8"), encoding="utf-8")
    corrupt_log = corrupt_attempt / "stdout.log"
    corrupt_log.write_text("Task is successful\n", encoding="utf-8")
    corrupt = collect_evidence(
        "candidate:corrupt_video",
        attempt,
        corrupt_event,
        corrupt_log,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert corrupt.task_success is False
    assert corrupt.failure_reason == "DATA_INTEGRITY_FAILED"

    imageio.mimsave(
        corrupt_camera,
        [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(num_steps)],
        fps=15,
    )
    source_task.write_text("tasks: [{name: modified}]\n", encoding="utf-8")
    changed_source_attempt = tmp_path / "attempt_changed_source"
    changed_source_attempt.mkdir()
    changed_source_event = changed_source_attempt / "episode_events.jsonl"
    changed_source_event.write_text(
        event_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    changed_source_log = changed_source_attempt / "stdout.log"
    changed_source_log.write_text("Task is successful\n", encoding="utf-8")
    changed_source = collect_evidence(
        "candidate:changed_source",
        attempt,
        changed_source_event,
        changed_source_log,
        0,
        False,
        expected_identity=expected_identity,
        data_generation_required=True,
        robot_profile_path=profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
    )
    assert changed_source.task_success is False
    assert changed_source.failure_reason == "SOURCE_INTEGRITY_FAILED"
