"""Source-scene provenance includes referenced assets and robot bytes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import agent.tools.source_integrity as source_integrity_module
from agent.contracts import ExecutionIdentity
from agent.evidence import collect_evidence
from agent.tools.source_integrity import (
    build_source_snapshot,
    verify_source_snapshot,
    write_source_snapshot,
)
from workflows.simbox.core.robots.profile import load_robot_profile


def _write_profile(path: Path, robot_asset: Path) -> Path:
    canonical_profile = Path(
        "workflows/simbox/core/configs/robots/fr3.yaml"
    )
    document = yaml.safe_load(canonical_profile.read_text(encoding="utf-8"))
    document["profile_id"] = "test_support_mounted_v1"
    document["path"] = str(robot_asset)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    asset_root = tmp_path / "assets"
    object_asset = asset_root / "objects/cup.usd"
    fixture_asset = asset_root / "fixtures/table.usd"
    robot_asset = tmp_path / "robot.usd"
    for path, content in (
        (object_asset, b"object"),
        (fixture_asset, b"fixture"),
        (robot_asset, b"robot"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    profile_path = _write_profile(tmp_path / "robot_profile.yaml", robot_asset)
    arena_path = tmp_path / "arena.yaml"
    arena_path.write_text(
        yaml.safe_dump(
            {
                "fixtures": [
                    {
                        "name": "table",
                        "target_class": "GeometryObject",
                        "path": "fixtures/table.usd",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "source_task",
                        "asset_root": str(asset_root),
                        "arena_file": str(arena_path),
                        "robots": [
                            {
                                "name": "robot",
                                "robot_config_file": str(profile_path),
                            }
                        ],
                        "objects": [
                            {
                                "name": "cup",
                                "target_class": "RigidObject",
                                "path": "objects/cup.usd",
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task_path, arena_path, object_asset, robot_asset


def _snapshot(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    task_path, arena_path, object_asset, robot_asset = _source_files(tmp_path)
    profile_path = tmp_path / "robot_profile.yaml"
    snapshot = build_source_snapshot(
        task_path,
        arena_path,
        [profile_path],
        repo_root=tmp_path,
    )
    snapshot_path = write_source_snapshot(snapshot, tmp_path / "source_snapshot.json")
    return snapshot_path, snapshot, object_asset, robot_asset


def _evidence_inputs(tmp_path: Path, source_hash: str):
    attempt = tmp_path / "attempt"
    episode = attempt / "data" / "episode_0"
    episode.mkdir(parents=True)
    identity = ExecutionIdentity(
        run_id="source_integrity_run",
        variant_id="test_variant",
        seed=0,
        profile_id="test_support_mounted_v1",
        profile_hash=load_robot_profile(tmp_path / "robot_profile.yaml").profile_hash,
        source_hash=source_hash,
        scene_revision="source",
    )
    (episode / "collision_world_audit.json").write_text(
        json.dumps(
            {
                "world_revision": 1,
                "physics_curobo_difference": {
                    "robot/left": {
                        "missing_in_curobo": [],
                        "unexpected_in_curobo": [],
                    }
                },
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
                        "predicate_id": "relation_0",
                        "subtask_id": "transfer",
                        "skill": "place",
                        "objects": ["cup", "tray"],
                        "relation": "on",
                        "terminal_success": True,
                        "success": True,
                        "checks": {"support_gap_ok": True},
                        "measurements": {"support_gap_m": 0.0},
                        "thresholds": {"support_gap_tolerance_m": 0.006},
                    }
                ],
                "primary_episode_dir": str(episode),
                "world_revision": 1,
                **identity.to_dict(),
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
                                "name": "robot",
                                "robot_config_file": str(tmp_path / "robot_profile.yaml"),
                            }
                        ],
                        "metadata": {
                            "agent_plan": {
                                "execution_variant_id": identity.variant_id,
                                "robot_profile_id": identity.profile_id,
                                "robot_profile_hash": identity.profile_hash,
                                "scene_revision": identity.scene_revision,
                                "subtasks": [
                                    {
                                        "subtask_id": "transfer",
                                        "center_object": "cup",
                                        "target_object": "tray",
                                        "relation": "on",
                                        "arm": "left",
                                    }
                                ],
                            }
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return attempt, event_path, log_path, compiled_task_path, identity


def test_source_snapshot_hashes_scene_and_robot_members(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, snapshot, _, _ = _snapshot(tmp_path)

    roles = {member["role"] for member in snapshot["members"]}
    result = verify_source_snapshot(snapshot_path, snapshot["source_hash"])

    assert {
        "source_task",
        "source_arena",
        "task.objects[cup].path",
        "arena.fixtures[table].path",
        "robot_profile:test_support_mounted_v1",
        "robot_canonical_asset:test_support_mounted_v1",
        "robot_selected_asset:test_support_mounted_v1",
    } <= roles
    assert result.source_unchanged is True
    assert result.identity_consistent is True
    assert result.errors == ()


def test_source_snapshot_rejects_changed_object_asset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, snapshot, object_asset, _ = _snapshot(tmp_path)
    object_asset.write_bytes(b"changed object")

    result = verify_source_snapshot(snapshot_path, snapshot["source_hash"])

    assert result.source_unchanged is False
    assert result.identity_consistent is False
    assert "task.objects[cup].path" in result.errors[0]


def test_source_snapshot_rejects_changed_or_missing_robot_asset(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, snapshot, _, robot_asset = _snapshot(tmp_path)
    robot_asset.write_bytes(b"changed robot")

    changed = verify_source_snapshot(snapshot_path, snapshot["source_hash"])
    assert changed.source_unchanged is False
    assert "robot_canonical_asset:test_support_mounted_v1" in changed.errors[0]

    robot_asset.unlink()
    missing = verify_source_snapshot(snapshot_path, snapshot["source_hash"])
    assert missing.source_unchanged is False
    assert "is missing" in missing.errors[0]


def test_source_snapshot_identity_hash_is_checked_separately(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, _, _, _ = _snapshot(tmp_path)

    result = verify_source_snapshot(snapshot_path, "0" * 64)

    assert result.source_unchanged is True
    assert result.identity_consistent is False
    assert result.errors == ("source snapshot does not match the execution identity",)


@pytest.mark.parametrize("changed_member", ["object", "robot"])
def test_collect_evidence_rejects_changed_source_asset(
    tmp_path: Path,
    monkeypatch,
    changed_member: str,
):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, snapshot, object_asset, robot_asset = _snapshot(tmp_path)
    attempt, event_path, log_path, compiled_task_path, identity = _evidence_inputs(
        tmp_path,
        snapshot["source_hash"],
    )

    unchanged = collect_evidence(
        "source:unchanged",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=identity,
        robot_profile_path=tmp_path / "robot_profile.yaml",
        compiled_task_path=compiled_task_path,
        source_snapshot_path=snapshot_path,
    )
    assert unchanged.task_success is True

    (object_asset if changed_member == "object" else robot_asset).write_bytes(b"mutated")
    changed = collect_evidence(
        f"source:changed_{changed_member}",
        attempt,
        event_path,
        log_path,
        0,
        False,
        expected_identity=identity,
        robot_profile_path=tmp_path / "robot_profile.yaml",
        compiled_task_path=compiled_task_path,
        source_snapshot_path=snapshot_path,
    )

    assert changed.task_success is False
    assert changed.failure_reason == "SOURCE_INTEGRITY_FAILED"
    assert any("source member hash changed" in error for error in changed.identity_errors)


def test_source_snapshot_rejects_forged_member_digest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: Path(profile.path).resolve(),
    )
    snapshot_path, snapshot, _, _ = _snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["members"][0]["sha256"] = "f" * 64
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_source_snapshot(snapshot_path, snapshot["source_hash"])

    assert result.source_unchanged is False
    assert result.identity_consistent is False
    assert result.errors == ("source snapshot aggregate hash is invalid",)


@pytest.mark.parametrize(
    ("member_name", "expected_role"),
    [
        ("curobo", "robot_curobo_config:test_support_mounted_v1:left"),
        ("urdf", "robot_curobo_urdf_path:test_support_mounted_v1:left"),
        (
            "collision_spheres",
            "robot_curobo_collision_spheres:test_support_mounted_v1:left",
        ),
        ("camera", "robot_camera_config:test_support_mounted_v1:hand:0"),
    ],
)
def test_source_snapshot_hashes_robot_runtime_dependencies(
    tmp_path: Path,
    monkeypatch,
    member_name: str,
    expected_role: str,
):
    task_path, arena_path, _, robot_asset = _source_files(tmp_path)
    profile_path = tmp_path / "robot_profile.yaml"
    profile_document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    urdf_path = tmp_path / "robot.urdf"
    collision_spheres_path = tmp_path / "collision_spheres.yml"
    curobo_path = tmp_path / "robot_curobo.yml"
    camera_path = tmp_path / "camera.yaml"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")
    collision_spheres_path.write_text("collision_spheres: {}\n", encoding="utf-8")
    curobo_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "urdf_path": str(urdf_path),
                        "collision_spheres": str(collision_spheres_path),
                        "use_usd_kinematics": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    camera_path.write_text("camera_type: TestCamera\n", encoding="utf-8")
    profile_document["arms"]["left"]["curobo_file"] = str(curobo_path)
    for camera in profile_document["camera_rig"]:
        camera["camera_file"] = str(camera_path)
    profile_path.write_text(
        yaml.safe_dump(profile_document, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: robot_asset.resolve(),
    )

    snapshot = build_source_snapshot(
        task_path,
        arena_path,
        [profile_path],
        repo_root=tmp_path,
    )
    snapshot_path = write_source_snapshot(snapshot, tmp_path / "source_snapshot.json")
    roles = {member["role"] for member in snapshot["members"]}
    assert expected_role in roles

    dependency_paths = {
        "curobo": curobo_path,
        "urdf": urdf_path,
        "collision_spheres": collision_spheres_path,
        "camera": camera_path,
    }
    dependency_paths[member_name].write_bytes(b"changed dependency")
    result = verify_source_snapshot(snapshot_path, snapshot["source_hash"])

    assert result.source_unchanged is False
    assert expected_role in result.errors[0]


def test_source_snapshot_hashes_resolved_usd_layers_and_textures(
    tmp_path: Path,
    monkeypatch,
):
    task_path, arena_path, object_asset, robot_asset = _source_files(tmp_path)
    object_asset.with_name("source.usda").write_text("#usda 1.0\n", encoding="utf-8")
    texture_path = object_asset.with_name("albedo.png")
    texture_path.write_bytes(b"texture")
    object_asset.write_text(
        "#usda 1.0\n"
        "(\n"
        "    subLayers = [@source.usda@]\n"
        ")\n"
        'def "Root" {\n'
        "    custom asset file = @albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        source_integrity_module,
        "resolve_robot_asset_path",
        lambda profile: robot_asset.resolve(),
    )

    snapshot = build_source_snapshot(
        task_path,
        arena_path,
        [tmp_path / "robot_profile.yaml"],
        repo_root=tmp_path,
    )
    snapshot_path = write_source_snapshot(snapshot, tmp_path / "source_snapshot.json")
    object_dependencies = [
        member
        for member in snapshot["members"]
        if member["role"].startswith("task.objects[cup].path.usd_dependency")
    ]

    assert {Path(member["path"]).name for member in object_dependencies} == {
        "source.usda",
        "albedo.png",
    }
    texture_path.write_bytes(b"changed texture")
    result = verify_source_snapshot(snapshot_path, snapshot["source_hash"])
    assert result.source_unchanged is False
    assert "task.objects[cup].path.usd_dependency" in result.errors[0]
