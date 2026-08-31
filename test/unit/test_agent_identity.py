"""Stable inventory identity and exact synthetic-scene selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import agent.orchestrator as orchestrator_module
from agent.contracts import ExecutionIdentity, ResolutionDecision
from agent.inventory import (
    INVENTORY_SCHEMA_VERSION,
    build_inventory,
    load_or_build_inventory,
    read_inventory,
    write_inventory,
)
from agent.orchestrator import AgentOrchestrator
from agent.resolver import AgentDecisionError, TaskResolver
from workflows.simbox.core.utils.episode_event_writer import emit_episode_saved


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_PROFILE = REPO_ROOT / "workflows/simbox/core/configs/robots/split_aloha.yaml"


def test_cli_backend_override_becomes_effective_nested_runtime_setting(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(orchestrator_module, "_create_backend", lambda *_args: object())
    monkeypatch.setattr(orchestrator_module, "TaskResolver", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(orchestrator_module, "RetentionManager", lambda *_args: object())
    original = {
        "execution": {"simulator_backend": "docker"},
        "generation": {"random_num": 1, "seed": 0},
    }

    orchestrator = AgentOrchestrator(
        settings=original,
        simulator_backend="conda",
        conda_env="interndata-isaac6",
        run_root=tmp_path,
    )

    assert orchestrator.settings["execution"]["simulator_backend"] == "conda"
    assert orchestrator.settings["execution"]["conda_env"] == "interndata-isaac6"
    assert original["execution"]["simulator_backend"] == "docker"


def _write_task(path: Path, *, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": name,
                        "task": "Banana",
                        "robots": [
                            {
                                "name": "robot_instance",
                                "robot_config_file": str(ROBOT_PROFILE),
                            }
                        ],
                        "delivery_active_objects": ["cup"],
                        "objects": [
                            {
                                "name": "cup",
                                "target_class": "RigidObject",
                                "rigidbody": True,
                                "collision_enabled": True,
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


def _same_scene_tasks(root: Path) -> tuple[Path, Path]:
    first = _write_task(
        root / "scene_a/assets/basic/source_a/simbox_task.yaml",
        name="shared_name",
    )
    second = _write_task(
        root / "scene_a/assets/basic/source_b/simbox_task.yaml",
        name="shared_name",
    )
    return first, second


def test_inventory_task_id_is_stable_and_unique_per_source_path(tmp_path: Path):
    root = tmp_path / "scenes"
    first, second = _same_scene_tasks(root)

    initial = build_inventory([root])
    repeated = build_inventory([root])
    by_source = {Path(item.source_task): item.task_id for item in initial}

    assert len(initial) == 2
    assert by_source[first.resolve()] != by_source[second.resolve()]
    assert by_source == {
        Path(item.source_task): item.task_id for item in repeated
    }


def test_inventory_cache_rebuilds_when_identity_schema_changes(tmp_path: Path):
    root = tmp_path / "scenes"
    _same_scene_tasks(root)
    index = tmp_path / "inventory.json"
    index.write_text(json.dumps({"version": 2, "tasks": []}), encoding="utf-8")

    manifests = load_or_build_inventory(index, [root])

    assert len(manifests) == 2
    assert json.loads(index.read_text(encoding="utf-8"))["version"] == INVENTORY_SCHEMA_VERSION
    assert read_inventory(index) == manifests


def test_synthetic_manifest_uses_exact_selected_source_and_stable_identity(
    tmp_path: Path,
):
    root = tmp_path / "scenes"
    _same_scene_tasks(root)
    candidates = build_inventory([root])
    selected = candidates[1]
    decision = ResolutionDecision.from_dict(
        {
            "mode": "reuse_scene_new_task",
            "selected_task_id": selected.task_id,
            "selected_source_task": selected.source_task,
            "selected_scene_id": selected.scene_id,
            "object_role_overrides": {"cup": "manipulated"},
            "decision_basis": "reuse this exact scene source",
        }
    )
    resolver = TaskResolver(backend=object())

    source = resolver.select_source_manifest(decision, list(reversed(candidates)))
    first = resolver.build_synthetic_manifest(decision, source)
    second = resolver.build_synthetic_manifest(decision, source)

    assert source.source_task == selected.source_task
    assert first.source_task == selected.source_task
    assert first.task_id == second.task_id
    assert first.task_id.startswith("synthetic_")

    mismatched = ResolutionDecision.from_dict(
        {
            **decision.to_dict(),
            "selected_source_task": candidates[0].source_task,
        }
    )
    with pytest.raises(AgentDecisionError, match="exactly one candidate"):
        resolver.select_source_manifest(mismatched, candidates)


def test_inventory_round_trip_uses_current_identity_schema(tmp_path: Path):
    root = tmp_path / "scenes"
    _same_scene_tasks(root)
    manifests = build_inventory([root])
    index = write_inventory(manifests, tmp_path / "inventory.json")

    payload = json.loads(index.read_text(encoding="utf-8"))

    assert payload["version"] == INVENTORY_SCHEMA_VERSION
    assert read_inventory(index) == manifests


@pytest.mark.parametrize(
    ("simulator_backend", "expected_command"),
    [
        ("docker", ["bash", "scripts/docker/up_simbox_isaac.sh"]),
        ("conda", ["bash", "scripts/simbox/run_simbox_task.sh"]),
    ],
)
def test_simbox_subprocess_receives_complete_execution_identity(
    tmp_path: Path,
    monkeypatch,
    simulator_backend,
    expected_command,
):
    orchestrator = object.__new__(AgentOrchestrator)
    orchestrator.gpu = 3
    orchestrator.random_num = 1
    orchestrator.conda_env = "interndata-isaac6"
    orchestrator.simulator_backend = simulator_backend
    orchestrator.timeout_sec = 30
    orchestrator.settings = {
        "debug": {},
        "execution": {
            "docker": {},
            "conda": {},
        },
    }
    identity = ExecutionIdentity(
        run_id="run_1",
        variant_id="variant_1",
        seed=4,
        profile_id="profile_1",
        profile_hash="a" * 64,
        source_hash="b" * 64,
        scene_revision="scene_1",
    )
    config_path = tmp_path / "task.yaml"
    config_path.write_text("tasks: []\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]

        @staticmethod
        def wait(timeout):
            assert timeout == 30
            return 0

    monkeypatch.setattr(orchestrator_module.subprocess, "Popen", FakeProcess)

    return_code, timed_out, _, _ = orchestrator._run_simbox(
        config_path,
        attempt_dir,
        identity,
        data_generation=False,
    )

    assert return_code == 0
    assert timed_out is False
    assert captured["command"] == expected_command
    env = captured["env"]
    assert env["INTERNDATA_SIMULATOR_BACKEND"] == simulator_backend
    assert env["INTERNDATA_SCREENSHOT_DIR"] == str(
        (attempt_dir / "screenshots").resolve()
    )
    assert {
        "INTERNDATA_RUN_ID": env["INTERNDATA_RUN_ID"],
        "INTERNDATA_VARIANT_ID": env["INTERNDATA_VARIANT_ID"],
        "INTERNDATA_RANDOM_SEED": env["INTERNDATA_RANDOM_SEED"],
        "INTERNDATA_PROFILE_ID": env["INTERNDATA_PROFILE_ID"],
        "INTERNDATA_PROFILE_HASH": env["INTERNDATA_PROFILE_HASH"],
        "INTERNDATA_SOURCE_HASH": env["INTERNDATA_SOURCE_HASH"],
        "INTERNDATA_SCENE_REVISION": env["INTERNDATA_SCENE_REVISION"],
    } == {
        "INTERNDATA_RUN_ID": "run_1",
        "INTERNDATA_VARIANT_ID": "variant_1",
        "INTERNDATA_RANDOM_SEED": "4",
        "INTERNDATA_PROFILE_ID": "profile_1",
        "INTERNDATA_PROFILE_HASH": "a" * 64,
        "INTERNDATA_SOURCE_HASH": "b" * 64,
        "INTERNDATA_SCENE_REVISION": "scene_1",
    }
    command = json.loads((attempt_dir / "command.json").read_text(encoding="utf-8"))
    assert command["command"] == expected_command
    assert command["env"]["INTERNDATA_SIMULATOR_BACKEND"] == simulator_backend
    assert command["env"]["INTERNDATA_SCREENSHOT_DIR"] == str(
        (attempt_dir / "screenshots").resolve()
    )


def test_episode_event_persists_execution_and_world_identity(
    tmp_path: Path,
    monkeypatch,
):
    event_path = tmp_path / "episode_events.jsonl"
    values = {
        "INTERNDATA_EPISODE_EVENT_PATH": str(event_path),
        "INTERNDATA_RUN_ID": "run_1",
        "INTERNDATA_VARIANT_ID": "variant_1",
        "INTERNDATA_RANDOM_SEED": "4",
        "INTERNDATA_PROFILE_ID": "profile_1",
        "INTERNDATA_PROFILE_HASH": "a" * 64,
        "INTERNDATA_SOURCE_HASH": "b" * 64,
        "INTERNDATA_SCENE_REVISION": "scene_1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    emit_episode_saved(
        status="failed",
        episode_dirs=[],
        num_steps=0,
        failing_subtask_id="transfer",
        world_revision=9,
    )

    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert {
        key: event[key]
        for key in (
            "run_id",
            "variant_id",
            "seed",
            "profile_id",
            "profile_hash",
            "source_hash",
            "scene_revision",
            "world_revision",
        )
    } == {
        "run_id": "run_1",
        "variant_id": "variant_1",
        "seed": 4,
        "profile_id": "profile_1",
        "profile_hash": "a" * 64,
        "source_hash": "b" * 64,
        "scene_revision": "scene_1",
        "world_revision": 9,
    }
    assert event["failing_subtask_id"] == "transfer"
