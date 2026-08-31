"""Offline API tests for workspace planning-gate orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "scripts/simbox/validate_workspace_candidates.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "workspace_candidate_validator_contract", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pick_place_probe_api_passes_candidate_arm_and_contract(monkeypatch, tmp_path):
    validator = _load_validator()
    captured = {}

    def fake_compile(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    class FakeProcess:
        pid = 4321

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["process_kwargs"] = kwargs
            result_path = Path(captured["args"][5])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "feasible": True,
                        "arm": "left",
                        "objects": ["object", "support"],
                        "attachment": {"to": "attached", "reason": "attach"},
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(validator, "compile_pick_place_probe_task", fake_compile)
    monkeypatch.setattr(validator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(validator, "_stop_process_group", lambda process: -15)
    candidate = {
        "candidate_id": "candidate_3",
        "world_xy": [1.0, 2.0],
        "yaw_deg": 45.0,
    }
    planning = {"collision_world": {"mode": "physics_schema"}}
    attach_paths = ["Aligned/collisions"]

    result = validator.run_pick_place_planning_probe(
        candidate,
        "left",
        "2",
        tmp_path / "source.yaml",
        "object",
        tmp_path / "run",
        30,
        "interndata",
        "conda",
        planning,
        attach_paths,
        seed=4,
    )

    assert captured["args"][1] is candidate
    assert captured["args"][2:4] == ("object", "left")
    assert captured["kwargs"] == {
        "planning": planning,
        "attach_prim_path_children": attach_paths,
    }
    env = captured["process_kwargs"]["env"]
    assert env["GPU_ID"] == "2"
    assert env["RANDOM_SEED"] == "4"
    assert env["INTERNDATA_SIMULATOR_BACKEND"] == "conda"
    assert captured["command"] == ["bash", "scripts/simbox/run_simbox_task.sh"]
    assert result["feasible"] is True
    assert result["results_complete"] is True
    assert result["terminated_after_result"] is True
    assert result["artifact"].endswith("candidate_3.left.json")
    assert result["result"]["attachment"]["to"] == "attached"
    assert result["result"]["seed"] == 4
    artifact = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
    assert artifact["seed"] == 4


def test_pick_probe_api_passes_seed_to_runtime_and_artifact(monkeypatch, tmp_path):
    validator = _load_validator()
    captured = {}

    def fake_compile(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    class FakeProcess:
        pid = 4322

        def __init__(self, command, **kwargs):
            captured["process_kwargs"] = kwargs
            result_dir = Path(captured["args"][4])
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "candidate_4.left.json").write_text(
                json.dumps(
                    {
                        "feasible": True,
                        "arm": "left",
                        "joint_success_count": 8,
                        "selected_grasp_score": 0.2,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(validator, "compile_probe_task", fake_compile)
    monkeypatch.setattr(validator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(validator, "_stop_process_group", lambda process: -15)
    planning = {"collision_world": {"mode": "physics_schema"}}

    result = validator._run_probe(
        {"candidate_id": "candidate_4", "world_xy": [1.0, 2.0], "yaw_deg": 0.0},
        "3",
        tmp_path / "source.yaml",
        "object",
        tmp_path / "run",
        30,
        "interndata",
        "docker",
        4,
        "left",
        planning,
        ["Aligned/collisions"],
        None,
        [],
        [],
        [],
        "full",
    )

    env = captured["process_kwargs"]["env"]
    assert env["RANDOM_SEED"] == "4"
    assert env["INTERNDATA_RANDOM_SEED"] == "4"
    assert env["INTERNDATA_GPU"] == "3"
    assert env["INTERNDATA_SIMULATOR_BACKEND"] == "docker"
    assert result["seed"] == 4
    pick_result = result["arms"]["left"]
    assert pick_result["seed"] == 4
    assert "/seed_4/results/" in pick_result["artifact"]
    artifact = json.loads(Path(pick_result["artifact"]).read_text(encoding="utf-8"))
    assert artifact["seed"] == 4


def test_validator_accepts_only_current_manifest_version(monkeypatch, tmp_path):
    validator = _load_validator()
    manifest_path = tmp_path / "candidates.json"
    args = SimpleNamespace(
        max_pick_candidates=3,
        manifest=manifest_path,
        arm="left",
        planning_gate="pick",
        candidate_id=[],
        seed=0,
    )
    monkeypatch.setattr(validator, "parse_args", lambda: args)
    manifest = {
        "version": 4,
        "source_task": str(tmp_path / "task.yaml"),
        "target": {"name": "object"},
        "required_arm": "left",
        "geometry_candidates": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator.main() == 2

    manifest["version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="expected workspace manifest version 4"):
        validator.main()


def test_probe_failure_summary_preserves_spawn_instability_cause():
    validator = _load_validator()
    probe_rows = [
        {
            "arms": {
                "left": {
                    "failure_code": "PROBE_SPAWN_UNSTABLE",
                    "spawn_check": {"stable": False},
                }
            }
        }
        for _ in range(8)
    ]

    status, failure_code, summary = validator._summarize_probe_failures(probe_rows)

    assert status == "spawn_unstable"
    assert failure_code == "PROBE_SPAWN_UNSTABLE"
    assert summary == {
        "probed_candidate_count": 8,
        "spawn_stable_count": 0,
        "spawn_unstable_count": 8,
        "failure_counts": {"PROBE_SPAWN_UNSTABLE": 8},
    }


def test_probe_failure_summary_uses_curobo_code_after_stable_spawn():
    validator = _load_validator()
    probe_rows = [
        {
            "arms": {
                "left": {
                    "failure_code": "NO_JOINT_GRASP_PLAN",
                    "spawn_check": {"stable": True},
                }
            }
        }
    ]

    status, failure_code, summary = validator._summarize_probe_failures(probe_rows)

    assert status == "no_safe_reachable_pose"
    assert failure_code == "NO_CUROBO_CANDIDATE"
    assert summary["spawn_stable_count"] == 1
