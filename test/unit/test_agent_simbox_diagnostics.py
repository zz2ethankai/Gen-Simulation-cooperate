"""Offline contract tests for the user-facing SimBox diagnostic tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.__main__ import build_parser
from agent.tools import simbox_diagnostics
from workflows.simbox.core.utils.camera_template import (
    resolve_camera_template_pose,
    robot_target_diagonal_pose,
)


def _settings() -> dict:
    return {
        "execution": {"gpu": 2, "conda_env": "interndata"},
        "generation": {"seed": 7},
        "debug": {
            "topdown_eye": [1.0, 1.5, 2.5],
            "topdown_target": [0.0, 0.0, 0.8],
            "topdown_resolution": [800, 600],
            "topdown_focal_length_mm": 20.0,
        },
    }


def _manifest(tmp_path: Path) -> Path:
    source_task = tmp_path / "source.yaml"
    source_task.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "objects": [
                            {
                                "name": "cup",
                                "attach_prim_path_children": ["Aligned/collisions"],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "source_candidates.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 4,
                "source_task": str(source_task),
                "required_arm": "left",
                "target": {"name": "cup"},
                "geometry_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_probe_dry_run_copies_manifest_and_expands_defaults(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    output_dir = tmp_path / "probe"
    args = build_parser(_settings()).parse_args(
        [
            "probe",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "candidate_004",
            "--simulator-backend",
            "conda",
            "--dry-run",
        ]
    )

    assert simbox_diagnostics.run_probe(args, _settings()) == 0

    summary = json.loads((output_dir / "probe_summary.json").read_text(encoding="utf-8"))
    command = summary["command"]
    assert (output_dir / "candidates.json").is_file()
    assert manifest.read_text(encoding="utf-8") != ""
    assert command[command.index("--arm") + 1] == "left"
    assert command[command.index("--gpus") + 1] == "2"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--simulator-backend") + 1] == "conda"
    assert command[command.index("--camera-eye") + 1 : command.index("--camera-eye") + 4] == [
        "1.0",
        "1.5",
        "2.5",
    ]
    assert "--capture-overview" in command
    assert "--capture-trajectory" in command
    assert "--attach-prim-path-child" in command
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_probe_camera_height_override_uses_relative_template(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    output_dir = tmp_path / "probe"
    args = build_parser(_settings()).parse_args(
        [
            "probe",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--camera-height-m",
            "1.2",
            "--dry-run",
        ]
    )

    assert simbox_diagnostics.run_probe(args, _settings()) == 0

    command = json.loads(capsys.readouterr().out)["command"]
    assert "--camera-eye" not in command
    assert command[command.index("--camera-template") + 1] == (
        "robot_target_overhead_v1"
    )
    params = json.loads(command[command.index("--camera-template-params-json") + 1])
    assert params["height_m"] == 1.2


def test_probe_requires_absolute_camera_points_as_a_pair(tmp_path):
    args = build_parser(_settings()).parse_args(
        [
            "probe",
            "--manifest",
            str(_manifest(tmp_path)),
            "--output-dir",
            str(tmp_path / "probe"),
            "--camera-eye",
            "0",
            "0",
            "2",
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="must be provided together"):
        simbox_diagnostics.run_probe(args, _settings())


def test_view_invokes_existing_physics_renderer_with_custom_camera(monkeypatch, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("tasks: []\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_dir = Path(command[command.index("--out-dir") + 1])
        image_dir = output_dir / "debug_overview"
        image_dir.mkdir(parents=True)
        (image_dir / "rgb_0000.png").write_bytes(b"png")
        (output_dir / "physics_audit.json").write_text(
            json.dumps({"physics_enabled": True}), encoding="utf-8"
        )
        (output_dir / "render_status.json").write_text(
            json.dumps({"return_code": 0, "error": None}), encoding="utf-8"
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(simbox_diagnostics.subprocess, "run", fake_run)
    args = build_parser(_settings()).parse_args(
        [
            "view",
            "--task",
            str(task),
            "--output-dir",
            str(tmp_path / "view"),
            "--eye",
            "0.5",
            "0.5",
            "2.0",
            "--target",
            "0",
            "0",
            "0.7",
            "--focal-length-mm",
            "24",
        ]
    )

    assert simbox_diagnostics.run_view(args, _settings()) == 0
    assert captured["command"][-2:] == ["--single-view", "debug_overview"]
    env = captured["kwargs"]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["INTERNDATA_ISAAC_ACTIVE_GPU"] == "2"
    assert env["TASK_RENDER_FOCAL_LENGTH_MM"] == "24.0"
    assert json.loads(env["TASK_RENDER_EXTRA_VIEWS_JSON"])[0] == {
        "name": "debug_overview",
        "eye": [0.5, 0.5, 2.0],
        "target": [0.0, 0.0, 0.7],
    }
    summary = json.loads(
        (tmp_path / "view" / "view_summary.json").read_text(encoding="utf-8")
    )
    assert summary["return_code"] == 0
    assert summary["subprocess_return_code"] == 0
    assert Path(summary["visualization_manifest"]).is_file()


def test_robot_target_camera_template_is_relative_but_returns_world_pose():
    pose = robot_target_diagonal_pose(
        [3.0, 2.0, 0.9],
        90.0,
        [3.0, 3.0, 1.0],
        {
            "behind_m": 0.75,
            "side_m": 0.85,
            "height_m": 1.2,
            "look_fraction": 0.65,
            "look_height_m": 0.2,
        },
    )

    assert pose["eye"] == [2.15, 1.25, 2.2]
    assert pose["target"] == [3.0, 2.65, 1.2]

    bounded = robot_target_diagonal_pose(
        [3.575, 0.5425, 0.901],
        -180.0,
        [3.51, 0.968, 0.883],
        room_bounds_xy=[0.0, 4.0, 0.0, 3.0],
    )
    assert 0.0 < bounded["eye"][0] < 4.0
    assert 0.0 < bounded["eye"][1] < 3.0

    overhead = resolve_camera_template_pose(
        "robot_target_overhead_v1",
        [3.575, 0.5425, 0.901],
        -180.0,
        [3.51, 0.968, 0.883],
    )
    assert overhead["eye"][:2] == overhead["target"][:2]
    assert overhead["eye"][2] > overhead["target"][2]
