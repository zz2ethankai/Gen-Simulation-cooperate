"""Artifact indexing and held-out qualification CLI contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from agent.__main__ import main
from agent.tools.artifacts import (
    ARTIFACT_ORDER,
    verify_variant_artifact_manifest,
    write_variant_artifact_manifest,
)
from agent.tools.signatures import variant_signature
from agent.tools.trace import TraceContext, TraceEvent, TraceWriter


def _write(path: Path, content: str = "artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _complete_attempt(root: Path) -> tuple[Path, Path, Path]:
    variant_root = root / "variants" / "split_aloha__left"
    attempt_dir = variant_root / "attempts" / "00"
    workspace_selection = (
        variant_root / "workspace_selection" / "position_selection.json"
    )
    episode_dir = attempt_dir / "data" / "episode_000000"
    pick_probe = _write(
        variant_root
        / "subtasks"
        / "transfer"
        / "workspace"
        / "probes"
        / "candidate_000"
        / "results"
        / "candidate_000.left.json",
        '{"feasible": true, "candidate_id": "candidate_000", "arm": "left", "seed": 0}\n',
    )
    place_probe = _write(
        variant_root
        / "subtasks"
        / "transfer"
        / "workspace"
        / "place_probes"
        / "candidate_000"
        / "results"
        / "candidate_000.left.json",
        '{"feasible": true, "candidate_id": "candidate_000", "arm": "left", "seed": 0}\n',
    )
    spawn_settle = _write(
        variant_root
        / "subtasks"
        / "transfer"
        / "workspace"
        / "spawn_settle.json",
        '{"stable": true, "candidate_id": "candidate_000", "arm": "left", "seed": 0}\n',
    )
    workspace_manifest = _write(
        variant_root / "subtasks" / "transfer" / "workspace" / "candidates.json",
        json.dumps(
            {
                "status": "planning_success",
                "selected_candidate": {
                    "candidate_id": "candidate_000",
                    "arm": "left",
                },
                "planning_probe_artifacts": {
                    "pick": str(pick_probe),
                    "place": str(place_probe),
                    "spawn_settle": str(spawn_settle),
                },
            }
        ),
    )
    _write(
        workspace_selection,
        json.dumps(
            {
                "mode": "single",
                "seed": 0,
                "candidate": {"candidate_id": "candidate_000", "arm": "left"},
                "subtasks": [
                    {
                        "subtask_id": "transfer",
                        "target": "cup",
                        "arm": "left",
                        "workspace_manifest_path": str(workspace_manifest),
                    }
                ],
            }
        ),
    )
    _write(
        variant_root / "scene_layout" / "candidate_000" / "static_validation.json",
        '{"hard_ok": true, "scene_revision": "source"}\n',
    )
    _write(attempt_dir / "trace.jsonl", '{"stage": "episode_evaluation"}\n')
    compiled_document = {
        "tasks": [
            {
                "metadata": {
                    "agent_plan": {
                        "workspace_candidate_id": "candidate_000",
                        "scene_revision": "source",
                    }
                }
            }
        ]
    }
    _write(attempt_dir / "task.yaml", json.dumps(compiled_document))
    _write(attempt_dir / "stdout.log", "episode complete\n")
    _write(
        episode_dir / "collision_world_audit.json",
        '{"physics_curobo_difference": {}}\n',
    )
    _write(episode_dir / "screenshots" / "after_settle.png", "png")
    _write(episode_dir / "images.rgb.global" / "demo.mp4", "mp4")
    _write(episode_dir / "states.lmdb", "lmdb")
    _write(
        attempt_dir / "evaluation.json",
        json.dumps({"data_integrity": {"required": True, "ok": True}}),
    )
    _write(
        attempt_dir / "evidence.json",
        json.dumps(
            {
                "episode_dir": str(episode_dir),
                "identity": {
                    "run_id": "run_1",
                    "variant_id": "split_aloha__left",
                    "seed": 0,
                    "profile_id": "profile_a",
                    "profile_hash": "a" * 64,
                    "source_hash": "b" * 64,
                    "scene_revision": "source",
                    "world_revision": 1,
                },
                "variant_signature": variant_signature(compiled_document),
            }
        ),
    )
    return variant_root, attempt_dir, workspace_selection


def test_artifact_manifest_indexes_only_existing_content_with_sha256(tmp_path: Path):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)

    manifest = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    assert manifest.complete is True
    assert manifest.failure_codes == []
    assert list(manifest.artifacts) == list(ARTIFACT_ORDER)
    assert manifest.data_required is True
    assert manifest.artifacts["screenshots"].required is False
    assert manifest.artifacts["scene_layout"].required is False
    assert manifest.artifacts["scene_layout"].present is False
    assert manifest.artifacts["videos"].required is True
    assert manifest.artifacts["data"].required is True
    for name, artifact in manifest.artifacts.items():
        if name == "scene_layout":
            continue
        assert artifact.present is True
        assert artifact.path is not None
        assert artifact.sha256 is not None
    task_path = attempt_dir / "task.yaml"
    assert (
        manifest.artifacts["compiled_task"].sha256
        == hashlib.sha256(task_path.read_bytes()).hexdigest()
    )
    persisted = json.loads(
        (attempt_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest.to_dict()


def test_next_attempt_trace_does_not_invalidate_completed_manifest(tmp_path: Path):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    manifest = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    TraceWriter(variant_root / "attempts" / "01" / "trace.jsonl").append(
        TraceEvent(
            TraceContext(
                run_id="run_1",
                variant_id=variant_root.name,
                attempt_id="01",
            ),
            stage="episode_evaluation",
            status="failed",
        )
    )

    verified = verify_variant_artifact_manifest(
        attempt_dir / "artifact_manifest.json"
    )
    assert verified == manifest
    assert verified.artifacts["trace"].path == str(
        (attempt_dir / "trace.jsonl").resolve()
    )


def test_artifact_manifest_reports_required_gaps_without_placeholders(tmp_path: Path):
    variant_root = tmp_path / "variants" / "fr3__left"
    attempt_dir = variant_root / "attempts" / "00"
    workspace_selection = (
        variant_root / "workspace_selection" / "position_selection.json"
    )
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "screenshots").mkdir()

    manifest = write_variant_artifact_manifest(
        variant_root,
        attempt_dir,
        workspace_selection,
        data_required=False,
    )

    assert manifest.complete is False
    assert "ARTIFACT_COMPILED_TASK_MISSING" in manifest.failure_codes
    assert "ARTIFACT_SPAWN_SETTLE_MISSING" in manifest.failure_codes
    assert "ARTIFACT_PLACE_PROBE_MISSING" in manifest.failure_codes
    assert manifest.artifacts["screenshots"].failure_code is None
    assert manifest.artifacts["screenshots"].present is False
    assert manifest.artifacts["videos"].failure_code is None
    assert manifest.artifacts["data"].failure_code is None
    assert not (attempt_dir / "task.yaml").exists()
    assert not (attempt_dir / "spawn_settle.json").exists()
    assert not workspace_selection.exists()
    assert (attempt_dir / "artifact_manifest.json").is_file()


def test_artifact_manifest_indexes_attempt_owned_topdown_screenshot(tmp_path: Path):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    topdown = _write(attempt_dir / "screenshots" / "topdown.png", "topdown")

    manifest = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    screenshots = manifest.artifacts["screenshots"]
    assert screenshots.present is True
    assert any(
        member.path == str(topdown.parent.resolve())
        for member in screenshots.members
    )


def test_artifact_manifest_rejects_probes_from_an_unselected_candidate(tmp_path: Path):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    manifest_path = (
        variant_root / "subtasks" / "transfer" / "workspace" / "candidates.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["planning_probe_artifacts"].values():
        value = json.loads(Path(artifact).read_text(encoding="utf-8"))
        value["candidate_id"] = "candidate_not_selected"
        Path(artifact).write_text(json.dumps(value), encoding="utf-8")

    result = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    assert result.complete is False
    assert "ARTIFACT_PICK_PROBE_MISSING" in result.failure_codes
    assert "ARTIFACT_PLACE_PROBE_MISSING" in result.failure_codes
    assert "ARTIFACT_SPAWN_SETTLE_MISSING" in result.failure_codes


def test_artifact_manifest_binds_compiled_and_selected_workspace_candidate(
    tmp_path: Path,
):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    compiled = json.loads((attempt_dir / "task.yaml").read_text(encoding="utf-8"))
    compiled["tasks"][0]["metadata"]["agent_plan"][
        "workspace_candidate_id"
    ] = "candidate_not_selected"
    (attempt_dir / "task.yaml").write_text(json.dumps(compiled), encoding="utf-8")

    result = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    assert result.complete is False
    assert "ARTIFACT_WORKSPACE_IDENTITY_MISMATCH" in result.failure_codes


def test_artifact_manifest_only_indexes_matching_scene_revision_validation(
    tmp_path: Path,
):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    validation_path = (
        variant_root / "scene_layout" / "candidate_000" / "static_validation.json"
    )
    validation_path.write_text(
        '{"hard_ok": true, "scene_revision": "stale_revision"}\n',
        encoding="utf-8",
    )

    result = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )

    assert result.complete is False
    assert "ARTIFACT_STATIC_VALIDATION_MISSING" in result.failure_codes


def test_artifact_manifest_binds_derived_task_arena_and_mutation_revision(
    tmp_path: Path,
):
    variant_root, attempt_dir, workspace_selection = _complete_attempt(tmp_path)
    revision_payload = {
        "source_task_hash": "c" * 64,
        "source_arena_hash": "d" * 64,
        "mutations": [
            {
                "kind": "move_entity_on_support",
                "entity": "cup",
                "world_xy_m": [0.1, 0.2],
            }
        ],
    }
    scene_revision = hashlib.sha256(
        json.dumps(revision_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    revision_dir = variant_root / "scene_layout" / "derived" / "scene"
    derived_arena = revision_dir / "simbox_arena.yaml"
    derived_task = revision_dir / "simbox_task.yaml"
    mutation_path = revision_dir / "scene_mutations.json"
    layout = {**revision_payload, "scene_revision": scene_revision}
    derived_document = {
        "tasks": [
            {
                "arena_file": str(derived_arena),
                "metadata": {"agent_scene_layout": layout},
            }
        ]
    }
    _write(derived_arena, "fixtures: []\n")
    _write(derived_task, json.dumps(derived_document))
    _write(mutation_path, json.dumps(layout))

    compiled = json.loads((attempt_dir / "task.yaml").read_text(encoding="utf-8"))
    compiled_task = compiled["tasks"][0]
    compiled_task["arena_file"] = str(derived_arena)
    compiled_task["metadata"]["agent_scene_layout"] = layout
    compiled_task["metadata"]["agent_plan"]["scene_revision"] = scene_revision
    (attempt_dir / "task.yaml").write_text(json.dumps(compiled), encoding="utf-8")
    signature = variant_signature(compiled)
    evidence_path = attempt_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["identity"]["scene_revision"] = scene_revision
    evidence["variant_signature"] = signature
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    validation_path = (
        variant_root / "scene_layout" / "candidate_000" / "static_validation.json"
    )
    validation_path.write_text(
        json.dumps({"hard_ok": True, "scene_revision": scene_revision}),
        encoding="utf-8",
    )

    manifest = write_variant_artifact_manifest(
        variant_root, attempt_dir, workspace_selection
    )
    verified = verify_variant_artifact_manifest(
        attempt_dir / "artifact_manifest.json"
    )

    assert manifest.complete is True
    assert manifest.artifacts["scene_layout"].required is True
    assert {Path(member.path).name for member in manifest.artifacts["scene_layout"].members} == {
        "simbox_task.yaml",
        "simbox_arena.yaml",
        "scene_mutations.json",
    }
    assert verified == manifest


def test_qualify_cli_reads_frozen_artifact_list_without_updating_admission(
    tmp_path: Path, monkeypatch, capsys
):
    artifact_list = _write(
        tmp_path / "heldout.json",
        json.dumps(
            [
                {
                    "identity": {
                        "run_id": "run_1",
                        "variant_id": "variant_100",
                        "seed": 100,
                        "profile_id": "profile_a",
                        "profile_hash": "a" * 64,
                        "source_hash": "b" * 64,
                        "scene_revision": "scene_100",
                        "world_revision": 100,
                    },
                    "artifact_manifest_path": str(
                        tmp_path / "missing_artifact_manifest.json"
                    ),
                }
            ]
        ),
    )
    output_dir = tmp_path / "qualification"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m agent",
            "qualify",
            "--artifacts",
            str(artifact_list),
            "--output-dir",
            str(output_dir),
        ],
    )

    return_code = main()

    assert return_code == 2
    summary = json.loads(
        (output_dir / "qualification_summary.json").read_text(encoding="utf-8")
    )
    assert summary["qualified"] is False
    assert summary["artifact_count"] == 1
    assert summary["observed_seeds"] == [100]
    printed = json.loads(capsys.readouterr().out)
    assert printed == summary
