"""Qualification gates over frozen held-out evaluation and evidence artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from agent.contracts import EpisodeIdentity, EvidenceBundle, dump_contract
from agent.tools.artifacts import (
    VariantArtifactManifest,
    write_variant_artifact_manifest,
)
from agent.tools.qualification import (
    HELDOUT_SEEDS,
    HeldOutVariantArtifact,
    qualify_heldout_variants,
)
from agent.tools.signatures import variant_signature


def _artifact(
    root: Path,
    seed: int,
    *,
    strict_success: bool = True,
    safety_ok: bool = True,
    collision_exact: bool | None = True,
    data_integrity_required: bool = True,
    data_integrity_ok: bool = True,
    compiled_identity_seed: int | None = None,
) -> HeldOutVariantArtifact:
    variant_root = root / str(seed) / "variants" / f"variant_{seed}"
    attempt_dir = variant_root / "attempts" / "00"
    episode_dir = attempt_dir / "data" / "episode_000000"
    workspace_dir = variant_root / "subtasks" / "transfer" / "workspace"
    workspace_selection_path = (
        variant_root / "workspace_selection" / "position_selection.json"
    )
    evaluation_path = attempt_dir / "evaluation.json"
    evidence_path = attempt_dir / "evidence.json"
    attempt_dir.mkdir(parents=True)
    collision_audit_present = collision_exact is not None
    compiled_seed = seed if compiled_identity_seed is None else compiled_identity_seed
    candidate_id = f"candidate_{compiled_seed}"
    scene_revision = "source"
    compiled_document = {
        "tasks": [
            {
                "metadata": {
                    "agent_plan": {
                        "workspace_candidate_id": candidate_id,
                        "scene_revision": scene_revision,
                    },
                    "robot_position_plan": {
                        "initial": {"world_xyz": [compiled_seed, 0, 0]}
                    },
                }
            }
        ]
    }
    _write(attempt_dir / "task.yaml", json.dumps(compiled_document))
    identity = EpisodeIdentity(
        run_id="qualification_run",
        variant_id=f"variant_{seed}",
        seed=seed,
        profile_id="profile_a",
        profile_hash="a" * 64,
        source_hash="b" * 64,
        scene_revision=scene_revision,
        world_revision=seed,
    )
    signature = variant_signature(compiled_document)
    checks = {
        "terminal_success": strict_success,
        "task_predicate_success": strict_success,
        "safety_artifact_present": True,
        "safety_ok": safety_ok,
        "collision_audit_present": collision_audit_present,
        "physics_curobo_exact": collision_exact is True,
        "data_integrity_ok": data_integrity_ok,
        "source_unchanged": True,
        "identity_consistent": True,
    }
    evaluation_path.write_text(
        json.dumps(
            {
                "strict_success": strict_success,
                "failure_code": None if strict_success else "EPISODE_FAILED",
                "checks": checks,
                "data_integrity": {
                    "required": data_integrity_required,
                    "ok": data_integrity_ok,
                    "reason": "",
                },
                "predicate_results": [],
                "identity": identity.to_dict(),
                "identity_errors": [],
                "variant_signature": signature,
            }
        ),
        encoding="utf-8",
    )
    dump_contract(
        EvidenceBundle(
            attempt_id="task:00",
            identity=identity,
            variant_signature=signature,
            status="success" if strict_success else "failed",
            task_success=strict_success,
            collision_world_ok=collision_exact,
            safety_events=[] if safety_ok else [{"trigger": "SAFETY_VIOLATION"}],
            episode_dir=str(episode_dir),
            artifact_refs=[str(evaluation_path)],
        ),
        evidence_path,
    )
    probe_payload = json.dumps(
        {
            "feasible": True,
            "candidate_id": candidate_id,
            "arm": "left",
            "seed": seed,
        }
    )
    pick_probe = _write(
        workspace_dir / "probes" / candidate_id / "result.json", probe_payload
    )
    place_probe = _write(
        workspace_dir / "place_probes" / candidate_id / "result.json", probe_payload
    )
    spawn_settle = _write(
        workspace_dir / "spawn_settle.json",
        json.dumps(
            {
                "stable": True,
                "candidate_id": candidate_id,
                "arm": "left",
                "seed": seed,
            }
        ),
    )
    _write(
        workspace_dir / "candidates.json",
        json.dumps(
            {
                "status": "planning_success",
                "selected_candidate": {
                    "candidate_id": candidate_id,
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
        workspace_selection_path,
        json.dumps(
            {
                "mode": "single",
                "seed": seed,
                "candidate": {"candidate_id": candidate_id, "arm": "left"},
                "subtasks": [
                    {
                        "subtask_id": "transfer",
                        "target": "cup",
                        "arm": "left",
                        "workspace_manifest_path": str(
                            workspace_dir / "candidates.json"
                        ),
                    }
                ],
            }
        ),
    )
    _write(
        variant_root / "static_validation.json",
        json.dumps({"hard_ok": True, "scene_revision": scene_revision}),
    )
    _write(attempt_dir / "trace.jsonl", '{"stage": "episode_evaluation"}\n')
    _write(attempt_dir / "stdout.log", "episode complete\n")
    _write(episode_dir / "collision_world_audit.json", '{"exact": true}')
    _write(episode_dir / "states.lmdb", "lmdb")
    if data_integrity_required:
        _write(episode_dir / "images.rgb.global" / "demo.mp4", "mp4")
    manifest = write_variant_artifact_manifest(
        variant_root,
        attempt_dir,
        workspace_selection_path,
        data_required=data_integrity_required,
    )
    assert manifest.complete is True
    return HeldOutVariantArtifact(
        identity=identity,
        artifact_manifest_path=str(attempt_dir / "artifact_manifest.json"),
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _manifest(artifact: HeldOutVariantArtifact) -> VariantArtifactManifest:
    return VariantArtifactManifest.from_dict(
        json.loads(Path(artifact.artifact_manifest_path).read_text(encoding="utf-8"))
    )


def _artifact_path(artifact: HeldOutVariantArtifact, name: str) -> Path:
    entry = _manifest(artifact).artifacts[name]
    assert entry.path is not None
    return Path(entry.path)


def _reindex(artifact: HeldOutVariantArtifact) -> None:
    manifest = _manifest(artifact)
    write_variant_artifact_manifest(
        Path(manifest.variant_root),
        Path(manifest.attempt_dir),
        Path(manifest.workspace_selection_path),
        data_required=manifest.data_required,
    )


def _heldout(root: Path) -> list[HeldOutVariantArtifact]:
    return [_artifact(root, seed) for seed in HELDOUT_SEEDS]


def test_qualification_accepts_twenty_distinct_strict_heldout_successes(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is True
    assert summary.strict_success_count == 20
    assert summary.failure_codes == []
    output = json.loads(
        (tmp_path / "result" / "qualification_summary.json").read_text(encoding="utf-8")
    )
    assert output == summary.to_dict()


def test_qualification_allows_exactly_eighteen_strict_successes(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    for seed in (118, 119):
        artifacts[seed - 100] = _artifact(
            tmp_path / "replacement",
            seed,
            strict_success=False,
            data_integrity_ok=False,
        )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is True
    assert summary.strict_success_count == 18
    assert summary.successful_data_integrity_failure_count == 0


def test_qualification_rejects_incomplete_seed_set(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")[:-3]

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.failure_codes == [
        "QUALIFICATION_ARTIFACT_COUNT_INVALID",
        "QUALIFICATION_HELDOUT_SEEDS_INVALID",
        "QUALIFICATION_STRICT_SUCCESS_THRESHOLD_NOT_MET",
    ]


def test_qualification_rejects_seventeen_strict_successes(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    for seed in (117, 118, 119):
        artifacts[seed - 100] = _artifact(
            tmp_path / "replacement",
            seed,
            strict_success=False,
            data_integrity_ok=False,
        )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.strict_success_count == 17
    assert summary.failure_codes == [
        "QUALIFICATION_STRICT_SUCCESS_THRESHOLD_NOT_MET"
    ]


def test_qualification_rejects_safety_violation_even_with_nineteen_successes(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    artifacts[0] = _artifact(
        tmp_path / "replacement",
        100,
        strict_success=False,
        safety_ok=False,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.safety_violation_count == 1
    assert summary.failure_codes == ["QUALIFICATION_SAFETY_VIOLATION"]


def test_qualification_distinguishes_collider_mismatch_from_missing_audit(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    artifacts[0] = _artifact(
        tmp_path / "replacement_mismatch",
        100,
        strict_success=False,
        collision_exact=False,
    )
    artifacts[1] = _artifact(
        tmp_path / "replacement_missing",
        101,
        strict_success=False,
        collision_exact=None,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.collider_mismatch_count == 1
    assert summary.collider_audit_missing_count == 1
    assert summary.failure_codes == [
        "QUALIFICATION_COLLIDER_AUDIT_MISSING",
        "QUALIFICATION_COLLIDER_MISMATCH",
    ]


def test_qualification_requires_data_integrity_to_be_enabled_for_successes(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    artifacts[0] = _artifact(
        tmp_path / "replacement",
        100,
        data_integrity_required=False,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.successful_data_integrity_failure_count == 1
    assert summary.failure_codes == [
        "QUALIFICATION_SUCCESS_DATA_INTEGRITY_FAILED"
    ]


def test_qualification_rejects_duplicate_accepted_variant_signatures(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    artifacts[1] = _artifact(
        tmp_path / "replacement",
        101,
        compiled_identity_seed=100,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.accepted_variant_count == 20
    assert summary.unique_accepted_variant_count == 19
    assert summary.failure_codes == [
        "QUALIFICATION_ACCEPTED_VARIANT_DUPLICATE"
    ]


def test_qualification_rejects_evaluation_evidence_disagreement(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    evidence_path = _artifact_path(artifacts[0], "evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["task_success"] = False
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    _reindex(artifacts[0])

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.seed_results[0].artifact_valid is False
    assert summary.failure_codes == ["QUALIFICATION_ARTIFACT_INVALID"]


def test_qualification_rejects_raw_member_changed_after_manifest(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    compiled_task = _artifact_path(artifacts[0], "compiled_task")
    compiled_task.write_text("tasks:\n  - tampered: true\n", encoding="utf-8")

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.seed_results[0].artifact_valid is False
    assert "size or SHA-256 changed for compiled_task" in str(
        summary.seed_results[0].artifact_error
    )
    assert summary.failure_codes == ["QUALIFICATION_ARTIFACT_INVALID"]


def test_qualification_rejects_caller_identity_that_disagrees_with_artifacts(
    tmp_path: Path,
):
    artifacts = _heldout(tmp_path / "artifacts")
    artifacts[0] = HeldOutVariantArtifact(
        identity=EpisodeIdentity(
            **{
                **artifacts[0].identity.to_dict(),
                "variant_id": "caller_forged_variant",
            }
        ),
        artifact_manifest_path=artifacts[0].artifact_manifest_path,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert summary.seed_results[0].artifact_valid is False
    assert "QUALIFICATION_ARTIFACT_INVALID" in summary.failure_codes


def test_qualification_rejects_mixed_profile_or_source_cohort(tmp_path: Path):
    artifacts = _heldout(tmp_path / "artifacts")
    replacement = _artifact(tmp_path / "replacement", 100)
    forged_identity = EpisodeIdentity(
        **{
            **replacement.identity.to_dict(),
            "profile_hash": "c" * 64,
        }
    )
    evaluation_path = _artifact_path(replacement, "evaluation")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["identity"] = forged_identity.to_dict()
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    evidence_path = _artifact_path(replacement, "evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["identity"] = forged_identity.to_dict()
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    _reindex(replacement)
    artifacts[0] = HeldOutVariantArtifact(
        identity=forged_identity,
        artifact_manifest_path=replacement.artifact_manifest_path,
    )

    summary = qualify_heldout_variants(artifacts, tmp_path / "result")

    assert summary.qualified is False
    assert "QUALIFICATION_IDENTITY_COHORT_MISMATCH" in summary.failure_codes
