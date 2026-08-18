"""Deterministic qualification over frozen held-out episode artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import Field

from ..contracts import (
    ContractModel,
    EpisodeIdentity,
    EvidenceBundle,
    dump_contract,
    load_contract,
)
from .artifacts import VariantArtifactManifest, verify_variant_artifact_manifest


HELDOUT_SEEDS = tuple(range(100, 120))
MIN_STRICT_SUCCESSES = 18
STRICT_CHECK_KEYS = (
    "terminal_success",
    "task_predicate_success",
    "safety_artifact_present",
    "safety_ok",
    "collision_audit_present",
    "physics_curobo_exact",
    "data_integrity_ok",
    "source_unchanged",
    "identity_consistent",
)


class HeldOutVariantArtifact(ContractModel):
    identity: EpisodeIdentity
    artifact_manifest_path: str


class QualificationSeedResult(ContractModel):
    seed: int
    identity: EpisodeIdentity
    variant_signature: str | None = None
    artifact_valid: bool
    strict_success: bool
    safety_ok: bool | None = None
    collision_exact: bool | None = None
    collision_audit_present: bool | None = None
    data_integrity_required: bool | None = None
    data_integrity_ok: bool | None = None
    source_unchanged: bool | None = None
    artifact_manifest_path: str
    evaluation_path: str | None = None
    evidence_path: str | None = None
    artifact_error: str | None = None


class QualificationFailure(ContractModel):
    code: str
    message: str
    seeds: list[int] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class QualificationSummary(ContractModel):
    qualified: bool
    expected_seeds: list[int]
    observed_seeds: list[int]
    artifact_count: int
    strict_success_count: int
    required_strict_success_count: int
    safety_violation_count: int
    collider_mismatch_count: int
    collider_audit_missing_count: int
    successful_data_integrity_failure_count: int
    source_integrity_failure_count: int
    accepted_variant_count: int
    unique_accepted_variant_count: int
    run_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    profile_hash: str | None = None
    source_hash: str | None = None
    failure_codes: list[str]
    failures: list[QualificationFailure]
    seed_results: list[QualificationSeedResult]


class _ArtifactError(ValueError):
    pass


def qualify_heldout_variants(
    artifacts: Sequence[HeldOutVariantArtifact],
    output_dir: Path,
) -> QualificationSummary:
    results = [_read_artifact(artifact) for artifact in artifacts]
    failures: list[QualificationFailure] = []
    observed_seeds = sorted(artifact.identity.seed for artifact in artifacts)

    if len(artifacts) != len(HELDOUT_SEEDS):
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_ARTIFACT_COUNT_INVALID",
                message=(
                    f"expected {len(HELDOUT_SEEDS)} held-out artifacts, "
                    f"received {len(artifacts)}"
                ),
            )
        )
    if observed_seeds != list(HELDOUT_SEEDS):
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_HELDOUT_SEEDS_INVALID",
                message=(
                    f"held-out seeds must be exactly {HELDOUT_SEEDS[0]}-"
                    f"{HELDOUT_SEEDS[-1]} with no duplicates"
                ),
                seeds=observed_seeds,
            )
        )

    cohort_keys = {
        (
            artifact.identity.profile_id,
            artifact.identity.profile_hash,
            artifact.identity.source_hash,
        )
        for artifact in artifacts
    }
    if len(cohort_keys) > 1:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_IDENTITY_COHORT_MISMATCH",
                message=(
                    "held-out artifacts must share one robot profile and source snapshot"
                ),
                seeds=observed_seeds,
                artifact_refs=[
                    artifact.artifact_manifest_path for artifact in artifacts
                ],
            )
        )

    invalid_results = [result for result in results if not result.artifact_valid]
    if invalid_results:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_ARTIFACT_INVALID",
                message="one or more attempt artifact manifests are invalid or inconsistent",
                seeds=sorted(result.seed for result in invalid_results),
                artifact_refs=_artifact_refs(invalid_results),
            )
        )

    strict_successes = [
        result
        for result in results
        if result.artifact_valid and result.strict_success
    ]
    if len(strict_successes) < MIN_STRICT_SUCCESSES:
        unsuccessful_results = [
            result
            for result in results
            if not result.artifact_valid or not result.strict_success
        ]
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_STRICT_SUCCESS_THRESHOLD_NOT_MET",
                message=(
                    f"strict success requires at least {MIN_STRICT_SUCCESSES}/"
                    f"{len(HELDOUT_SEEDS)}, observed {len(strict_successes)}"
                ),
                seeds=sorted(result.seed for result in unsuccessful_results),
                artifact_refs=_artifact_refs(unsuccessful_results),
            )
        )

    safety_violations = [
        result
        for result in results
        if result.artifact_valid and result.safety_ok is False
    ]
    if safety_violations:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_SAFETY_VIOLATION",
                message="qualification requires zero safety violations",
                seeds=sorted(result.seed for result in safety_violations),
                artifact_refs=_artifact_refs(safety_violations),
            )
        )

    missing_collision_audits = [
        result
        for result in results
        if result.artifact_valid and result.collision_audit_present is False
    ]
    if missing_collision_audits:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_COLLIDER_AUDIT_MISSING",
                message="qualification requires a Physics/CuRobo audit for every seed",
                seeds=sorted(result.seed for result in missing_collision_audits),
                artifact_refs=_artifact_refs(missing_collision_audits),
            )
        )

    collider_mismatches = [
        result
        for result in results
        if result.artifact_valid
        and result.collision_audit_present is True
        and result.collision_exact is False
    ]
    if collider_mismatches:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_COLLIDER_MISMATCH",
                message="qualification requires zero Physics/CuRobo collider mismatches",
                seeds=sorted(result.seed for result in collider_mismatches),
                artifact_refs=_artifact_refs(collider_mismatches),
            )
        )

    incomplete_successes = [
        result
        for result in strict_successes
        if result.data_integrity_required is not True
        or result.data_integrity_ok is not True
    ]
    if incomplete_successes:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_SUCCESS_DATA_INTEGRITY_FAILED",
                message="every strict-success episode must contain required, complete data",
                seeds=sorted(result.seed for result in incomplete_successes),
                artifact_refs=_artifact_refs(incomplete_successes),
            )
        )

    changed_sources = [
        result
        for result in results
        if result.artifact_valid and result.source_unchanged is False
    ]
    if changed_sources:
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_SOURCE_INTEGRITY_FAILED",
                message="qualification requires source task and arena hashes to remain unchanged",
                seeds=sorted(result.seed for result in changed_sources),
                artifact_refs=_artifact_refs(changed_sources),
            )
        )

    signature_counts = Counter(
        result.variant_signature
        for result in strict_successes
        if result.variant_signature is not None
    )
    duplicate_signatures = sorted(
        signature for signature, count in signature_counts.items() if count > 1
    )
    if duplicate_signatures:
        duplicate_results = [
            result
            for result in strict_successes
            if result.variant_signature in duplicate_signatures
        ]
        failures.append(
            QualificationFailure(
                code="QUALIFICATION_ACCEPTED_VARIANT_DUPLICATE",
                message=(
                    "accepted variant_signature values must be unique: "
                    + ", ".join(duplicate_signatures)
                ),
                seeds=sorted(result.seed for result in duplicate_results),
                artifact_refs=_artifact_refs(duplicate_results),
            )
        )

    summary = QualificationSummary(
        qualified=not failures,
        expected_seeds=list(HELDOUT_SEEDS),
        observed_seeds=observed_seeds,
        artifact_count=len(artifacts),
        strict_success_count=len(strict_successes),
        required_strict_success_count=MIN_STRICT_SUCCESSES,
        safety_violation_count=len(safety_violations),
        collider_mismatch_count=len(collider_mismatches),
        collider_audit_missing_count=len(missing_collision_audits),
        successful_data_integrity_failure_count=len(incomplete_successes),
        source_integrity_failure_count=len(changed_sources),
        accepted_variant_count=len(strict_successes),
        unique_accepted_variant_count=len(signature_counts),
        run_ids=sorted({artifact.identity.run_id for artifact in artifacts}),
        profile_id=(
            artifacts[0].identity.profile_id if len(cohort_keys) == 1 and artifacts else None
        ),
        profile_hash=(
            artifacts[0].identity.profile_hash if len(cohort_keys) == 1 and artifacts else None
        ),
        source_hash=(
            artifacts[0].identity.source_hash if len(cohort_keys) == 1 and artifacts else None
        ),
        failure_codes=[failure.code for failure in failures],
        failures=failures,
        seed_results=sorted(results, key=lambda result: result.seed),
    )
    dump_contract(summary, output_dir / "qualification_summary.json")
    return summary


def _read_artifact(artifact: HeldOutVariantArtifact) -> QualificationSeedResult:
    manifest_path = Path(artifact.artifact_manifest_path).expanduser().resolve()
    evaluation_path: Path | None = None
    evidence_path: Path | None = None
    try:
        manifest = verify_variant_artifact_manifest(manifest_path)
        if manifest.identity != artifact.identity:
            raise _ArtifactError(
                "qualification identity disagrees with artifact manifest identity"
            )
        evaluation_path = _artifact_file(manifest, "evaluation")
        evidence_path = _artifact_file(manifest, "evidence")
        evaluation = _read_json_object(evaluation_path)
        evidence = load_contract(EvidenceBundle, evidence_path)
        evaluation_identity_value = evaluation.get("identity")
        if not isinstance(evaluation_identity_value, dict):
            raise _ArtifactError("evaluation.identity must be an object")
        evaluation_identity = EpisodeIdentity.from_dict(evaluation_identity_value)
        if evaluation_identity != artifact.identity:
            raise _ArtifactError("qualification identity disagrees with evaluation identity")
        if evidence.identity != artifact.identity:
            raise _ArtifactError("qualification identity disagrees with EvidenceBundle identity")
        if evidence.identity_errors:
            raise _ArtifactError("EvidenceBundle contains identity errors")
        if evidence.attempt_id != f"task:{manifest.attempt_id}":
            raise _ArtifactError(
                "EvidenceBundle attempt_id disagrees with artifact manifest attempt_id"
            )
        variant_signature = evaluation.get("variant_signature")
        if (
            not isinstance(variant_signature, str)
            or len(variant_signature) != 64
            or any(character not in "0123456789abcdef" for character in variant_signature)
        ):
            raise _ArtifactError("evaluation.variant_signature must be a SHA-256 digest")
        if evidence.variant_signature != variant_signature:
            raise _ArtifactError(
                "evaluation variant_signature disagrees with EvidenceBundle.variant_signature"
            )
        if manifest.variant_signature != variant_signature:
            raise _ArtifactError(
                "evaluation variant_signature disagrees with artifact manifest"
            )
        checks = _mapping(evaluation, "checks")
        data_integrity = _mapping(evaluation, "data_integrity")
        strict_success = _boolean(evaluation, "strict_success")
        safety_ok = _boolean(checks, "safety_ok")
        safety_artifact_present = _boolean(checks, "safety_artifact_present")
        collision_exact = _boolean(checks, "physics_curobo_exact")
        collision_audit_present = _boolean(checks, "collision_audit_present")
        data_integrity_ok = _boolean(data_integrity, "ok")
        data_integrity_required = _boolean(data_integrity, "required")
        source_unchanged = _boolean(checks, "source_unchanged")
        if manifest.data_required != data_integrity_required:
            raise _ArtifactError(
                "evaluation data requirement disagrees with artifact manifest"
            )
        if _boolean(checks, "data_integrity_ok") != data_integrity_ok:
            raise _ArtifactError("evaluation data_integrity fields disagree")
        if evidence.task_success != strict_success:
            raise _ArtifactError("evaluation strict_success disagrees with EvidenceBundle.task_success")
        if safety_ok != (not evidence.safety_events):
            raise _ArtifactError("evaluation safety_ok disagrees with EvidenceBundle.safety_events")
        if strict_success and not safety_artifact_present:
            raise _ArtifactError("strict success requires a safety artifact")
        if collision_audit_present != (evidence.collision_world_ok is not None):
            raise _ArtifactError(
                "evaluation collision_audit_present disagrees with EvidenceBundle.collision_world_ok"
            )
        if collision_exact != (evidence.collision_world_ok is True):
            raise _ArtifactError(
                "evaluation physics_curobo_exact disagrees with EvidenceBundle.collision_world_ok"
            )
        strict_checks = [_boolean(checks, key) for key in STRICT_CHECK_KEYS]
        if strict_success != all(strict_checks):
            raise _ArtifactError("evaluation strict_success disagrees with its strict checks")
        return QualificationSeedResult(
            seed=artifact.identity.seed,
            identity=artifact.identity,
            variant_signature=variant_signature,
            artifact_valid=True,
            strict_success=strict_success,
            safety_ok=safety_ok,
            collision_exact=collision_exact,
            collision_audit_present=collision_audit_present,
            data_integrity_required=data_integrity_required,
            data_integrity_ok=data_integrity_ok,
            source_unchanged=source_unchanged,
            artifact_manifest_path=str(manifest_path),
            evaluation_path=str(evaluation_path),
            evidence_path=str(evidence_path),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return QualificationSeedResult(
            seed=artifact.identity.seed,
            identity=artifact.identity,
            artifact_valid=False,
            strict_success=False,
            artifact_manifest_path=str(manifest_path),
            evaluation_path=str(evaluation_path) if evaluation_path is not None else None,
            evidence_path=str(evidence_path) if evidence_path is not None else None,
            artifact_error=str(exc),
        )


def _artifact_file(manifest: VariantArtifactManifest, name: str) -> Path:
    entry = manifest.artifacts[name]
    if entry.kind != "file" or len(entry.members) != 1:
        raise _ArtifactError(f"artifact manifest {name} entry must contain one file")
    return Path(entry.members[0].path)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise _ArtifactError(f"evaluation root is not an object: {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise _ArtifactError(f"evaluation.{key} must be an object")
    return item


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise _ArtifactError(f"{key} must be a boolean")
    return item


def _artifact_refs(results: Sequence[QualificationSeedResult]) -> list[str]:
    return sorted(
        {
            ref
            for result in results
            for ref in (
                result.artifact_manifest_path,
                result.evaluation_path,
                result.evidence_path,
            )
            if ref is not None
        }
    )
