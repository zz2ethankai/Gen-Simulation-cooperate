"""Normalize SimBox episode artifacts into deterministic failure evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .contracts import Diagnosis, EvidenceBundle


AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent
CONTAINER_ROOT = Path("/workspace")


def _host_artifact_path(value: str) -> Path:
    path = Path(value)
    try:
        relative = path.relative_to(CONTAINER_ROOT)
    except ValueError:
        return path
    return REPO_ROOT / relative


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _last_event(path: Path) -> dict[str, Any] | None:
    values = _jsonl(path)
    return values[-1] if values else None


def _failure_codes() -> tuple[dict[str, str], list[str]]:
    payload = yaml.safe_load((AGENT_DIR / "registry" / "failure_codes.yaml").read_text(encoding="utf-8"))
    by_code: dict[str, str] = {}
    for category, codes in payload.get("categories", {}).items():
        for code in codes:
            by_code[str(code)] = str(category)
    return by_code, sorted(by_code, key=len, reverse=True)


def collect_evidence(
    attempt_id: str,
    attempt_dir: Path,
    event_path: Path,
    log_path: Path,
    return_code: int | None,
    timed_out: bool,
) -> EvidenceBundle:
    event = _last_event(event_path)
    stdout = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    episode_value = str(event.get("primary_episode_dir", "")) if event else ""
    episode_dir = _host_artifact_path(episode_value) if episode_value else None
    if episode_dir is not None and not episode_dir.exists():
        episode_dir = None

    artifact_refs = [str(path) for path in (event_path, log_path) if path.is_file()]
    object_events: list[dict[str, Any]] = []
    safety_events: list[dict[str, Any]] = []
    collision_world_ok: bool | None = None
    if episode_dir is not None:
        audit_path = episode_dir / "collision_world_audit.json"
        object_path = episode_dir / "object_state_events.jsonl"
        safety_path = episode_dir / "safety_events.jsonl"
        audit = _json(audit_path)
        object_events = _jsonl(object_path)
        safety_events = _jsonl(safety_path)
        if audit is not None:
            differences = audit.get("physics_curobo_difference")
            if isinstance(differences, dict):
                collision_world_ok = all(
                    not item.get("missing_in_curobo") and not item.get("unexpected_in_curobo")
                    for item in differences.values()
                    if isinstance(item, dict)
                )
        for path in (
            audit_path,
            object_path,
            safety_path,
            episode_dir / "trajectory_debug.usda",
            episode_dir / "skill_targets_debug.usda",
        ):
            if path.is_file():
                artifact_refs.append(str(path))
        artifact_refs.extend(str(path) for path in sorted(episode_dir.glob("images.rgb.*/demo.mp4")))

    by_code, known_codes = _failure_codes()
    del by_code
    log_signals = [code for code in known_codes if code in stdout]
    log_signals.extend(
        match.group(1)
        for match in re.finditer(r"(?:failure|reason)[=: ]+([A-Za-z0-9_\-]+)", stdout, re.IGNORECASE)
    )
    if timed_out:
        log_signals.insert(0, "TASK_TIMEOUT")
    elif return_code not in {0, None}:
        log_signals.append("PROCESS_FAILED")

    task_success = bool(event and event.get("status") == "success") or "Task is successful" in stdout
    failure_reason = str(event.get("failure_reason") or "") if event else ""
    if not failure_reason and log_signals:
        failure_reason = log_signals[0]
    status = "success" if task_success else "failed"
    bundle = EvidenceBundle(
        attempt_id=attempt_id,
        status=status,
        task_success=task_success,
        event_status=str(event.get("status")) if event else None,
        failure_reason=failure_reason or ("EVENT_MISSING" if event is None else None),
        episode_dir=str(episode_dir) if episode_dir else None,
        collision_world_ok=collision_world_ok,
        object_state_events=object_events,
        safety_events=safety_events,
        artifact_refs=artifact_refs,
        log_signals=list(dict.fromkeys(log_signals)),
    )
    (attempt_dir / "evidence.json").write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def classify_evidence(evidence: EvidenceBundle) -> Diagnosis:
    if evidence.task_success:
        return Diagnosis(
            stage="complete",
            failure_code="NONE",
            category="success",
            root_cause="SimBox reported successful task completion",
            retryable=False,
            recommended_action="continue to the next object subtask",
            evidence_refs=evidence.artifact_refs,
        )
    by_code, known_codes = _failure_codes()
    if evidence.collision_world_ok is False:
        code = "PHYSICS_CUROBO_WORLD_MISMATCH"
        category = "asset_contract"
    elif evidence.safety_events:
        last = evidence.safety_events[-1]
        code = str(last.get("trigger") or evidence.failure_reason or "UNKNOWN_FAILURE")
        category = by_code.get(code, "execution")
    else:
        candidates = [str(evidence.failure_reason or ""), *evidence.log_signals]
        code = next((known for known in known_codes if any(known in item for item in candidates)), "")
        if not code:
            code = str(evidence.failure_reason or "UNKNOWN_FAILURE")
        category = by_code.get(code, "unknown")

    non_retryable = {
        "PHYSICS_CUROBO_WORLD_MISMATCH",
        "ATTACH_COLLISION_CONFIG_MISSING",
        "ATTACH_COLLISION_CONFIG_INVALID",
        "ATTACH_COLLISION_CONFIG_CONFLICT",
        "ATTACH_COLLISION_PRIM_NOT_FOUND",
        "ATTACH_COLLISION_PRIM_NOT_COLLIDABLE",
        "ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT",
        "ATTACH_COLLISION_PRIM_AMBIGUOUS",
        "UNSUPPORTED_CONCURRENT_MANIPULATION",
        "EVENT_MISSING",
        "DOCKER_RUNTIME_UNAVAILABLE",
        "DOCKER_IMAGE_MISSING",
        "DOCKER_START_FAILED",
        "DOCKER_WAIT_FAILED",
        "ISAAC_CONTAINER_FAILED",
    }
    workspace_codes = {
        "NO_GEOMETRY_CANDIDATE",
        "NO_CUROBO_CANDIDATE",
        "NO_COMMON_WORKSPACE_CANDIDATE",
        "NO_COMMON_CUROBO_WORKSPACE_CANDIDATE",
        "NO_JOINT_GRASP_PLAN",
    }
    retryable = code not in non_retryable
    if code in workspace_codes:
        action = "select another geometry-feasible workspace candidate and rerun planning-only"
        workspace_action = "replan"
    elif code == "GRASP_CONTACT_MISSING":
        action = "revise one grasp approach parameter or grasp candidate, then retry"
        workspace_action = "keep"
    elif code in {"attached_object_dropped", "attached_object_translation_slip", "attached_object_rotation_slip"}:
        action = "revise one grasp/close/lift parameter, then retry"
        workspace_action = "keep"
    elif code == "NO_COLLISION_FREE_PREPLACE_PLAN":
        action = "revise the deterministic Place target geometry or workspace candidate"
        workspace_action = "replan"
    elif category in {"configuration", "asset_contract"}:
        action = "repair the deterministic configuration or asset contract before executing again"
        workspace_action = "block"
        retryable = False
    elif category == "unknown":
        action = "inspect structured artifacts and selected keyframes before changing configuration"
        workspace_action = "keep"
        retryable = True
    else:
        action = "revise one parameter directly related to the recorded trigger"
        workspace_action = "keep"
    return Diagnosis(
        stage=category,
        failure_code=code,
        category=category,
        root_cause=f"SimBox evidence selected {code} as the primary failure",
        confidence=1.0 if category != "unknown" else 0.4,
        retryable=retryable,
        recommended_action=action,
        workspace_action=workspace_action,
        evidence_refs=evidence.artifact_refs,
    )
