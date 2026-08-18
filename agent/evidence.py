"""Normalize SimBox episode artifacts into deterministic failure evidence."""

from __future__ import annotations

import json
import math
import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import lmdb
import imageio.v2 as imageio
import numpy as np
import yaml

from workflows.simbox.core.robots.profile import load_robot_profile
from .tools.signatures import variant_signature

from .contracts import (
    Diagnosis,
    EpisodeIdentity,
    EvidenceBundle,
    ExecutionIdentity,
)
from .tools.source_integrity import verify_source_snapshot


AGENT_DIR = Path(__file__).resolve().parent


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    result: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}:{line_number}: record is not an object")
            continue
        result.append(value)
    return result, errors


def _last_event(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    values, errors = _jsonl(path)
    return (values[-1] if values else None), errors


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
    *,
    expected_identity: ExecutionIdentity,
    data_generation_required: bool = False,
    robot_profile_path: str | Path | None = None,
    compiled_task_path: str | Path | None = None,
    source_snapshot_path: str | Path | None = None,
) -> EvidenceBundle:
    event, event_artifact_errors = _last_event(event_path)
    stdout = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    episode_value = str(event.get("primary_episode_dir", "")) if event else ""
    episode_dir = Path(episode_value) if episode_value else None
    episode_path_error = None
    if episode_dir is not None:
        episode_dir = episode_dir.expanduser().resolve()
        attempt_data_dir = (attempt_dir / "data").resolve()
        if not episode_dir.is_dir():
            episode_path_error = "episode directory does not exist"
            episode_dir = None
        else:
            try:
                episode_dir.relative_to(attempt_data_dir)
            except ValueError:
                episode_path_error = (
                    "episode directory is outside the current attempt data directory"
                )
                episode_dir = None

    artifact_refs = [str(path) for path in (event_path, log_path) if path.is_file()]
    object_events: list[dict[str, Any]] = []
    safety_events: list[dict[str, Any]] = []
    object_artifact_errors: list[str] = []
    safety_artifact_errors: list[str] = []
    collision_audit: dict[str, Any] | None = None
    if episode_dir is not None:
        audit_path = episode_dir / "collision_world_audit.json"
        object_path = episode_dir / "object_state_events.jsonl"
        safety_path = episode_dir / "safety_events.jsonl"
        audit = _json(audit_path)
        collision_audit = audit
        object_events, object_artifact_errors = _jsonl(object_path)
        safety_events, safety_artifact_errors = _jsonl(safety_path)
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

    strict_evaluation = _strict_evaluation(
        event,
        episode_dir,
        safety_events,
        collision_audit=collision_audit,
        expected_identity=expected_identity,
        data_generation_required=data_generation_required,
        robot_profile_path=robot_profile_path,
        compiled_task_path=compiled_task_path,
        source_snapshot_path=source_snapshot_path,
        episode_path_error=episode_path_error,
        event_artifact_errors=event_artifact_errors,
        object_artifact_errors=object_artifact_errors,
        safety_artifact_errors=safety_artifact_errors,
    )
    evaluation_path = attempt_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(strict_evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_refs.append(str(evaluation_path))
    task_success = bool(strict_evaluation["strict_success"])
    collision_world_ok = (
        bool(strict_evaluation["checks"]["physics_curobo_exact"])
        if strict_evaluation["checks"]["collision_audit_present"]
        else None
    )
    failure_reason = str(event.get("failure_reason") or "") if event else ""
    if not task_success:
        failure_reason = str(strict_evaluation.get("failure_code") or failure_reason)
    if not failure_reason and log_signals:
        failure_reason = log_signals[0]
    status = "success" if task_success else "failed"
    bundle = EvidenceBundle(
        attempt_id=attempt_id,
        failing_subtask_id=strict_evaluation.get("failing_subtask_id"),
        identity=(
            EpisodeIdentity.from_dict(strict_evaluation["identity"])
            if isinstance(strict_evaluation.get("identity"), dict)
            else None
        ),
        identity_errors=list(strict_evaluation["identity_errors"]),
        variant_signature=strict_evaluation.get("variant_signature"),
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


def _collision_audit_exact(
    differences: dict[str, Any], expected_controllers: set[str]
) -> bool:
    if not differences or set(differences) != expected_controllers:
        return False
    for controller, value in differences.items():
        if not controller or not isinstance(value, dict):
            return False
        missing = value.get("missing_in_curobo")
        unexpected = value.get("unexpected_in_curobo")
        if not isinstance(missing, list) or not isinstance(unexpected, list):
            return False
        if missing or unexpected:
            return False
    return True


def _expected_collision_controllers(
    compiled_document: dict[str, Any] | None,
    robot_profile_path: str | Path | None,
) -> tuple[set[str], str | None]:
    if compiled_document is None:
        return set(), "cannot verify collision controllers without the compiled task"
    if robot_profile_path is None:
        return set(), "cannot verify collision controllers without the robot profile"
    task = compiled_document["tasks"][0]
    robots = task.get("robots")
    if not isinstance(robots, list) or len(robots) != 1 or not isinstance(robots[0], dict):
        return set(), "compiled task must contain exactly one robot instance"
    instance_name = str(robots[0].get("name") or "").strip()
    if not instance_name:
        return set(), "compiled robot instance has no name"
    try:
        profile = load_robot_profile(robot_profile_path)
    except (OSError, ValueError) as exc:
        return set(), f"cannot load collision controller profile: {exc}"
    return {f"{instance_name}/{arm_id}" for arm_id in profile.arms}, None


def _compiled_task(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "compiled task is missing"
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"compiled task is invalid: {exc}"
    if not isinstance(document, dict):
        return None, "compiled task root is not a mapping"
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        return None, "compiled task must contain exactly one task"
    return document, None


def _variant_signature(document: dict[str, Any] | None) -> str | None:
    if document is None:
        return None
    try:
        return variant_signature(document)
    except ValueError:
        return None


def _verified_failing_subtask_id(
    event: dict[str, Any] | None,
    compiled_document: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    raw_value = (event or {}).get("failing_subtask_id")
    if raw_value in {None, ""}:
        return None, None
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None, "episode event failing_subtask_id must be a non-empty string"
    if compiled_document is None:
        return None, "cannot verify failing_subtask_id without the compiled task"
    task = compiled_document["tasks"][0]
    agent_plan = (task.get("metadata") or {}).get("agent_plan")
    subtasks = agent_plan.get("subtasks") if isinstance(agent_plan, dict) else None
    if not isinstance(subtasks, list):
        return None, "compiled task has no metadata.agent_plan.subtasks"
    matches = [
        item
        for item in subtasks
        if isinstance(item, dict) and item.get("subtask_id") == raw_value
    ]
    if len(matches) != 1:
        return None, f"episode event references unknown failing subtask {raw_value!r}"
    return raw_value, None


def _predicate_payload_error(
    predicate_results: Any,
    compiled_document: dict[str, Any] | None,
) -> str | None:
    if compiled_document is None:
        return "cannot verify predicates without the compiled task"
    if not isinstance(predicate_results, list) or not predicate_results:
        return "episode has no terminal predicate results"
    task = compiled_document["tasks"][0]
    agent_plan = (task.get("metadata") or {}).get("agent_plan")
    subtasks = agent_plan.get("subtasks") if isinstance(agent_plan, dict) else None
    if not isinstance(subtasks, list) or not subtasks:
        return "compiled task has no metadata.agent_plan.subtasks"
    expected: dict[str, tuple[str, str, str]] = {}
    for item in subtasks:
        if not isinstance(item, dict):
            return "compiled subtask metadata is not a mapping"
        subtask_id = str(item.get("subtask_id") or "").strip()
        relation = str(item.get("relation") or "").strip()
        manipulated = str(item.get("center_object") or "").strip()
        target = str(item.get("target_object") or "").strip()
        if not subtask_id or not relation or not manipulated or not target:
            return "compiled subtask metadata is incomplete"
        if subtask_id in expected:
            return f"compiled task has duplicate subtask {subtask_id!r}"
        expected[subtask_id] = relation, manipulated, target
    actual: set[str] = set()
    for index, result in enumerate(predicate_results):
        if not isinstance(result, dict):
            return f"predicate result {index} is not a mapping"
        subtask_id = str(result.get("subtask_id") or "").strip()
        if not subtask_id or subtask_id not in expected or subtask_id in actual:
            return f"predicate result {index} has unknown or duplicate subtask_id"
        actual.add(subtask_id)
        relation, manipulated, target = expected[subtask_id]
        if str(result.get("predicate_id") or "").strip() == "":
            return f"predicate result {index} has no predicate_id"
        if result.get("skill") != "place":
            return f"predicate result {index} is not a Place predicate"
        if result.get("relation") != relation:
            return f"predicate result {index} relation does not match compiled task"
        if result.get("objects") != [manipulated, target]:
            return f"predicate result {index} objects do not match compiled task"
        if result.get("terminal_success") is not True or result.get("success") is not True:
            return f"predicate result {index} is not a strict terminal success"
        checks = result.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or any(not isinstance(value, bool) or not value for value in checks.values())
        ):
            return f"predicate result {index} has invalid checks"
        for field in ("measurements", "thresholds"):
            value = result.get(field)
            if not isinstance(value, dict) or not value or not _finite_numeric_tree(value):
                return f"predicate result {index} has invalid {field}"
    if actual != set(expected):
        return "predicate results do not cover every compiled subtask"
    return None


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_finite_numeric_tree(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str) and key and _finite_numeric_tree(item)
            for key, item in value.items()
        )
    return False


def _verified_episode_identity(
    event: dict[str, Any] | None,
    collision_audit: dict[str, Any] | None,
    compiled_document: dict[str, Any] | None,
    expected: ExecutionIdentity,
) -> tuple[EpisodeIdentity | None, list[str]]:
    errors: list[str] = []
    if event is None:
        return None, ["episode event is missing"]
    for field, expected_value in expected.to_dict().items():
        if event.get(field) != expected_value:
            errors.append(
                f"episode event {field}={event.get(field)!r} does not match {expected_value!r}"
            )
    world_revision = event.get("world_revision")
    if isinstance(world_revision, bool) or not isinstance(world_revision, int) or world_revision < 0:
        errors.append("episode event world_revision must be a non-negative integer")
    audit_revision = (
        collision_audit.get("world_revision")
        if isinstance(collision_audit, dict)
        else None
    )
    if audit_revision != world_revision:
        errors.append("episode event and collision audit world_revision disagree")
    if compiled_document is None:
        errors.append("compiled task identity is unavailable")
    else:
        task = compiled_document["tasks"][0]
        agent_plan = (task.get("metadata") or {}).get("agent_plan")
        if not isinstance(agent_plan, dict):
            errors.append("compiled task has no metadata.agent_plan identity")
        else:
            compiled_values = {
                "variant_id": agent_plan.get("execution_variant_id"),
                "profile_id": agent_plan.get("robot_profile_id"),
                "profile_hash": agent_plan.get("robot_profile_hash"),
                "scene_revision": agent_plan.get("scene_revision"),
            }
            for field, value in compiled_values.items():
                if value != getattr(expected, field):
                    errors.append(
                        f"compiled task {field}={value!r} does not match {getattr(expected, field)!r}"
                    )
    if errors:
        return None, errors
    return EpisodeIdentity(**expected.to_dict(), world_revision=world_revision), []


def _strict_evaluation(
    event: dict[str, Any] | None,
    episode_dir: Path | None,
    safety_events: list[dict[str, Any]],
    *,
    collision_audit: dict[str, Any] | None,
    expected_identity: ExecutionIdentity,
    data_generation_required: bool,
    robot_profile_path: str | Path | None,
    compiled_task_path: str | Path | None,
    source_snapshot_path: str | Path | None,
    episode_path_error: str | None = None,
    event_artifact_errors: list[str] | None = None,
    object_artifact_errors: list[str] | None = None,
    safety_artifact_errors: list[str] | None = None,
) -> dict[str, Any]:
    event_artifact_errors = list(event_artifact_errors or [])
    object_artifact_errors = list(object_artifact_errors or [])
    safety_artifact_errors = list(safety_artifact_errors or [])
    terminal_success = bool(
        event
        and not event_artifact_errors
        and event.get("event") == "episode_saved"
        and event.get("finalized") is True
        and event.get("status") == "success"
    )
    predicate_results = (event or {}).get("predicate_results")
    safety_artifact_present = bool(
        episode_dir is not None and (episode_dir / "safety_events.jsonl").is_file()
    )
    safety_ok = (
        safety_artifact_present
        and not safety_artifact_errors
        and not safety_events
    )
    source_integrity = verify_source_snapshot(
        source_snapshot_path,
        expected_identity.source_hash,
    )
    compiled_document, compiled_error = _compiled_task(compiled_task_path)
    identity, identity_errors = _verified_episode_identity(
        event,
        collision_audit,
        compiled_document,
        expected_identity,
    )
    if compiled_error:
        identity_errors.append(compiled_error)
    predicate_payload_error = _predicate_payload_error(
        predicate_results,
        compiled_document,
    )
    predicate_success = bool(
        event
        and event.get("task_predicate_success") is True
        and predicate_payload_error is None
    )
    if predicate_payload_error:
        identity_errors.append(predicate_payload_error)
    expected_controllers, controller_error = _expected_collision_controllers(
        compiled_document,
        robot_profile_path,
    )
    if controller_error:
        identity_errors.append(controller_error)
    differences = (
        collision_audit.get("physics_curobo_difference")
        if isinstance(collision_audit, dict)
        else None
    )
    collision_audit_present = isinstance(differences, dict)
    collision_exact = bool(
        collision_audit_present
        and not controller_error
        and _collision_audit_exact(differences, expected_controllers)
    )
    failing_subtask_id, failing_subtask_error = _verified_failing_subtask_id(
        event, compiled_document
    )
    if failing_subtask_error:
        identity_errors.append(failing_subtask_error)
    identity_errors.extend(source_integrity.errors)
    if episode_path_error:
        identity_errors.append(episode_path_error)
    identity_errors.extend(event_artifact_errors)
    identity_errors.extend(object_artifact_errors)
    identity_errors.extend(safety_artifact_errors)
    variant_signature = _variant_signature(compiled_document)
    identity_consistent = bool(
        identity is not None
        and source_integrity.identity_consistent
        and not identity_errors
        and variant_signature is not None
    )
    data_checks = _data_integrity(
        episode_dir,
        event,
        robot_profile_path=robot_profile_path,
        expected_profile_hash=expected_identity.profile_hash,
        compiled_task_path=compiled_task_path,
    ) if data_generation_required else {
        "required": False,
        "ok": True,
        "reason": "data generation was not requested",
        "num_steps": int((event or {}).get("num_steps") or 0),
        "video_stream_count": int((event or {}).get("video_stream_count") or 0),
        "video_files": [],
        "lmdb_files": [],
        "metadata_files": [],
    }
    checks = {
        "terminal_success": terminal_success,
        "task_predicate_success": predicate_success,
        "safety_ok": safety_ok,
        "safety_artifact_present": safety_artifact_present,
        "event_artifact_valid": not event_artifact_errors,
        "object_artifact_valid": not object_artifact_errors,
        "safety_artifact_valid": not safety_artifact_errors,
        "collision_audit_present": collision_audit_present,
        "physics_curobo_exact": collision_exact,
        "data_integrity_ok": bool(data_checks["ok"]),
        "source_unchanged": source_integrity.source_unchanged,
        "identity_consistent": identity_consistent,
    }
    strict_success = all(checks.values())
    failure_code = None
    if event_artifact_errors:
        failure_code = "EVENT_ARTIFACT_INVALID"
    elif event is None:
        failure_code = "EVENT_MISSING"
    elif not terminal_success:
        failure_code = str(event.get("failure_reason") or "EPISODE_TERMINAL_FAILED")
    elif not predicate_success:
        failure_code = "PLACE_PREDICATE_FAILED"
    elif not safety_artifact_present:
        failure_code = "SAFETY_ARTIFACT_MISSING"
    elif safety_artifact_errors:
        failure_code = "SAFETY_ARTIFACT_INVALID"
    elif not safety_ok:
        failure_code = str(safety_events[-1].get("trigger") or "SAFETY_VIOLATION")
    elif not collision_audit_present:
        failure_code = "COLLISION_WORLD_AUDIT_MISSING"
    elif not collision_exact:
        failure_code = "PHYSICS_CUROBO_WORLD_MISMATCH"
    elif not data_checks["ok"]:
        failure_code = "DATA_INTEGRITY_FAILED"
    elif not checks["source_unchanged"]:
        failure_code = "SOURCE_INTEGRITY_FAILED"
    elif not identity_consistent:
        failure_code = "IDENTITY_MISMATCH"
    return {
        "strict_success": strict_success,
        "failure_code": failure_code,
        "failing_subtask_id": failing_subtask_id,
        "checks": checks,
        "data_integrity": data_checks,
        "predicate_results": predicate_results if isinstance(predicate_results, list) else [],
        "identity": identity.to_dict() if identity is not None else None,
        "identity_errors": identity_errors,
        "variant_signature": variant_signature,
    }


def _data_integrity(
    episode_dir: Path | None,
    event: dict[str, Any] | None,
    *,
    robot_profile_path: str | Path | None,
    expected_profile_hash: str | None,
    compiled_task_path: str | Path | None,
) -> dict[str, Any]:
    if episode_dir is None:
        return {
            "required": True,
            "ok": False,
            "reason": "episode directory is missing",
            "num_steps": 0,
            "video_stream_count": 0,
            "video_files": [],
            "lmdb_files": [],
            "metadata_files": [],
        }
    num_steps = int((event or {}).get("num_steps") or 0)
    video_count = int((event or {}).get("video_stream_count") or 0)
    video_files = sorted(str(path) for path in episode_dir.rglob("demo.mp4") if path.is_file())
    lmdb_path = episode_dir / "lmdb"
    canonical_data_path = lmdb_path / "data.mdb"
    lmdb_files = [str(canonical_data_path)] if canonical_data_path.is_file() else []
    unexpected_lmdb_files = sorted(
        str(path)
        for path in episode_dir.rglob("data.mdb")
        if path.is_file() and path != canonical_data_path
    )
    metadata_files = sorted(
        str(path)
        for pattern in ("meta_info.pkl", "metadata.json", "episode_metadata.json")
        for path in episode_dir.rglob(pattern)
        if path.is_file()
    )
    missing: list[str] = []
    invalid: list[str] = []
    expected_cameras: list[str] = []
    expected_proprio: list[str] = []
    expected_actions: list[str] = []
    expected_sample_sizes: dict[str, int | None] = {}
    profile_cameras: list[str] = []
    profile_hash = ""
    if robot_profile_path is None:
        missing.append("robot profile contract")
    else:
        try:
            profile = load_robot_profile(robot_profile_path)
        except (OSError, ValueError) as exc:
            invalid.append(f"robot profile contract: {exc}")
        else:
            profile_hash = profile.profile_hash
            if expected_profile_hash and profile_hash != expected_profile_hash:
                invalid.append("robot profile hash mismatch")
            profile_cameras = [
                f"images.rgb.{camera.save_name}" for camera in profile.camera_rig
            ]
            for arm_id, adapter in profile.data_adapter.arms.items():
                arm_profile = profile.arms[arm_id]
                state_keys = [
                    adapter.joint_position_key,
                    adapter.gripper_position_key,
                ]
                if adapter.gripper_pose_key:
                    state_keys.append(adapter.gripper_pose_key)
                expected_proprio.extend(state_keys)
                prefix = f"{adapter.action_name}_" if adapter.action_name else ""
                master_joint = f"master_actions.{prefix}joint.position"
                master_gripper = f"master_actions.{prefix}gripper.position"
                master_openness = f"master_actions.{prefix}gripper.openness"
                derived_state_keys = [
                    key.replace("states.", "actions.", 1)
                    for key in state_keys
                    if key.startswith("states.")
                ]
                arm_actions = [
                    master_joint,
                    master_gripper,
                    master_openness,
                    *(
                        [f"master_actions.{prefix}gripper.pose"]
                        if adapter.gripper_pose_key
                        else []
                    ),
                    *derived_state_keys,
                    f"actions.{prefix}gripper.openness",
                ]
                expected_actions.extend(arm_actions)
                joint_size = len(arm_profile.command_joint_names)
                expected_sample_sizes[adapter.joint_position_key] = joint_size
                expected_sample_sizes[master_joint] = joint_size
                expected_sample_sizes[
                    adapter.joint_position_key.replace("states.", "actions.", 1)
                ] = joint_size
                expected_sample_sizes[adapter.gripper_position_key] = None
                expected_sample_sizes[master_gripper] = None
                expected_sample_sizes[
                    adapter.gripper_position_key.replace("states.", "actions.", 1)
                ] = None
                expected_sample_sizes[master_openness] = 1
                expected_sample_sizes[f"actions.{prefix}gripper.openness"] = 1
                if adapter.gripper_pose_key:
                    expected_sample_sizes[adapter.gripper_pose_key] = 6
                    expected_sample_sizes[
                        adapter.gripper_pose_key.replace("states.", "actions.", 1)
                    ] = 6
                    expected_sample_sizes[f"master_actions.{prefix}gripper.pose"] = 6
    if compiled_task_path is None:
        missing.append("compiled task camera contract")
    else:
        try:
            compiled_document = yaml.safe_load(
                Path(compiled_task_path).read_text(encoding="utf-8")
            )
            tasks = compiled_document.get("tasks") if isinstance(compiled_document, dict) else None
            if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
                raise ValueError("compiled task must contain exactly one task")
            compiled_task = tasks[0]
            agent_plan = (compiled_task.get("metadata") or {}).get("agent_plan")
            if not isinstance(agent_plan, dict):
                raise ValueError("compiled task has no metadata.agent_plan")
            if agent_plan.get("robot_profile_hash") != profile_hash:
                raise ValueError("compiled task robot profile hash mismatch")
            cameras = compiled_task.get("cameras")
            if not isinstance(cameras, list) or not cameras:
                raise ValueError("compiled task has no cameras")
            save_names = []
            for camera in cameras:
                if not isinstance(camera, dict) or not camera.get("save_name"):
                    raise ValueError("compiled camera has no save_name")
                save_names.append(str(camera["save_name"]))
            if len(save_names) != len(set(save_names)):
                raise ValueError("compiled camera save_name values are not unique")
            expected_cameras = [f"images.rgb.{name}" for name in save_names]
            missing_profile_cameras = sorted(set(profile_cameras) - set(expected_cameras))
            if missing_profile_cameras:
                raise ValueError(
                    "compiled task is missing profile cameras: "
                    + ", ".join(missing_profile_cameras)
                )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            invalid.append(f"compiled task camera contract: {exc}")
    if num_steps <= 0:
        missing.append("action/state steps")
    if video_count <= 0 or not video_files:
        missing.append("camera video streams")
    elif expected_cameras and video_count != len(expected_cameras):
        invalid.append(
            f"event video_stream_count={video_count} does not match expected={len(expected_cameras)}"
        )
    if not canonical_data_path.is_file():
        missing.append("canonical LMDB lmdb/data.mdb")
    elif lmdb_path.is_symlink() or canonical_data_path.is_symlink():
        invalid.append("canonical LMDB path must not contain symbolic links")
    if unexpected_lmdb_files:
        invalid.append(
            "unexpected LMDB data stores: " + ", ".join(unexpected_lmdb_files)
        )
    if not metadata_files:
        missing.append("episode metadata")
    meta: dict[str, Any] = {}
    meta_path = episode_dir / "meta_info.pkl"
    if meta_path.is_file():
        try:
            with meta_path.open("rb") as stream:
                loaded_meta = pickle.load(stream)
            if not isinstance(loaded_meta, dict):
                raise TypeError("metadata root is not a mapping")
            meta = loaded_meta
        except (OSError, pickle.PickleError, EOFError, TypeError) as exc:
            invalid.append(f"episode metadata: {exc}")
    meta_steps = int(meta.get("num_steps") or 0) if meta else 0
    if meta and meta_steps != num_steps:
        invalid.append(f"metadata num_steps={meta_steps} does not match event={num_steps}")
    meta_keys = meta.get("keys") if isinstance(meta.get("keys"), dict) else {}
    proprio_keys = _decoded_keys(meta_keys.get("proprio_data", []))
    action_keys = _decoded_keys(meta_keys.get("action_data", []))
    for key in expected_proprio:
        if key not in proprio_keys:
            missing.append(f"proprio key {key}")
    for key in expected_actions:
        if key not in action_keys:
            missing.append(f"action key {key}")
    image_steps = meta.get("image_valid_step_ids")
    image_steps = image_steps if isinstance(image_steps, dict) else {}
    expected_image_lmdb_keys: list[str] = []
    for camera_key in expected_cameras:
        frame_ids = image_steps.get(camera_key)
        if camera_key not in meta_keys or not isinstance(frame_ids, (list, tuple)):
            missing.append(f"camera stream {camera_key}")
            continue
        if (
            len(frame_ids) != num_steps
            or any(not isinstance(frame_id, int) for frame_id in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
        ):
            invalid.append(
                f"camera stream {camera_key} has invalid frame ids for {num_steps} steps"
            )
        camera_lmdb_keys = [f"{camera_key}/{frame_id:04d}" for frame_id in frame_ids]
        expected_image_lmdb_keys.extend(camera_lmdb_keys)
        if _decoded_keys(meta_keys.get(camera_key, [])) != set(camera_lmdb_keys):
            invalid.append(f"camera stream {camera_key} metadata keys do not match frame ids")
        video_path = episode_dir / camera_key / "demo.mp4"
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            missing.append(f"camera video {camera_key}")
        else:
            try:
                frame_count = _video_frame_count(video_path)
            except (OSError, RuntimeError, ValueError) as exc:
                invalid.append(f"camera video {camera_key}: {exc}")
            else:
                if frame_count != len(frame_ids):
                    invalid.append(
                        f"camera video {camera_key} has {frame_count} decoded frames "
                        f"for {len(frame_ids)} metadata frames"
                    )

    if canonical_data_path.is_file():
        try:
            environment = lmdb.open(
                str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=1
            )
            try:
                with environment.begin() as transaction:
                    for key in [*expected_proprio, *expected_actions]:
                        raw = transaction.get(key.encode("utf-8"))
                        if raw is None:
                            missing.append(f"LMDB key {key}")
                            continue
                        values = pickle.loads(raw)
                        if not isinstance(values, (list, tuple)) or len(values) != num_steps:
                            actual = len(values) if isinstance(values, (list, tuple)) else "non-sequence"
                            invalid.append(
                                f"LMDB key {key} has {actual} samples for {num_steps} steps"
                            )
                            continue
                        sample_error = _numeric_sample_error(
                            values,
                            expected_sample_sizes.get(key),
                        )
                        if sample_error:
                            invalid.append(f"LMDB key {key} {sample_error}")
                    for key in expected_image_lmdb_keys:
                        raw = transaction.get(key.encode("utf-8"))
                        if raw is None:
                            missing.append(f"LMDB image key {key}")
                            continue
                        try:
                            image = pickle.loads(raw)
                        except (pickle.PickleError, EOFError, TypeError, ValueError) as exc:
                            invalid.append(f"LMDB image key {key}: {exc}")
                            continue
                        image_error = _encoded_image_error(image)
                        if image_error:
                            invalid.append(f"LMDB image key {key} {image_error}")
            finally:
                environment.close()
        except (lmdb.Error, OSError, pickle.PickleError, EOFError) as exc:
            invalid.append(f"LMDB data: {exc}")
    return {
        "required": True,
        "ok": not missing and not invalid,
        "reason": "; ".join(
            [
                *(f"missing {item}" for item in missing),
                *(f"invalid {item}" for item in invalid),
            ]
        ),
        "num_steps": num_steps,
        "video_stream_count": video_count,
        "video_files": video_files,
        "lmdb_files": lmdb_files,
        "unexpected_lmdb_files": unexpected_lmdb_files,
        "metadata_files": metadata_files,
        "profile_hash": profile_hash,
        "expected_cameras": expected_cameras,
        "expected_proprio_keys": expected_proprio,
        "expected_action_keys": expected_actions,
        "missing": missing,
        "invalid": invalid,
    }


def _decoded_keys(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple)):
        return set()
    return {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    }


def _numeric_sample_error(
    values: list[Any] | tuple[Any, ...],
    expected_size: int | None,
) -> str | None:
    sample_shapes: set[tuple[int, ...]] = set()
    for index, value in enumerate(values):
        try:
            sample = np.asarray(value)
        except (TypeError, ValueError) as exc:
            return f"sample {index} is not numeric: {exc}"
        if sample.dtype.kind not in "biufc" or sample.size == 0:
            return f"sample {index} is not a non-empty numeric value"
        try:
            finite = bool(np.isfinite(sample).all())
        except TypeError:
            finite = False
        if not finite:
            return f"sample {index} contains non-finite values"
        normalized_shape = sample.shape if sample.shape else (1,)
        sample_shapes.add(normalized_shape)
        if expected_size is not None and sample.size != expected_size:
            return (
                f"sample {index} has {sample.size} values; "
                f"expected {expected_size}"
            )
    if len(sample_shapes) != 1:
        return f"has inconsistent sample shapes {sorted(sample_shapes)}"
    return None


def _encoded_image_error(value: Any) -> str | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(value, dtype=np.uint8)
    else:
        try:
            encoded = np.asarray(value, dtype=np.uint8).reshape(-1)
        except (TypeError, ValueError) as exc:
            return f"is not an encoded byte array: {exc}"
    if encoded.size == 0:
        return "is empty"
    try:
        decoded = imageio.imread(BytesIO(encoded.tobytes()))
    except Exception:
        return "cannot be decoded"
    if not isinstance(decoded, np.ndarray) or decoded.ndim not in {2, 3} or decoded.size == 0:
        return "does not decode to a non-empty image"
    return None


def _video_frame_count(path: Path) -> int:
    try:
        reader = imageio.get_reader(path)
        try:
            count = sum(1 for _ in reader)
        finally:
            reader.close()
    except Exception as exc:
        raise RuntimeError("video cannot be decoded") from exc
    if count <= 0:
        raise ValueError("video contains no decoded frames")
    return count


def classify_evidence(evidence: EvidenceBundle) -> Diagnosis:
    if evidence.task_success:
        return Diagnosis(
            stage="complete",
            failure_code="NONE",
            failing_subtask_id=None,
            category="success",
            root_cause="SimBox reported successful task completion",
            retryable=False,
            recommended_action="continue to the next object subtask",
            evidence_refs=evidence.artifact_refs,
        )
    by_code, known_codes = _failure_codes()
    if evidence.failure_reason == "COLLISION_WORLD_AUDIT_MISSING":
        code = "COLLISION_WORLD_AUDIT_MISSING"
        category = "asset_contract"
    elif evidence.failure_reason in {
        "SAFETY_ARTIFACT_MISSING",
        "DATA_INTEGRITY_FAILED",
        "SOURCE_INTEGRITY_FAILED",
    }:
        code = str(evidence.failure_reason)
        category = "data_integrity"
    elif evidence.failure_reason == "IDENTITY_MISMATCH":
        code = "IDENTITY_MISMATCH"
        category = "data_integrity"
    elif evidence.collision_world_ok is False:
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
        "COLLISION_WORLD_AUDIT_MISSING",
        "DATA_INTEGRITY_FAILED",
        "SAFETY_ARTIFACT_MISSING",
        "SOURCE_INTEGRITY_FAILED",
        "IDENTITY_MISMATCH",
        "ATTACH_COLLISION_CONFIG_MISSING",
        "ATTACH_COLLISION_CONFIG_INVALID",
        "ATTACH_COLLISION_CONFIG_CONFLICT",
        "ATTACH_COLLISION_PRIM_NOT_FOUND",
        "ATTACH_COLLISION_PRIM_NOT_COLLIDABLE",
        "ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT",
        "ATTACH_COLLISION_PRIM_AMBIGUOUS",
        "UNSUPPORTED_CONCURRENT_MANIPULATION",
        "EVENT_MISSING",
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
        failing_subtask_id=evidence.failing_subtask_id,
        category=category,
        root_cause=f"SimBox evidence selected {code} as the primary failure",
        confidence=1.0 if category != "unknown" else 0.4,
        retryable=retryable,
        recommended_action=action,
        workspace_action=workspace_action,
        evidence_refs=evidence.artifact_refs,
    )
