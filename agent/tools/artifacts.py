"""Content-addressed index for one execution-variant attempt."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import Field

from ..contracts import ContractModel, EpisodeIdentity, dump_contract, load_contract
from .signatures import variant_signature as compute_variant_signature


ARTIFACT_ORDER = (
    "compiled_task",
    "scene_layout",
    "static_validation",
    "workspace_selection",
    "spawn_settle",
    "collision_audit",
    "pick_probe",
    "place_probe",
    "trace",
    "log",
    "screenshots",
    "videos",
    "data",
    "evaluation",
    "evidence",
)

MISSING_ARTIFACT_CODES = {
    name: f"ARTIFACT_{name.upper()}_MISSING" for name in ARTIFACT_ORDER
}

ALWAYS_REQUIRED = frozenset(
    {
        "compiled_task",
        "static_validation",
        "workspace_selection",
        "spawn_settle",
        "collision_audit",
        "pick_probe",
        "place_probe",
        "trace",
        "log",
        "evaluation",
        "evidence",
    }
)


class ArtifactMember(ContractModel):
    path: str
    sha256: str
    size_bytes: int


class ArtifactIndexEntry(ContractModel):
    required: bool
    present: bool
    path: str | None = None
    sha256: str | None = None
    kind: Literal["missing", "file", "directory", "collection"]
    size_bytes: int = 0
    members: list[ArtifactMember] = Field(default_factory=list)
    failure_code: str | None = None


class VariantArtifactManifest(ContractModel):
    schema_version: int = 1
    variant_id: str
    attempt_id: str
    identity: EpisodeIdentity | None = None
    variant_signature: str | None = None
    variant_root: str
    attempt_dir: str
    workspace_selection_path: str
    data_required: bool
    complete: bool
    failure_codes: list[str]
    artifacts: dict[str, ArtifactIndexEntry]


def verify_variant_artifact_manifest(path: Path) -> VariantArtifactManifest:
    """Load one canonical attempt manifest and verify every indexed byte."""

    manifest_path = path.expanduser().resolve()
    manifest = load_contract(VariantArtifactManifest, manifest_path)
    variant_root = Path(manifest.variant_root).expanduser().resolve()
    attempt_dir = Path(manifest.attempt_dir).expanduser().resolve()
    workspace_selection_path = (
        Path(manifest.workspace_selection_path).expanduser().resolve()
    )

    if manifest_path != attempt_dir / "artifact_manifest.json":
        raise ValueError("artifact manifest must be stored in its indexed attempt directory")
    if not _is_within(attempt_dir, variant_root):
        raise ValueError("artifact attempt directory is outside variant_root")
    if manifest.variant_id != variant_root.name:
        raise ValueError("artifact manifest variant_id disagrees with variant_root")
    if manifest.attempt_id != attempt_dir.name:
        raise ValueError("artifact manifest attempt_id disagrees with attempt_dir")
    if not _is_within(workspace_selection_path, variant_root):
        raise ValueError("workspace selection is outside variant_root")
    if not manifest.complete or manifest.failure_codes:
        raise ValueError("artifact manifest is incomplete")
    if manifest.identity is None:
        raise ValueError("artifact manifest identity is missing")
    if manifest.identity.variant_id != manifest.variant_id:
        raise ValueError("artifact manifest identity disagrees with variant_id")
    if not _is_sha256(manifest.variant_signature):
        raise ValueError("artifact manifest variant_signature is not a SHA-256 digest")
    if set(manifest.artifacts) != set(ARTIFACT_ORDER):
        raise ValueError("artifact manifest does not contain the canonical artifact set")

    expected_required = {
        name: name in ALWAYS_REQUIRED
        or (name == "scene_layout" and manifest.identity.scene_revision != "source")
        or (manifest.data_required and name in {"videos", "data"})
        for name in ARTIFACT_ORDER
    }
    for name in ARTIFACT_ORDER:
        entry = manifest.artifacts[name]
        if entry.required != expected_required[name]:
            raise ValueError(f"artifact required policy disagrees for {name}")
        if entry.required and not entry.present:
            raise ValueError(f"required artifact is missing: {name}")
        _verify_artifact_entry(name, entry, variant_root)

    _require_single_file(
        manifest.artifacts["compiled_task"], attempt_dir / "task.yaml", "compiled_task"
    )
    _require_single_file(
        manifest.artifacts["workspace_selection"],
        workspace_selection_path,
        "workspace_selection",
    )
    _require_single_file(
        manifest.artifacts["evaluation"],
        attempt_dir / "evaluation.json",
        "evaluation",
    )
    _require_single_file(
        manifest.artifacts["evidence"], attempt_dir / "evidence.json", "evidence"
    )
    _require_single_file(
        manifest.artifacts["trace"], attempt_dir / "trace.jsonl", "trace"
    )
    compiled_workspace_id, compiled_scene_revision = _compiled_workspace_identity(
        attempt_dir / "task.yaml"
    )
    selected_workspace_id = _workspace_selection_candidate_id(
        workspace_selection_path
    )
    if (
        compiled_workspace_id is None
        or selected_workspace_id is None
        or compiled_workspace_id != selected_workspace_id
    ):
        raise ValueError("compiled task disagrees with selected workspace candidate")
    if compiled_scene_revision != manifest.identity.scene_revision:
        raise ValueError("compiled task disagrees with artifact manifest scene revision")
    if _compiled_task_signature(attempt_dir / "task.yaml") != manifest.variant_signature:
        raise ValueError("compiled task disagrees with artifact manifest variant signature")
    _verify_scene_layout_artifacts(
        manifest.artifacts["scene_layout"],
        attempt_dir / "task.yaml",
        compiled_scene_revision,
        variant_root,
    )
    for member in manifest.artifacts["static_validation"].members:
        validation = _read_json_object(Path(member.path))
        if not validation or validation.get("scene_revision") != compiled_scene_revision:
            raise ValueError("static validation belongs to another scene revision")
    rebuilt = build_variant_artifact_manifest(
        variant_root,
        attempt_dir,
        workspace_selection_path,
        data_required=manifest.data_required,
    )
    if rebuilt != manifest:
        raise ValueError("artifact manifest disagrees with the canonical attempt index")
    return manifest


def build_variant_artifact_manifest(
    variant_root: Path,
    attempt_dir: Path,
    workspace_selection_path: Path,
    *,
    data_required: bool | None = None,
) -> VariantArtifactManifest:
    """Index existing attempt evidence without copying or synthesizing artifacts."""

    variant_root = variant_root.expanduser().resolve()
    attempt_dir = attempt_dir.expanduser().resolve()
    workspace_selection_path = workspace_selection_path.expanduser().resolve()
    evidence_path = attempt_dir / "evidence.json"
    evaluation_path = attempt_dir / "evaluation.json"
    compiled_task_path = attempt_dir / "task.yaml"
    compiled_workspace_id, compiled_scene_revision = _compiled_workspace_identity(
        compiled_task_path
    )
    compiled_variant_signature = _compiled_task_signature(compiled_task_path)
    selected_workspace_id = _workspace_selection_candidate_id(
        workspace_selection_path
    )
    episode_dir = _episode_dir(evidence_path, attempt_dir / "episode_events.jsonl")
    workspace_manifests = _selected_workspace_manifests(
        variant_root, workspace_selection_path
    )
    pick_probes, place_probes, probe_spawn_settle = _probe_artifacts(
        workspace_manifests
    )
    spawn_settle_paths = [attempt_dir / "spawn_settle.json", *probe_spawn_settle]
    collision_audit_paths = [attempt_dir / "collision_world_audit.json"]
    if episode_dir is not None:
        spawn_settle_paths.append(episode_dir / "spawn_settle.json")
        collision_audit_paths.append(episode_dir / "collision_world_audit.json")

    if data_required is None:
        data_required = _evaluation_requires_data(evaluation_path)
    required = {
        name: name in ALWAYS_REQUIRED
        or (name == "scene_layout" and compiled_scene_revision not in {None, "source"})
        or (data_required and name in {"videos", "data"})
        for name in ARTIFACT_ORDER
    }

    locations: dict[str, list[Path]] = {
        "compiled_task": [compiled_task_path],
        "scene_layout": _scene_layout_paths(
            compiled_task_path, variant_root, compiled_scene_revision
        ),
        "static_validation": _static_validation_paths(
            variant_root, compiled_scene_revision
        ),
        "workspace_selection": [workspace_selection_path],
        "spawn_settle": _existing_paths(spawn_settle_paths),
        "collision_audit": _existing_paths(collision_audit_paths),
        "pick_probe": pick_probes,
        "place_probe": place_probes,
        "trace": [attempt_dir / "trace.jsonl"],
        "log": [attempt_dir / "stdout.log"],
        "screenshots": _screenshot_paths(attempt_dir, episode_dir),
        "videos": _video_paths(episode_dir),
        "data": _data_paths(attempt_dir, episode_dir),
        "evaluation": [evaluation_path],
        "evidence": [evidence_path],
    }
    artifacts = {
        name: _index_paths(
            locations[name], required[name], MISSING_ARTIFACT_CODES[name]
        )
        for name in ARTIFACT_ORDER
    }
    failure_codes = [
        entry.failure_code
        for entry in artifacts.values()
        if entry.failure_code is not None
    ]
    if compiled_task_path.is_file() and (
        compiled_workspace_id is None
        or selected_workspace_id is None
        or compiled_workspace_id != selected_workspace_id
    ):
        failure_codes.append("ARTIFACT_WORKSPACE_IDENTITY_MISMATCH")
    evidence_value = _read_json_object(evidence_path) or {}
    identity_value = evidence_value.get("identity")
    try:
        identity = (
            EpisodeIdentity.from_dict(identity_value)
            if isinstance(identity_value, dict)
            else None
        )
    except (TypeError, ValueError):
        identity = None
    variant_signature = evidence_value.get("variant_signature")
    if identity is None or not isinstance(variant_signature, str):
        failure_codes.append("ARTIFACT_IDENTITY_MISSING")
    elif identity.variant_id != variant_root.name:
        failure_codes.append("ARTIFACT_IDENTITY_MISMATCH")
    elif len(variant_signature) != 64 or any(
        character not in "0123456789abcdef" for character in variant_signature
    ):
        failure_codes.append("ARTIFACT_IDENTITY_MISMATCH")
    elif variant_signature != compiled_variant_signature:
        failure_codes.append("ARTIFACT_VARIANT_SIGNATURE_MISMATCH")
    elif compiled_scene_revision is None or (
        identity.scene_revision != compiled_scene_revision
    ):
        failure_codes.append("ARTIFACT_SCENE_REVISION_MISMATCH")
    return VariantArtifactManifest(
        variant_id=variant_root.name,
        attempt_id=attempt_dir.name,
        identity=identity,
        variant_signature=(
            variant_signature if isinstance(variant_signature, str) else None
        ),
        variant_root=str(variant_root),
        attempt_dir=str(attempt_dir),
        workspace_selection_path=str(workspace_selection_path),
        data_required=bool(data_required),
        complete=not failure_codes,
        failure_codes=failure_codes,
        artifacts=artifacts,
    )


def write_variant_artifact_manifest(
    variant_root: Path,
    attempt_dir: Path,
    workspace_selection_path: Path,
    *,
    output_path: Path | None = None,
    data_required: bool | None = None,
) -> VariantArtifactManifest:
    manifest = build_variant_artifact_manifest(
        variant_root,
        attempt_dir,
        workspace_selection_path,
        data_required=data_required,
    )
    dump_contract(manifest, output_path or attempt_dir / "artifact_manifest.json")
    return manifest


def _episode_dir(evidence_path: Path, event_path: Path) -> Path | None:
    evidence = _read_json_object(evidence_path)
    raw_path = evidence.get("episode_dir") if evidence else None
    if not raw_path and event_path.is_file():
        for line in reversed(
            event_path.read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("primary_episode_dir"):
                raw_path = event["primary_episode_dir"]
                break
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = evidence_path.parent / path
    path = path.resolve()
    return path if path.is_dir() else None


@dataclass(frozen=True)
class _SelectedWorkspace:
    path: Path
    candidate_id: str
    arm: str
    seed: int


def _selected_workspace_manifests(
    variant_root: Path, workspace_selection_path: Path
) -> list[_SelectedWorkspace]:
    selection = _read_json_object(workspace_selection_path)
    if selection is None:
        return []
    candidate = selection.get("candidate")
    seed = selection.get("seed")
    mode = selection.get("mode")
    if not isinstance(candidate, dict) or not isinstance(seed, int) or seed < 0:
        return []
    selected_id = str(candidate.get("candidate_id") or "")
    if not selected_id:
        return []
    selected: list[_SelectedWorkspace] = []
    if mode == "single":
        subtasks = selection.get("subtasks")
        if not isinstance(subtasks, list) or len(subtasks) != 1:
            return []
        subtask = subtasks[0]
        if not isinstance(subtask, dict):
            return []
        subtask_id = str(subtask.get("subtask_id") or "")
        arm = str(subtask.get("arm") or "")
        raw_path = subtask.get("workspace_manifest_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return []
        path = Path(raw_path).expanduser().resolve()
        if not _is_within(path, variant_root):
            return []
        if _manifest_selection_matches(path, selected_id, arm):
            selected.append(_SelectedWorkspace(path.resolve(), selected_id, arm, seed))
        return selected
    if mode != "common":
        return []
    validated = candidate.get("validated_subtasks")
    if not isinstance(validated, list) or not validated:
        return []
    for subtask in validated:
        if not isinstance(subtask, dict) or not isinstance(subtask.get("selected"), dict):
            return []
        subtask_id = str(subtask.get("subtask_id") or "")
        arm = str(subtask.get("arm") or "")
        probe_candidate_id = str(subtask["selected"].get("candidate_id") or "")
        raw_path = subtask.get("workspace_manifest_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return []
        path = Path(raw_path).expanduser().resolve()
        if not _is_within(path, variant_root):
            return []
        if not _manifest_selection_matches(path, probe_candidate_id, arm):
            return []
        selected.append(
            _SelectedWorkspace(path.resolve(), probe_candidate_id, arm, seed)
        )
    return selected


def _manifest_selection_matches(path: Path, candidate_id: str, arm: str) -> bool:
    manifest = _read_json_object(path)
    selected = manifest.get("selected_candidate") if manifest else None
    return bool(
        isinstance(selected, dict)
        and str(selected.get("candidate_id") or "") == candidate_id
        and str(selected.get("arm") or "") == arm
        and manifest.get("status") == "planning_success"
    )


def _static_validation_paths(
    variant_root: Path, scene_revision: str | None
) -> list[Path]:
    if scene_revision is None:
        return []
    return sorted(
        {
            path.resolve()
            for path in variant_root.rglob("static_validation.json")
            if path.is_file()
            and (_read_json_object(path) or {}).get("scene_revision")
            == scene_revision
        },
        key=str,
    )


def _scene_layout_paths(
    compiled_task_path: Path,
    variant_root: Path,
    scene_revision: str | None,
) -> list[Path]:
    if scene_revision in {None, "source"}:
        return []
    document = _compiled_task_document(compiled_task_path)
    task = _single_task(document)
    arena_value = task.get("arena_file") if task else None
    if not isinstance(arena_value, str) or not arena_value:
        return []
    arena_path = Path(arena_value).expanduser()
    if not arena_path.is_absolute():
        arena_path = compiled_task_path.parent / arena_path
    arena_path = arena_path.resolve()
    paths = [
        arena_path.parent / "simbox_task.yaml",
        arena_path.parent / "simbox_arena.yaml",
        arena_path.parent / "scene_mutations.json",
    ]
    return paths if all(_is_within(path, variant_root) for path in paths) else []


def _verify_scene_layout_artifacts(
    entry: ArtifactIndexEntry,
    compiled_task_path: Path,
    scene_revision: str | None,
    variant_root: Path,
) -> None:
    if scene_revision == "source":
        if entry.present:
            raise ValueError("source scene must not claim derived scene artifacts")
        return
    if scene_revision is None:
        raise ValueError("compiled task scene revision is missing")
    expected_paths = _scene_layout_paths(
        compiled_task_path, variant_root, scene_revision
    )
    if len(expected_paths) != 3 or {
        Path(member.path).resolve() for member in entry.members
    } != {path.resolve() for path in expected_paths}:
        raise ValueError("derived scene artifact set is incomplete")

    derived_task = _compiled_task_document(expected_paths[0])
    task = _single_task(derived_task)
    metadata = task.get("metadata") if task else None
    layout = metadata.get("agent_scene_layout") if isinstance(metadata, dict) else None
    arena_value = task.get("arena_file") if task else None
    mutation = _read_json_object(expected_paths[2])
    if not isinstance(layout, dict) or mutation is None:
        raise ValueError("derived scene metadata is invalid")
    if (
        layout.get("scene_revision") != scene_revision
        or mutation.get("scene_revision") != scene_revision
        or Path(str(arena_value)).expanduser().resolve() != expected_paths[1].resolve()
    ):
        raise ValueError("derived scene revision or arena binding disagrees")
    revision_payload = {
        "source_task_hash": layout.get("source_task_hash"),
        "source_arena_hash": layout.get("source_arena_hash"),
        "mutations": layout.get("mutations"),
    }
    if mutation != {**revision_payload, "scene_revision": scene_revision}:
        raise ValueError("derived scene mutation record disagrees with task metadata")
    computed_revision = hashlib.sha256(
        json.dumps(revision_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    if computed_revision != scene_revision:
        raise ValueError("derived scene revision digest is invalid")


def _probe_artifacts(
    workspace_manifests: Iterable[_SelectedWorkspace],
) -> tuple[list[Path], list[Path], list[Path]]:
    pick: set[Path] = set()
    place: set[Path] = set()
    spawn_settle: set[Path] = set()
    for selected_workspace in workspace_manifests:
        manifest_path = selected_workspace.path
        manifest = _read_json_object(manifest_path)
        if manifest is None:
            continue
        explicit = manifest.get("planning_probe_artifacts")
        if isinstance(explicit, dict):
            _add_selected_probe(
                pick, explicit.get("pick"), manifest_path.parent, selected_workspace
            )
            _add_selected_probe(
                place, explicit.get("place"), manifest_path.parent, selected_workspace
            )
            _add_selected_probe(
                spawn_settle,
                explicit.get("spawn_settle"),
                manifest_path.parent,
                selected_workspace,
            )
        for row in manifest.get("curobo_results") or []:
            if not isinstance(row, dict):
                continue
            for arm_result in (row.get("arms") or {}).values():
                if isinstance(arm_result, dict):
                    _add_selected_probe(
                        pick,
                        arm_result.get("artifact"),
                        manifest_path.parent,
                        selected_workspace,
                    )
        for row in manifest.get("place_probe_results") or []:
            if isinstance(row, dict):
                _add_selected_probe(
                    place, row.get("artifact"), manifest_path.parent, selected_workspace
                )
    return (
        sorted(pick, key=str),
        sorted(place, key=str),
        sorted(spawn_settle, key=str),
    )


def _add_selected_probe(
    paths: set[Path],
    raw_path: Any,
    base_dir: Path,
    selected: _SelectedWorkspace,
) -> None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    value = _read_json_object(path)
    if not path.is_file() or value is None:
        return
    if (
        str(value.get("candidate_id") or "") != selected.candidate_id
        or str(value.get("arm") or "") != selected.arm
        or value.get("seed") != selected.seed
    ):
        return
    paths.add(path)


def _evaluation_requires_data(path: Path) -> bool:
    evaluation = _read_json_object(path)
    data_integrity = evaluation.get("data_integrity") if evaluation else None
    return bool(
        isinstance(data_integrity, dict) and data_integrity.get("required") is True
    )


def _compiled_workspace_identity(path: Path) -> tuple[str | None, str | None]:
    document = _compiled_task_document(path)
    task = _single_task(document)
    if task is None:
        return None, None
    metadata = task.get("metadata")
    agent_plan = metadata.get("agent_plan") if isinstance(metadata, dict) else None
    if not isinstance(agent_plan, dict):
        return None, None
    candidate_id = agent_plan.get("workspace_candidate_id")
    scene_revision = agent_plan.get("scene_revision")
    return (
        candidate_id if isinstance(candidate_id, str) and candidate_id else None,
        scene_revision if isinstance(scene_revision, str) and scene_revision else None,
    )


def _compiled_task_signature(path: Path) -> str | None:
    document = _compiled_task_document(path)
    if document is None:
        return None
    try:
        return compute_variant_signature(document)
    except ValueError:
        return None


def _compiled_task_document(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return document if isinstance(document, dict) else None


def _single_task(document: dict[str, Any] | None) -> dict[str, Any] | None:
    tasks = document.get("tasks") if document else None
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        return None
    return tasks[0]


def _workspace_selection_candidate_id(path: Path) -> str | None:
    selection = _read_json_object(path)
    candidate = selection.get("candidate") if selection else None
    candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
    return candidate_id if isinstance(candidate_id, str) and candidate_id else None


def _screenshot_paths(attempt_dir: Path, episode_dir: Path | None) -> list[Path]:
    roots = [attempt_dir / "screenshots"]
    if episode_dir is not None:
        roots.append(episode_dir / "screenshots")
    return _existing_paths(roots)


def _video_paths(episode_dir: Path | None) -> list[Path]:
    if episode_dir is None:
        return []
    return sorted(
        (
            path.resolve()
            for path in episode_dir.glob("images.rgb.*/demo.mp4")
            if path.is_file()
        ),
        key=str,
    )


def _data_paths(attempt_dir: Path, episode_dir: Path | None) -> list[Path]:
    if episode_dir is not None:
        return [episode_dir]
    return _existing_paths([attempt_dir / "data"])


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file()
            or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))
        },
        key=str,
    )


def _index_paths(
    raw_paths: Iterable[Path], required: bool, missing_code: str
) -> ArtifactIndexEntry:
    paths = _existing_paths(raw_paths)
    if not paths:
        return ArtifactIndexEntry(
            required=required,
            present=False,
            kind="missing",
            failure_code=missing_code if required else None,
        )
    members = [_member(path) for path in paths]
    if len(members) == 1:
        path = paths[0]
        member = members[0]
        return ArtifactIndexEntry(
            required=required,
            present=True,
            path=member.path,
            sha256=member.sha256,
            kind="directory" if path.is_dir() else "file",
            size_bytes=member.size_bytes,
            members=members,
        )
    common_root = Path(os.path.commonpath([str(path) for path in paths]))
    if common_root.is_file():
        common_root = common_root.parent
    digest = hashlib.sha256()
    for member in members:
        member_path = Path(member.path)
        try:
            relative = member_path.relative_to(common_root).as_posix()
        except ValueError:
            relative = member.path
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member.sha256.encode("ascii"))
        digest.update(b"\0")
    return ArtifactIndexEntry(
        required=required,
        present=True,
        path=str(common_root),
        sha256=digest.hexdigest(),
        kind="collection",
        size_bytes=sum(member.size_bytes for member in members),
        members=members,
    )


def _member(path: Path) -> ArtifactMember:
    if path.is_file():
        return ArtifactMember(
            path=str(path),
            sha256=_file_sha256(path),
            size_bytes=path.stat().st_size,
        )
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=str)
    digest = hashlib.sha256()
    size_bytes = 0
    for file_path in files:
        file_hash = _file_sha256(file_path)
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        size_bytes += file_path.stat().st_size
    return ArtifactMember(
        path=str(path),
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def _verify_artifact_entry(
    name: str, entry: ArtifactIndexEntry, variant_root: Path
) -> None:
    member_paths: list[Path] = []
    for recorded in entry.members:
        member_path = Path(recorded.path).expanduser().resolve()
        if str(member_path) != recorded.path:
            raise ValueError(f"artifact member path is not canonical for {name}")
        if not _is_within(member_path, variant_root):
            raise ValueError(f"artifact member is outside variant_root for {name}")
        if not member_path.exists():
            raise ValueError(f"artifact member does not exist for {name}: {member_path}")
        actual = _member(member_path)
        if actual != recorded:
            raise ValueError(f"artifact member size or SHA-256 changed for {name}")
        member_paths.append(member_path)

    actual_entry = _index_paths(
        member_paths, entry.required, MISSING_ARTIFACT_CODES[name]
    )
    if actual_entry != entry:
        raise ValueError(f"artifact index entry changed for {name}")


def _require_single_file(
    entry: ArtifactIndexEntry, expected_path: Path, name: str
) -> None:
    expected_path = expected_path.expanduser().resolve()
    if (
        entry.kind != "file"
        or entry.path != str(expected_path)
        or len(entry.members) != 1
        or entry.members[0].path != str(expected_path)
    ):
        raise ValueError(f"artifact {name} does not belong to its canonical attempt")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_sha256(value: str | None) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
