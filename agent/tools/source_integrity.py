"""Deterministic provenance for every file that defines a source scene."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from workflows.simbox.core.robots.profile import (
    RobotModelProfile,
    load_robot_profile,
    resolve_robot_asset_path,
)


SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONFIG_REFERENCE_KEYS = {"arena_file", "camera_file", "robot_config_file"}
_ASSET_REFERENCE_KEYS = {"path", "obj_info_path"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SourceIntegrityError(ValueError):
    """A source snapshot cannot be built or validated exactly."""


@dataclass(frozen=True)
class SourceMember:
    path: str
    role: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceIntegrityResult:
    source_unchanged: bool
    identity_consistent: bool
    source_hash: str | None
    errors: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usd_dependency_paths(path: Path) -> tuple[Path, ...]:
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        return ()
    try:
        from pxr import UsdUtils
    except ImportError as exc:
        raise SourceIntegrityError(
            "USD dependency enumeration requires the pxr runtime"
        ) from exc
    try:
        layers, assets, _ = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise SourceIntegrityError(
            f"cannot enumerate USD dependencies for {path}: {exc}"
        ) from exc
    values = [
        Path(str(getattr(layer, "realPath", "")))
        for layer in layers
        if str(getattr(layer, "realPath", ""))
    ]
    values.extend(Path(str(asset)) for asset in assets)
    root = path.resolve()
    return tuple(
        sorted(
            {
                value.resolve()
                for value in values
                if value.is_file() and value.resolve() != root
            },
            key=str,
        )
    )


def _source_file_members(path: Path, role: str) -> list[SourceMember]:
    members = [SourceMember(str(path), role, _file_sha256(path))]
    members.extend(
        SourceMember(
            str(dependency),
            f"{role}.usd_dependency[{index:03d}]",
            _file_sha256(dependency),
        )
        for index, dependency in enumerate(_usd_dependency_paths(path))
    )
    return members


def canonical_source_hash(members: Sequence[SourceMember]) -> str:
    if not members:
        raise SourceIntegrityError("source snapshot must contain at least one member")
    ordered = sorted(members, key=lambda member: (member.role, member.path))
    roles = [member.role for member in ordered]
    if len(roles) != len(set(roles)):
        raise SourceIntegrityError("source snapshot member roles must be unique")
    for member in ordered:
        if not member.role or not Path(member.path).is_absolute():
            raise SourceIntegrityError("source snapshot members require a role and absolute path")
        if _DIGEST_PATTERN.fullmatch(member.sha256) is None:
            raise SourceIntegrityError(
                f"source snapshot member has an invalid SHA-256 digest: {member.role}"
            )
    payload = json.dumps(
        [member.to_dict() for member in ordered],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _yaml_document(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SourceIntegrityError(f"cannot read source document {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SourceIntegrityError(f"source document root must be a mapping: {path}")
    return dict(value)


def _first_task(document: Mapping[str, Any]) -> dict[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise SourceIntegrityError("source task must contain exactly one tasks[0] mapping")
    return dict(tasks[0])


def _resolve_root(value: Any, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.absolute() if path.is_absolute() else (repo_root / path).absolute()


def _resolve_config_reference(
    value: str,
    *,
    document_path: Path,
    repo_root: Path,
) -> Path | None:
    path = Path(value).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (repo_root / path, document_path.parent / path)
    )
    return next((candidate.absolute() for candidate in candidates if candidate.is_file()), None)


def _resolve_asset_reference(value: str, asset_root: Path) -> Path | None:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else asset_root / path
    return candidate.absolute() if candidate.is_file() else None


def _role_component(value: Mapping[str, Any], index: int) -> str:
    name = str(value.get("name") or "").strip()
    return f"[{name}]" if name else f"[{index}]"


def _entity_references(
    values: Any,
    *,
    role: str,
    document_path: Path,
    repo_root: Path,
    asset_root: Path,
) -> list[tuple[Path, str]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise SourceIntegrityError(f"{role} must be a list")
    result: list[tuple[Path, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise SourceIntegrityError(f"{role}[{index}] must be a mapping")
        item_role = f"{role}{_role_component(value, index)}"
        local_asset_root = asset_root
        if value.get("asset_root") not in {None, ""}:
            local_asset_root = _resolve_root(value["asset_root"], repo_root)
        for key in sorted(_ASSET_REFERENCE_KEYS):
            reference = value.get(key)
            if not isinstance(reference, str) or not reference.strip():
                continue
            path = _resolve_asset_reference(reference, local_asset_root)
            if path is not None:
                result.append((path, f"{item_role}.{key}"))
        for key in sorted(_CONFIG_REFERENCE_KEYS):
            reference = value.get(key)
            if not isinstance(reference, str) or not reference.strip():
                continue
            path = _resolve_config_reference(
                reference,
                document_path=document_path,
                repo_root=repo_root,
            )
            if path is not None:
                result.append((path, f"{item_role}.{key}"))
    return result


def _camera_references(
    values: Any,
    *,
    role: str,
    document_path: Path,
    repo_root: Path,
) -> list[tuple[Path, str]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise SourceIntegrityError(f"{role} must be a list")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise SourceIntegrityError(f"{role}[{index}] must be a mapping")
        reference = value.get("camera_file")
        if not isinstance(reference, str) or not reference.strip():
            continue
        path = _resolve_config_reference(
            reference,
            document_path=document_path,
            repo_root=repo_root,
        )
        if path is not None:
            result.append(
                (path, f"{role}{_role_component(value, index)}.camera_file")
            )
    return result


def _required_file_reference(
    value: str,
    *,
    owner: Path,
    repo_root: Path,
    label: str,
) -> Path:
    path = Path(value).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (repo_root / path, _PROJECT_ROOT / path, owner.parent / path)
    )
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise SourceIntegrityError(
            f"{label} does not exist: {value}; candidates={list(map(str, candidates))}"
        )
    return resolved


def _curobo_runtime_members(
    profile: RobotModelProfile,
    arm_id: str,
    curobo_path: Path,
) -> list[SourceMember]:
    document = _yaml_document(curobo_path)
    robot_cfg = document.get("robot_cfg")
    if not isinstance(robot_cfg, Mapping):
        raise SourceIntegrityError(f"CuRobo config is missing robot_cfg: {curobo_path}")
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, Mapping):
        raise SourceIntegrityError(f"CuRobo config is missing robot_cfg.kinematics: {curobo_path}")

    content_root = curobo_path.parent
    while content_root.name != "content" and content_root != content_root.parent:
        content_root = content_root.parent
    if content_root.name != "content":
        content_root = curobo_path.parent

    members = [
        SourceMember(
            str(curobo_path),
            f"robot_curobo_config:{profile.profile_id}:{arm_id}",
            _file_sha256(curobo_path),
        )
    ]
    references = (
        ("urdf_path", content_root / "assets", True),
        (
            "usd_path",
            content_root / "assets",
            bool(kinematics.get("use_usd_kinematics")),
        ),
        ("collision_spheres", content_root / "configs" / "robot", True),
    )
    for key, root, required in references:
        value = kinematics.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else root / path
        if not candidate.is_file():
            if required:
                raise SourceIntegrityError(
                    f"CuRobo {key} does not exist for {profile.profile_id}:{arm_id}: {candidate}"
                )
            continue
        resolved = candidate.resolve()
        members.append(
            SourceMember(
                str(resolved),
                f"robot_curobo_{key}:{profile.profile_id}:{arm_id}",
                _file_sha256(resolved),
            )
        )
    return members


def _robot_members(profile: RobotModelProfile, repo_root: Path) -> list[SourceMember]:
    profile_path = profile.source_path.resolve()
    asset_path = resolve_robot_asset_path(profile).absolute()
    profile_hash = _file_sha256(profile_path)
    asset_hash = _file_sha256(asset_path)
    members = [
        SourceMember(
            str(profile_path),
            f"robot_profile:{profile.profile_id}",
            profile_hash,
        ),
        SourceMember(
            str(asset_path),
            f"robot_canonical_asset:{profile.profile_id}",
            asset_hash,
        ),
        SourceMember(
            str(asset_path),
            f"robot_selected_asset:{profile.profile_id}",
            asset_hash,
        ),
    ]
    members.extend(
        _source_file_members(
            asset_path,
            f"robot_canonical_asset_dependencies:{profile.profile_id}",
        )[1:]
    )
    for arm_id, arm in sorted(profile.arms.items()):
        curobo_path = _required_file_reference(
            arm.curobo_file,
            owner=profile_path,
            repo_root=repo_root,
            label=f"arms.{arm_id}.curobo_file",
        )
        members.extend(_curobo_runtime_members(profile, arm_id, curobo_path))
    for index, camera in enumerate(profile.camera_rig):
        camera_path = _required_file_reference(
            camera.camera_file,
            owner=profile_path,
            repo_root=repo_root,
            label=f"camera_rig[{index}].camera_file",
        )
        members.append(
            SourceMember(
                str(camera_path),
                f"robot_camera_config:{profile.profile_id}:{camera.save_name}:{index}",
                _file_sha256(camera_path),
            )
        )
    extra_depth_file = profile.kinematics.get("extra_depth_file")
    if isinstance(extra_depth_file, str) and extra_depth_file:
        extra_depth_path = asset_path.with_name(extra_depth_file)
        if not extra_depth_path.is_file():
            raise SourceIntegrityError(
                f"robot extra_depth_file does not exist for {profile.profile_id}: "
                f"{extra_depth_path}"
            )
        members.append(
            SourceMember(
                str(extra_depth_path.resolve()),
                f"robot_kinematics_extra_depth:{profile.profile_id}",
                _file_sha256(extra_depth_path),
            )
        )
    return members


def build_source_snapshot(
    source_task: str | Path,
    source_arena: str | Path,
    robot_profile_paths: Iterable[str | Path],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    task_path = Path(source_task).expanduser().resolve()
    arena_path = Path(source_arena).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    task_document = _yaml_document(task_path)
    arena_document = _yaml_document(arena_path)
    task = _first_task(task_document)
    task_asset_root = _resolve_root(task.get("asset_root", root), root)

    members = [
        SourceMember(str(task_path), "source_task", _file_sha256(task_path)),
        SourceMember(str(arena_path), "source_arena", _file_sha256(arena_path)),
    ]
    references: list[tuple[Path, str]] = []
    for collection in ("objects", "distractors", "robots"):
        references.extend(
            _entity_references(
                task.get(collection),
                role=f"task.{collection}",
                document_path=task_path,
                repo_root=root,
                asset_root=task_asset_root,
            )
        )
    references.extend(
        _camera_references(
            task.get("cameras"),
            role="task.cameras",
            document_path=task_path,
            repo_root=root,
        )
    )
    references.extend(
        _entity_references(
            arena_document.get("fixtures"),
            role="arena.fixtures",
            document_path=arena_path,
            repo_root=root,
            asset_root=task_asset_root,
        )
    )
    references.extend(
        _camera_references(
            arena_document.get("cameras"),
            role="arena.cameras",
            document_path=arena_path,
            repo_root=root,
        )
    )
    for path, role in references:
        if path not in {task_path, arena_path}:
            members.extend(_source_file_members(path, role))
    loaded_profiles: dict[str, RobotModelProfile] = {}
    for profile_path in robot_profile_paths:
        resolved_profile_path = Path(profile_path).expanduser()
        if not resolved_profile_path.is_absolute():
            resolved_profile_path = root / resolved_profile_path
        profile = load_robot_profile(resolved_profile_path)
        existing = loaded_profiles.get(profile.profile_id)
        if existing is not None and existing.source_path != profile.source_path:
            raise SourceIntegrityError(
                f"profile_id {profile.profile_id!r} resolves to multiple profile files"
            )
        loaded_profiles[profile.profile_id] = profile
    for profile_id in sorted(loaded_profiles):
        members.extend(_robot_members(loaded_profiles[profile_id], root))
    ordered = sorted(members, key=lambda member: (member.role, member.path))
    source_hash = canonical_source_hash(ordered)
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "source_task": str(task_path),
        "source_arena": str(arena_path),
        "source_task_hash": _file_sha256(task_path),
        "source_arena_hash": _file_sha256(arena_path),
        "members": [member.to_dict() for member in ordered],
        "source_hash": source_hash,
    }


def write_source_snapshot(snapshot: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(snapshot), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _snapshot_members(snapshot: Mapping[str, Any]) -> list[SourceMember]:
    if snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise SourceIntegrityError("source snapshot schema_version is invalid")
    values = snapshot.get("members")
    if not isinstance(values, list) or not values:
        raise SourceIntegrityError("source snapshot members must be a non-empty list")
    members = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {"path", "role", "sha256"}:
            raise SourceIntegrityError(
                f"source snapshot members[{index}] must contain path, role, and sha256"
            )
        members.append(
            SourceMember(
                path=str(value["path"]),
                role=str(value["role"]),
                sha256=str(value["sha256"]),
            )
        )
    canonical_source_hash(members)
    by_role = {member.role: member for member in members}
    for role, path_key, hash_key in (
        ("source_task", "source_task", "source_task_hash"),
        ("source_arena", "source_arena", "source_arena_hash"),
    ):
        member = by_role.get(role)
        if member is None:
            raise SourceIntegrityError(f"source snapshot is missing {role}")
        if snapshot.get(path_key) != member.path or snapshot.get(hash_key) != member.sha256:
            raise SourceIntegrityError(f"source snapshot {role} projection is inconsistent")
    return members


def verify_source_snapshot(
    source_snapshot_path: str | Path | None,
    expected_source_hash: str,
) -> SourceIntegrityResult:
    if source_snapshot_path is None:
        return SourceIntegrityResult(False, False, None, ("source snapshot is missing",))
    path = Path(source_snapshot_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise SourceIntegrityError("source snapshot root must be a mapping")
        members = _snapshot_members(value)
        declared_hash = value.get("source_hash")
        aggregate_hash = canonical_source_hash(members)
        if declared_hash != aggregate_hash:
            raise SourceIntegrityError("source snapshot aggregate hash is invalid")
        for member in members:
            member_path = Path(member.path)
            if not member_path.is_file():
                raise SourceIntegrityError(
                    f"source member is missing: {member.role} ({member.path})"
                )
            if _file_sha256(member_path) != member.sha256:
                raise SourceIntegrityError(
                    f"source member hash changed: {member.role} ({member.path})"
                )
    except (OSError, json.JSONDecodeError, SourceIntegrityError) as exc:
        return SourceIntegrityResult(False, False, None, (str(exc),))
    if aggregate_hash != expected_source_hash:
        return SourceIntegrityResult(
            True,
            False,
            aggregate_hash,
            ("source snapshot does not match the execution identity",),
        )
    return SourceIntegrityResult(True, True, aggregate_hash, ())
