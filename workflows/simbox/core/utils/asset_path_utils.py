"""Shared path resolution for task-level and per-object SimBox assets."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping


_TEXTURE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tga", ".tif", ".tiff"}


def resolve_asset_root(default_asset_root: str, cfg: Mapping) -> str:
    """Return an absolute asset root, honoring an optional object override."""
    value = cfg.get("asset_root", default_asset_root)
    return os.path.abspath(os.path.expanduser(str(value)))


def resolve_asset_path(default_asset_root: str, cfg: Mapping) -> str:
    """Resolve cfg.path against its object root without requiring Isaac imports."""
    path = Path(os.path.expanduser(str(cfg["path"])))
    if path.is_absolute():
        return str(path.resolve())
    return str((Path(resolve_asset_root(default_asset_root, cfg)) / path).resolve())


def _existing_texture_files(candidates: list[Path]) -> list[str]:
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in _TEXTURE_SUFFIXES:
            return [str(candidate.resolve())]
        if candidate.is_dir():
            files = sorted(
                str(path.resolve())
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in _TEXTURE_SUFFIXES
            )
            if files:
                return files
    return []


def resolve_texture_paths(
    asset_root: str,
    texture_name: str | None,
    *,
    texture_file: str | None = None,
) -> list[str]:
    """Resolve explicit textures and libraries from legacy and InterData layouts."""
    root = Path(os.path.expanduser(str(asset_root)))
    roots = [root, *root.parents]

    if texture_file:
        configured_file = Path(os.path.expanduser(str(texture_file)))
        explicit_candidates = (
            [configured_file]
            if configured_file.is_absolute()
            else [candidate_root / configured_file for candidate_root in roots]
        )
        return _existing_texture_files(explicit_candidates)

    if not texture_name:
        return []
    configured = Path(os.path.expanduser(str(texture_name)))
    if configured.is_absolute():
        library_candidates = [configured]
    else:
        library_candidates = [candidate_root / configured for candidate_root in roots]
        library_candidates.extend(
            candidate_root / "texture_libs" / configured for candidate_root in roots
        )
        library_candidates.extend(
            candidate_root / "interdata" / "texture_libs" / configured
            for candidate_root in roots
        )
    return _existing_texture_files(library_candidates)


def select_texture_path(asset_root: str, cfg: Mapping[str, Any]) -> str:
    """Select one configured texture or fail with actionable path context."""
    texture_name = cfg.get("texture_lib")
    texture_file = cfg.get("texture_file")
    texture_paths = resolve_texture_paths(
        asset_root,
        str(texture_name) if texture_name else None,
        texture_file=str(texture_file) if texture_file else None,
    )
    if not texture_paths:
        raise FileNotFoundError(
            "No texture files found: "
            f"asset_root={os.path.abspath(os.path.expanduser(str(asset_root)))!r}, "
            f"texture_file={texture_file!r}, texture_lib={texture_name!r}"
        )
    if texture_file:
        return texture_paths[0]
    if bool(cfg.get("apply_randomization", False)):
        return random.choice(texture_paths)
    texture_id = int(cfg.get("texture_id", 0))
    if texture_id < 0 or texture_id >= len(texture_paths):
        raise IndexError(
            f"texture_id {texture_id} is outside [0, {len(texture_paths) - 1}] "
            f"for texture_lib={texture_name!r}"
        )
    return texture_paths[texture_id]
