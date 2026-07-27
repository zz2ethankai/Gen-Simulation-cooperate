"""Shared path resolution for task-level and per-object SimBox assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


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
