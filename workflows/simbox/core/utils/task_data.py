"""Compatibility normalization for SimBox runtime data metadata."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any


def normalize_runtime_data_config(
    task_cfg: MutableMapping[str, Any],
    task_cfg_path: str | Path,
) -> MutableMapping[str, Any]:
    """Populate runtime metadata missing from converted scene tasks."""

    data = task_cfg.get("data")
    if data is None:
        data = {}
        task_cfg["data"] = data
    if not hasattr(data, "get") or not hasattr(data, "__setitem__"):
        raise TypeError("task data must be a mapping")

    fallback_name = str(
        task_cfg.get("name")
        or task_cfg.get("task")
        or Path(task_cfg_path).stem
    ).strip()
    if not fallback_name:
        fallback_name = "simbox_task"

    if not str(data.get("task_dir") or "").strip():
        data["task_dir"] = fallback_name
    if not str(data.get("collect_info") or "").strip():
        data["collect_info"] = fallback_name
    if not str(data.get("version") or "").strip():
        data["version"] = "v1.0"
    if data.get("update") is None:
        data["update"] = True

    max_episode_length = data.get("max_episode_length")
    if max_episode_length is None:
        max_episode_length = task_cfg.get("max_episode_length")
    if max_episode_length is None:
        max_episode_length = 10_000
    if isinstance(max_episode_length, bool):
        raise TypeError("max_episode_length must be a positive integer")
    try:
        max_episode_length = int(max_episode_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_episode_length must be a positive integer") from exc
    if max_episode_length <= 0:
        raise ValueError("max_episode_length must be a positive integer")
    data["max_episode_length"] = max_episode_length
    if task_cfg.get("max_episode_length") is None:
        task_cfg["max_episode_length"] = max_episode_length
    return data
