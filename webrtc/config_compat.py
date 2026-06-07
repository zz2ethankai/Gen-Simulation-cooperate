"""Compatibility helpers for loading existing scene YAMLs without owning them.

This module intentionally lives outside the main pipeline.  It may call the
repository's existing conversion helpers so the viewer follows the same config
rules, but it never writes converted YAML back to the source tree.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _import_converter():
    scripts_dir = REPO_ROOT / "scripts"
    scripts_dir_str = str(scripts_dir)
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)
    import convert_download_scene_configs as converter  # pylint: disable=import-outside-toplevel

    return converter


def _scene_dir_from_task(task_path: Path) -> Path:
    name = task_path.name
    if name not in {"task.yaml", "simbox_task.yaml"}:
        return task_path.parent
    return task_path.parent


def _load_simbox_task(task_path: Path) -> dict[str, Any]:
    payload = _load_yaml(task_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{task_path} must contain a mapping")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        raise ValueError(f"{task_path} is not a supported SimBox task YAML")
    task = tasks[0]
    arena_file = task.get("arena_file")
    if not isinstance(arena_file, str) or not arena_file:
        raise ValueError(f"{task_path} missing tasks[0].arena_file")
    arena_path = _repo_path(Path(arena_file))
    arena_payload = _load_yaml(arena_path)
    if not isinstance(arena_payload, dict):
        raise ValueError(f"{arena_path} must contain a mapping")
    return {
        "source": "simbox",
        "scene_dir": task_path.parent,
        "task_path": task_path,
        "arena_path": arena_path,
        "task": task,
        "arena": arena_payload,
        "asset_root": _repo_path(Path(str(task.get("asset_root", task_path.parent)))),
    }


def _convert_download_task_in_memory(task_path: Path, *, include_robot: bool) -> dict[str, Any]:
    converter = _import_converter()
    scene_dir = _scene_dir_from_task(task_path)
    source_dir = converter.select_source_yaml_dir(scene_dir)
    arena_path = source_dir / "arena.yaml"
    source_task_path = source_dir / "task.yaml"
    if not arena_path.exists() or not source_task_path.exists():
        raise FileNotFoundError(f"{source_dir} must contain arena.yaml and task.yaml")

    arena_cfg = converter.load_yaml(arena_path)
    task_cfg = converter.load_yaml(source_task_path)
    if not isinstance(arena_cfg, dict) or not isinstance(task_cfg, dict):
        raise ValueError(f"{source_dir} arena/task YAML must be mappings")
    task_cfg = converter.task_with_robot_pose_fallback(scene_dir, task_cfg)

    asset_root_cfg = scene_dir.relative_to(REPO_ROOT) if scene_dir.is_absolute() else scene_dir
    asset_root_abs = converter.repo_path(asset_root_cfg)
    with tempfile.TemporaryDirectory(prefix="interndata_webrtc_") as tmp:
        tmp_arena = Path(tmp) / "arena.yaml"
        arena_payload = converter.build_arena_payload(arena_cfg, source_dir, asset_root_abs)
        task_payload = converter.build_task_payload(
            source_dir,
            task_cfg,
            arena_payload,
            tmp_arena,
            asset_root_cfg=asset_root_cfg,
            asset_root_abs=asset_root_abs,
            object_mode="geometry",
            default_prim_path_child="Aligned",
            include_robot=include_robot,
            robot_name=converter.DEFAULT_ROBOT_NAME,
            max_episode_length=1000,
        )

    task = task_payload["tasks"][0]
    return {
        "source": "download",
        "scene_dir": scene_dir,
        "task_path": task_path,
        "arena_path": arena_path,
        "task": task,
        "arena": arena_payload,
        "asset_root": asset_root_abs,
    }


def load_scene_config(task_path: str | Path, *, include_robot: bool = False) -> SimpleNamespace:
    path = _repo_path(Path(task_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    payload = _load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")

    if isinstance(payload.get("tasks"), list):
        data = _load_simbox_task(path)
    elif payload.get("format") == "task" or "task_objects" in payload:
        data = _convert_download_task_in_memory(path, include_robot=include_robot)
    else:
        raise ValueError(f"{path} is not a supported task YAML")

    return SimpleNamespace(**data)
