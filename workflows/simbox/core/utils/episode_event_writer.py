"""Best-effort SimBox episode event emission.

The parallel Docker launcher enables this by setting
``INTERNDATA_EPISODE_EVENT_PATH``. When the variable is absent, these helpers are
no-ops. Event write failures are intentionally non-fatal: generation data should
not be lost because observability failed.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _directory_size(path: Path) -> int:
    total = 0
    try:
        iterator = path.rglob("*")
    except OSError:
        return 0
    for file_path in iterator:
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            pass
    return total


def _load_meta(episode_dir: Path) -> dict[str, Any]:
    meta_path = episode_dir / "meta_info.pkl"
    try:
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _image_step_count(meta: dict[str, Any], image_key: str) -> int | None:
    image_valid = meta.get("image_valid_step_ids")
    if isinstance(image_valid, dict):
        value = image_valid.get(image_key)
        if isinstance(value, (list, tuple)):
            return len(value)
    keys = meta.get("keys")
    if isinstance(keys, dict):
        value = keys.get(image_key)
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _summarize_episode_dir(episode_dir: Path, *, video_fps: float, action_fps: float) -> dict[str, Any]:
    meta = _load_meta(episode_dir)
    num_steps = int(meta.get("num_steps") or 0)
    streams: list[dict[str, Any]] = []

    for video_path in sorted(episode_dir.glob("images.rgb.*/demo.mp4")):
        image_key = video_path.parent.name
        frame_count = _image_step_count(meta, image_key)
        duration_sec = (frame_count / video_fps) if frame_count is not None and video_fps > 0 else None
        try:
            size_bytes = video_path.stat().st_size
        except OSError:
            size_bytes = 0
        streams.append(
            {
                "name": image_key,
                "path": video_path.as_posix(),
                "frame_count": frame_count,
                "fps": video_fps,
                "duration_sec": duration_sec,
                "size_bytes": size_bytes,
            }
        )

    durations = [stream["duration_sec"] for stream in streams if stream["duration_sec"] is not None]
    if durations:
        trajectory_duration_sec = max(durations)
        duration_source = "hook_image_frame_count"
    elif num_steps > 0 and video_fps > 0:
        trajectory_duration_sec = num_steps / video_fps
        duration_source = "hook_num_steps_fallback"
    else:
        trajectory_duration_sec = 0.0
        duration_source = "missing"

    return {
        "episode_dir": episode_dir.as_posix(),
        "episode_name": episode_dir.name,
        "num_steps": num_steps,
        "action_fps": action_fps,
        "action_duration_sec": (num_steps / action_fps) if action_fps > 0 else 0.0,
        "video_fps": video_fps,
        "video_streams": streams,
        "video_stream_count": len(streams),
        "video_total_sec_all_cameras": sum(durations),
        "trajectory_duration_sec": trajectory_duration_sec,
        "duration_source": duration_source,
        "episode_bytes": _directory_size(episode_dir),
    }


def emit_episode_saved(
    *,
    status: str,
    episode_dirs: list[str | Path],
    num_steps: int,
    failure_reason: str | None = None,
    task_name: str | None = None,
    task_dir: str | None = None,
    collect_info: str | None = None,
) -> None:
    event_path_raw = os.environ.get("INTERNDATA_EPISODE_EVENT_PATH")
    if not event_path_raw:
        return

    event_path = Path(event_path_raw)
    video_fps = _env_float("INTERNDATA_VIDEO_FPS", 15.0)
    action_fps = _env_float("INTERNDATA_ACTION_FPS", 30.0)
    episode_paths = [Path(path) for path in episode_dirs]
    summaries = [
        _summarize_episode_dir(path, video_fps=video_fps, action_fps=action_fps)
        for path in episode_paths
    ]

    trajectory_durations = [item["trajectory_duration_sec"] for item in summaries]
    action_steps = max([item["num_steps"] for item in summaries] or [int(num_steps or 0)])
    record = {
        "event": "episode_saved",
        "event_id": uuid.uuid4().hex,
        "time": datetime.now(timezone.utc).isoformat(),
        "time_unix": time.time(),
        "run_id": os.environ.get("INTERNDATA_RUN_ID", ""),
        "job_id": os.environ.get("INTERNDATA_JOB_ID", ""),
        "worker": os.environ.get("INTERNDATA_WORKER", ""),
        "gpu": os.environ.get("INTERNDATA_GPU", ""),
        "task_path": os.environ.get("INTERNDATA_TASK_PATH", ""),
        "dataset_root": os.environ.get("INTERNDATA_DATASET_ROOT", ""),
        "seed": os.environ.get("INTERNDATA_RANDOM_SEED", ""),
        "status": status,
        "failure_reason": failure_reason or "",
        "finalized": True,
        "task_name": task_name or "",
        "task_dir": task_dir or "",
        "collect_info": collect_info or "",
        "episode_dirs": [path.as_posix() for path in episode_paths],
        "primary_episode_dir": episode_paths[0].as_posix() if episode_paths else "",
        "num_steps": action_steps,
        "action_fps": action_fps,
        "action_duration_sec": (action_steps / action_fps) if action_fps > 0 else 0.0,
        "video_fps": video_fps,
        "trajectory_duration_sec": max(trajectory_durations) if trajectory_durations else 0.0,
        "duration_source": "hook_episode_dirs",
        "video_stream_count": sum(item["video_stream_count"] for item in summaries),
        "episode_bytes": sum(item["episode_bytes"] for item in summaries),
        "episode_summaries": summaries,
    }

    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - observability must not break generation.
        print(f"[episode_event_writer] failed to write {event_path}: {type(exc).__name__}: {exc}")
