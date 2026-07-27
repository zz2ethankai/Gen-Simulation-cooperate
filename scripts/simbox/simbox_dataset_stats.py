#!/usr/bin/env python3
"""Summarize SimBox LMDB episode datasets.

An episode is considered trainable only when both `meta_info.pkl` and
`lmdb/data.mdb` exist in the same episode directory. Preview videos are used for
duration accounting, not as the source of truth for trainable payload presence.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class VideoInfo:
    path: Path
    duration_sec: float | None
    frame_count: int | None
    fps: float | None
    size_bytes: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Dataset root to scan")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for episode_stats.csv, dataset_stats.json, and dataset_stats.md",
    )
    parser.add_argument("--fallback-video-fps", type=float, default=15.0)
    parser.add_argument("--action-fps", type=float, default=30.0)
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--title", default="SimBox Dataset Stats")
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def run_ffprobe(video_path: Path) -> VideoInfo:
    size_bytes = video_path.stat().st_size if video_path.exists() else 0
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return VideoInfo(video_path, None, None, None, size_bytes, "ffprobe_not_found")

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,nb_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        streams = json.loads(output).get("streams") or []
        stream = streams[0] if streams else {}
    except Exception as exc:  # noqa: BLE001 - report probe failure, keep scanning.
        return VideoInfo(video_path, None, None, None, size_bytes, f"{type(exc).__name__}: {exc}")

    duration = None
    try:
        if stream.get("duration") not in (None, "N/A"):
            duration = float(stream["duration"])
    except ValueError:
        duration = None

    frame_count = safe_int(stream.get("nb_frames"))
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    if duration is None and frame_count is not None and fps:
        duration = frame_count / fps
    if frame_count is None and duration is not None and fps:
        frame_count = int(round(duration * fps))

    return VideoInfo(video_path, duration, frame_count, fps, size_bytes)


def directory_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict):
        raise TypeError(f"meta_info.pkl did not contain a dict: {meta_path}")
    return meta


def image_step_counts(meta: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    image_valid = meta.get("image_valid_step_ids")
    if isinstance(image_valid, dict):
        for key, value in image_valid.items():
            if isinstance(value, (list, tuple)):
                counts[str(key)] = len(value)
    keys = meta.get("keys")
    if isinstance(keys, dict):
        for key, value in keys.items():
            key_str = str(key)
            if key_str.startswith("images.") and isinstance(value, (list, tuple)):
                counts.setdefault(key_str, len(value))
    return counts


def discover_candidates(root: Path) -> tuple[dict[Path, dict[str, bool]], int]:
    candidates: dict[Path, dict[str, bool]] = {}
    for meta_path in root.rglob("meta_info.pkl"):
        candidates.setdefault(meta_path.parent, {"meta": False, "lmdb": False})["meta"] = True
    for data_path in root.rglob("data.mdb"):
        if data_path.parent.name != "lmdb":
            continue
        candidates.setdefault(data_path.parent.parent, {"meta": False, "lmdb": False})["lmdb"] = True

    video_only_dirs = {
        video_path.parent.parent
        for video_path in root.rglob("demo.mp4")
        if video_path.parent.name.startswith("images.")
    }
    video_only_without_payload = len([p for p in video_only_dirs if p not in candidates])
    return candidates, video_only_without_payload


def summarize_episode(
    episode_dir: Path,
    root: Path,
    meta: dict[str, Any],
    fallback_video_fps: float,
    action_fps: float,
) -> dict[str, Any]:
    num_steps = safe_int(meta.get("num_steps")) or 0
    videos = sorted(episode_dir.glob("images.rgb.*/demo.mp4"))
    video_infos = [run_ffprobe(path) for path in videos]
    durations = [info.duration_sec for info in video_infos if info.duration_sec is not None and info.duration_sec > 0]
    frame_counts = [info.frame_count for info in video_infos if info.frame_count is not None]
    video_errors = [info.error for info in video_infos if info.error]

    if durations:
        trajectory_duration_sec = max(durations)
        duration_source = "video_max_duration"
    elif num_steps > 0 and fallback_video_fps > 0:
        trajectory_duration_sec = num_steps / fallback_video_fps
        duration_source = "num_steps_fallback"
    else:
        trajectory_duration_sec = 0.0
        duration_source = "missing"

    action_duration_sec = num_steps / action_fps if action_fps > 0 else 0.0
    keys = meta.get("keys") if isinstance(meta.get("keys"), dict) else {}
    action_keys = keys.get("action_data", []) if isinstance(keys, dict) else []
    proprio_keys = keys.get("proprio_data", []) if isinstance(keys, dict) else []
    img_counts = image_step_counts(meta)

    lmdb_dir = episode_dir / "lmdb"
    lmdb_bytes = directory_size(lmdb_dir) if lmdb_dir.exists() else 0
    video_bytes = sum(info.size_bytes for info in video_infos)

    is_name_prefixed_fail_episode = episode_dir.name.startswith("fail")
    return {
        "episode_path": rel(episode_dir, root),
        "episode_name": episode_dir.name,
        "is_name_prefixed_fail_episode": is_name_prefixed_fail_episode,
        "is_fail_episode": is_name_prefixed_fail_episode,
        "num_steps": num_steps,
        "action_duration_sec": round(action_duration_sec, 6),
        "trajectory_duration_sec": round(trajectory_duration_sec, 6),
        "duration_source": duration_source,
        "video_stream_count": len(video_infos),
        "video_total_sec_all_cameras": round(sum(durations), 6),
        "video_max_frame_count": max(frame_counts) if frame_counts else 0,
        "video_probe_error_count": len(video_errors),
        "video_bytes": video_bytes,
        "lmdb_bytes": lmdb_bytes,
        "episode_bytes": directory_size(episode_dir),
        "action_key_count": len(action_keys) if isinstance(action_keys, list) else 0,
        "proprio_key_count": len(proprio_keys) if isinstance(proprio_keys, list) else 0,
        "image_key_count": len(img_counts),
        "image_step_counts": json.dumps(img_counts, ensure_ascii=False, sort_keys=True),
        "language_instruction": str(meta.get("language_instruction", "")),
    }


def numeric(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            pass
    return result


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_path",
        "episode_name",
        "is_name_prefixed_fail_episode",
        "is_fail_episode",
        "num_steps",
        "trajectory_duration_sec",
        "action_duration_sec",
        "duration_source",
        "video_stream_count",
        "video_total_sec_all_cameras",
        "video_max_frame_count",
        "video_probe_error_count",
        "video_bytes",
        "lmdb_bytes",
        "episode_bytes",
        "action_key_count",
        "proprio_key_count",
        "image_key_count",
        "image_step_counts",
        "language_instruction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Valid episodes: `{summary['valid_episodes']}`",
        f"- Total trajectory duration: `{summary['total_trajectory_duration_sec']:.3f}` sec (`{summary['total_trajectory_duration_hours']:.6f}` hours)",
        f"- Total action steps: `{summary['total_action_steps']}`",
        f"- Total action duration: `{summary['total_action_duration_sec']:.3f}` sec",
        f"- Name-prefixed fail episode directories: `{summary['name_prefixed_fail_episode_count']}`",
        f"- Missing video episodes: `{summary['episodes_missing_video']}`",
        f"- Skipped incomplete candidates: `{summary['skipped_incomplete_candidates']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average steps | {summary['avg_steps']:.3f} |",
        f"| Median steps | {summary['median_steps']:.3f} |",
        f"| Average trajectory sec | {summary['avg_trajectory_duration_sec']:.3f} |",
        f"| Median trajectory sec | {summary['median_trajectory_duration_sec']:.3f} |",
        f"| Video streams | {summary['total_video_streams']} |",
        f"| All-camera video seconds | {summary['video_total_sec_all_cameras']:.3f} |",
        f"| Episode bytes | {summary['total_episode_bytes']} |",
        f"| LMDB bytes | {summary['total_lmdb_bytes']} |",
        f"| Video bytes | {summary['total_video_bytes']} |",
        "",
        "Files:",
        f"- Per-episode CSV: `{summary['episode_csv']}`",
        f"- Summary JSON: `{summary['summary_json']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Error: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    candidates, video_only_without_payload = discover_candidates(root)
    rows: list[dict[str, Any]] = []
    load_errors: list[str] = []
    skipped_missing_meta = 0
    skipped_missing_lmdb = 0

    for episode_dir, state in sorted(candidates.items(), key=lambda item: item[0].as_posix()):
        has_meta = state.get("meta", False)
        has_lmdb = state.get("lmdb", False)
        if not has_meta:
            skipped_missing_meta += 1
            continue
        if not has_lmdb:
            skipped_missing_lmdb += 1
            continue
        try:
            meta = load_meta(episode_dir / "meta_info.pkl")
            rows.append(
                summarize_episode(
                    episode_dir=episode_dir,
                    root=root,
                    meta=meta,
                    fallback_video_fps=args.fallback_video_fps,
                    action_fps=args.action_fps,
                )
            )
        except Exception as exc:  # noqa: BLE001 - continue to report remaining episodes.
            load_errors.append(f"{rel(episode_dir, root)}: {type(exc).__name__}: {exc}")

    steps = numeric([row["num_steps"] for row in rows])
    trajectory_durations = numeric([row["trajectory_duration_sec"] for row in rows])
    action_durations = numeric([row["action_duration_sec"] for row in rows])

    csv_path = output_dir / "episode_stats.csv"
    json_path = output_dir / "dataset_stats.json"
    md_path = output_dir / "dataset_stats.md"
    write_csv(csv_path, rows)

    summary = {
        "dataset_root": root.as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_episode_dirs": len(candidates),
        "valid_episodes": len(rows),
        "valid_episode_count": len(rows),
        "name_prefixed_fail_episode_count": sum(1 for row in rows if row["is_name_prefixed_fail_episode"]),
        "name_prefixed_success_like_episode_count": sum(
            1 for row in rows if not row["is_name_prefixed_fail_episode"]
        ),
        "fail_episode_count": sum(1 for row in rows if row["is_name_prefixed_fail_episode"]),
        "success_like_episode_count": sum(1 for row in rows if not row["is_name_prefixed_fail_episode"]),
        "fail_episode_count_is_name_based": True,
        "skipped_missing_meta": skipped_missing_meta,
        "skipped_missing_lmdb": skipped_missing_lmdb,
        "skipped_load_errors": len(load_errors),
        "skipped_incomplete_candidates": skipped_missing_meta + skipped_missing_lmdb + len(load_errors),
        "video_only_dirs_without_payload": video_only_without_payload,
        "episodes_missing_video": sum(1 for row in rows if int(row["video_stream_count"]) == 0),
        "video_probe_error_count": sum(int(row["video_probe_error_count"]) for row in rows),
        "total_action_steps": int(sum(steps)),
        "avg_steps": mean(steps),
        "median_steps": median(steps),
        "total_trajectory_duration_sec": sum(trajectory_durations),
        "total_trajectory_sec": sum(trajectory_durations),
        "total_trajectory_duration_hours": sum(trajectory_durations) / 3600.0,
        "avg_trajectory_duration_sec": mean(trajectory_durations),
        "median_trajectory_duration_sec": median(trajectory_durations),
        "total_action_duration_sec": sum(action_durations),
        "total_video_streams": sum(int(row["video_stream_count"]) for row in rows),
        "video_total_sec_all_cameras": sum(float(row["video_total_sec_all_cameras"]) for row in rows),
        "total_episode_bytes": sum(int(row["episode_bytes"]) for row in rows),
        "total_lmdb_bytes": sum(int(row["lmdb_bytes"]) for row in rows),
        "total_video_bytes": sum(int(row["video_bytes"]) for row in rows),
        "fallback_video_fps": args.fallback_video_fps,
        "action_fps": args.action_fps,
        "load_errors": load_errors[:100],
        "episode_csv": csv_path.as_posix(),
        "summary_json": json_path.as_posix(),
        "summary_markdown": md_path.as_posix(),
    }

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, args.title, summary)

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.fail_on_empty and not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
