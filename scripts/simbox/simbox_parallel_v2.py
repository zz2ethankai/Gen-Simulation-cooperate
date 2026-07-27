#!/usr/bin/env python3
"""Config-driven SimBox Docker parallel generation launcher.

This module is intentionally host-side and lightweight. It does not import
Nimbus/Isaac modules; each concrete job starts a short-lived Docker container
that runs the normal DataEngine launcher inside Isaac Sim.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced as a config error.
    raise SystemExit(f"PyYAML is required for SimBox parallel v2: {exc}") from exc


STARTUP_HANG_EXIT_CODE = 124
TASK_TIMEOUT_EXIT_CODE = 124
FATAL_GUARD_EXIT_CODE = 137


FATAL_LOG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gpu_oom", re.compile(r"Out of GPU memory|ERROR_OUT_OF_DEVICE_MEMORY|cudaErrorMemoryAllocation|out of memory|Out of device memory|Unable to allocate|subAllocate\(\) failed", re.I)),
    ("gpu_crash", re.compile(r"GPU crash is detected|nv-gpudmp", re.I)),
    ("cuda_illegal_address", re.compile(r"cudaErrorIllegalAddress|CUDA error 700|illegal memory access", re.I)),
    ("segfault", re.compile(r"Segmentation fault|Fatal Python error", re.I)),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return safe or "item"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def rel_repo_path(path: str | Path, repo_root: Path) -> str:
    path_s = str(path)
    path_p = Path(path_s)
    if path_p.is_absolute():
        try:
            return path_p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path_p.as_posix()
    return path_s


def ensure_repo_visible_for_docker(path: str, repo_root: Path, label: str) -> None:
    path_p = Path(path)
    if not path_p.is_absolute():
        return
    try:
        path_p.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Docker backend cannot see {label} outside repo mount: {path}") from exc


def path_inside_repo(path: str | Path, repo_root: Path) -> bool:
    path_p = Path(path)
    if not path_p.is_absolute():
        return True
    try:
        path_p.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def resolve_host_path(path: str | Path, repo_root: Path) -> Path:
    path_p = Path(path)
    return path_p if path_p.is_absolute() else repo_root / path_p


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_cli_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def apply_cli_overrides(config: dict[str, Any], extras: list[str]) -> None:
    """Apply a small subset of OmegaConf-style dot overrides.

    Supported forms:
      --parallel.dry_run=true
      parallel.gpus=[0,1]
    """
    for raw in extras:
        item = raw[2:] if raw.startswith("--") else raw
        if "=" not in item:
            raise ValueError(f"Unsupported parallel v2 override '{raw}'. Use --path.to.key=value.")
        path_s, value_s = item.split("=", 1)
        keys = [key for key in path_s.split(".") if key]
        if not keys:
            raise ValueError(f"Invalid override path: {raw}")
        node: dict[str, Any] = config
        for key in keys[:-1]:
            next_node = node.setdefault(key, {})
            if not isinstance(next_node, dict):
                raise ValueError(f"Cannot override through non-mapping path: {path_s}")
            node = next_node
        node[keys[-1]] = parse_cli_value(value_s)


def get_path(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_value(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def float_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    value_s = str(value).strip()
    return value_s or None


def list_of_ints(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"{name} must be an int, comma-separated string, or list")


def read_de_config_output_dir(config_path: str, repo_root: Path) -> str:
    path = resolve_host_path(config_path, repo_root)
    data = load_yaml(path)
    config_name = data.get("name")
    output_dir = get_path(data, ("store_stage", "writer", "args", "output_dir"))
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f"Missing store_stage.writer.args.output_dir in {config_path}")
    result = output_dir.strip()
    if "${name}" in result:
        if not isinstance(config_name, str) or not config_name.strip():
            raise ValueError(f"{config_path} uses ${{name}} but has no top-level name")
        result = result.replace("${name}", config_name.strip())
    return result.rstrip("/")


def read_task_output_metadata(task_path: str, repo_root: Path) -> dict[str, Any]:
    data = load_yaml(resolve_host_path(task_path, repo_root))
    tasks = data.get("tasks")
    task = tasks[0] if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict) else {}
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    robots = task.get("robots") if isinstance(task.get("robots"), list) else []
    robot_names = [
        str(robot.get("name"))
        for robot in robots
        if isinstance(robot, dict) and str_or_none(robot.get("name"))
    ]
    return {
        "task_class": str(task.get("task") or "BananaBaseTask"),
        "task_dir": str(task_data.get("task_dir") or slugify_task(task_path)),
        "collect_info": str(task_data.get("collect_info") or ""),
        "robot_names": robot_names,
    }


def discover_task_paths(source: str, repo_root: Path) -> list[str]:
    source = rel_repo_path(source, repo_root)
    source_path = resolve_host_path(source, repo_root)
    if source.endswith((".yaml", ".yml")):
        if not source_path.is_file():
            raise FileNotFoundError(f"Task yaml not found: {source}")
        return [source]
    if source_path.is_dir():
        paths = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        return [path.relative_to(repo_root).as_posix() for path in paths]
    if source_path.is_file():
        tasks: list[str] = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tasks.append(rel_repo_path(stripped, repo_root))
        return tasks
    raise FileNotFoundError(f"Task source is not a yaml, list file, or directory: {source}")


def slugify_task(task_path: str) -> str:
    slug = task_path
    slug = slug[2:] if slug.startswith("./") else slug
    prefix = "workflows/simbox/core/configs/tasks/"
    if slug.startswith(prefix):
        slug = slug[len(prefix) :]
    slug = re.sub(r"\.ya?ml$", "", slug)
    slug = re.sub(r"[/ :]+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug).strip("-")
    digest = hashlib.sha1(task_path.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:80] or 'task'}_{digest}"


def stable_seed(run_id: str, job_id: str, shard_idx: int, seed_base: int | None, concrete_idx: int) -> int:
    if seed_base is not None:
        return int(seed_base) + concrete_idx
    digest = hashlib.sha1(f"{run_id}:{job_id}:{shard_idx}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2147483647


def split_counts(total: int, shard_count: int) -> list[int]:
    base = total // shard_count
    rem = total % shard_count
    return [base + (1 if idx < rem else 0) for idx in range(shard_count)]


def tail_lines(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def display_status(job: "ConcreteJob") -> str:
    if job.suspect_hang and job.status in {"starting", "running"}:
        return "suspect_hang"
    return job.status


def classify_failure(log_path: Path, docker_start_err: Path, status: str, exit_code: int | None) -> str:
    text = "\n".join(tail_lines(log_path, 160) + tail_lines(docker_start_err, 80)).lower()
    if status == "startup_hang":
        return "isaac_startup_hang"
    if status == "timeout":
        return "task_timeout"
    for reason, pattern in FATAL_LOG_PATTERNS:
        if pattern.search(text):
            return reason
    if docker_start_err.exists() and docker_start_err.stat().st_size > 0 and not log_path.exists():
        return "docker_start_failed"
    if "cuda out of memory" in text or "outofmemory" in text or "out of gpu memory" in text:
        return "cuda_oom"
    if "no such file" in text or "not found" in text or "missing" in text:
        return "asset_missing"
    if "configuration error" in text or "omegaconf" in text:
        return "config_error"
    if "traceback (most recent call last)" in text or "exception" in text:
        return "python_traceback"
    if exit_code == STARTUP_HANG_EXIT_CODE:
        return "task_timeout"
    return "unknown_failure"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file_path in path.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            pass
    return total


def human_bytes(size: int | float) -> str:
    value = float(size or 0)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}TiB"


def human_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    seconds_i = int(max(0, float(seconds)))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def percent(part: int | float, total: int | float) -> float:
    total_f = float(total or 0)
    return 0.0 if total_f <= 0 else (float(part or 0) / total_f) * 100.0


def compact_text(value: str, max_len: int) -> str:
    value = str(value or "")
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def compact_path(value: str, max_len: int = 46) -> str:
    value = str(value or "")
    if len(value) <= max_len:
        return value
    parts = value.split("/")
    if len(parts) >= 3:
        tail = "/".join(parts[-3:])
        if len(tail) + 2 <= max_len:
            return f".../{tail}"
    return "..." + value[-(max_len - 3) :]


@dataclass(frozen=True)
class ParallelSettings:
    backend: str
    gpus: list[int]
    workers_per_gpu: int
    run_id: str
    compose_file: str
    isaac_image: str
    task_timeout_sec: int
    stats_after_run: bool
    dry_run: bool
    task_preflight: bool


@dataclass(frozen=True)
class StartupGuard:
    enabled: bool
    marker: str
    timeout_sec: int
    retry: int


@dataclass(frozen=True)
class MonitorSettings:
    enabled: bool
    mode: str
    refresh_sec: float
    silent_warn_sec: int
    keep_finished_rows: int
    theme: str
    compact_paths: bool
    show_gpu_panel: bool
    show_data_panel: bool


@dataclass(frozen=True)
class ProgressSettings:
    enabled: bool
    mode: str
    event_poll_interval_sec: float
    dataset_scan_interval_sec: float
    action_fps: float
    video_fps: float
    final_ffprobe_verify: bool


@dataclass(frozen=True)
class FailedEpisodeCleanupSettings:
    enabled: bool
    mode: str
    require_finalized_event: bool
    require_run_time_window: bool
    delete_dirs: bool
    keep_summary: bool


@dataclass(frozen=True)
class CacheCleanupSettings:
    enabled: bool
    scope: str
    root: str
    cleanup_on_success: bool
    cleanup_on_failure: bool
    cleanup_on_interrupt: bool
    keep_summary: bool


@dataclass(frozen=True)
class FailureGuardSettings:
    enabled: bool
    kill_on_fatal_log: bool
    kill_on_suspect_hang: bool
    suspect_hang_kill_sec: int


@dataclass(frozen=True)
class GpuSamplingSettings:
    enabled: bool
    interval_sec: float
    output: str


@dataclass
class ConcreteJob:
    queue_idx: int
    job_id: str
    source_job_id: str
    task_path: str
    de_config: str
    random_num: int
    dataset_root: str
    dataset_root_source: str
    scene_info: str | None
    seed: int
    task_class: str
    output_task_dir: str
    output_collect_info: str
    output_robot_names: list[str]
    allowed_gpus: set[int]
    fixed_gpu: int | None
    shard_idx: int
    shard_count: int
    task_dir: Path
    data_engine_name: str
    launcher_args: list[str]
    docker_mounts: list[tuple[str, str]] = field(default_factory=list)
    status: str = "pending"
    worker_name: str = ""
    gpu: int | None = None
    attempt: int = 0
    max_attempts: int = 1
    container_id: str = ""
    container_name: str = ""
    start_time: float | None = None
    end_time: float | None = None
    exit_code: int | None = None
    failure_reason: str = ""
    log_path: Path | None = None
    docker_start_err: Path | None = None
    command_path: Path | None = None
    episode_event_path: Path | None = None
    episode_event_container_path: str = ""
    startup_complete: bool = False
    last_log_size: int = 0
    last_log_time: float | None = None
    suspect_hang: bool = False
    fatal_detected: bool = False
    fatal_reason: str = ""
    fatal_log_line: str = ""
    current_gpu_mem_mib: int = 0
    peak_gpu_mem_mib: int = 0
    last_gpu_sample_time: str = ""
    processed_event_ids: set[str] = field(default_factory=set)
    generated_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    success_trajectory_sec: float = 0.0
    success_action_steps: int = 0
    success_action_duration_sec: float = 0.0
    success_data_bytes: int = 0
    failed_data_bytes: int = 0
    deleted_failed_episode_count: int = 0
    freed_failed_episode_bytes: int = 0
    last_episode_time: str = ""


class ParallelRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        run_dir: Path,
        settings: ParallelSettings,
        startup_guard: StartupGuard,
        monitor: MonitorSettings,
        progress: ProgressSettings,
        failed_cleanup: FailedEpisodeCleanupSettings,
        cache_cleanup: CacheCleanupSettings,
        failure_guard: FailureGuardSettings,
        gpu_sampling: GpuSamplingSettings,
        jobs: list[ConcreteJob],
        run_start_unix: float | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.settings = settings
        self.startup_guard = startup_guard
        self.monitor = monitor
        self.progress = progress
        self.failed_cleanup = failed_cleanup
        self.cache_cleanup = cache_cleanup
        self.failure_guard = failure_guard
        self.gpu_sampling = gpu_sampling
        self.jobs = jobs
        self.run_start_unix = run_start_unix or time.time()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.running_containers: set[str] = set()
        self.manifest_path = run_dir / "manifest.jsonl"
        self.failure_path = run_dir / "failures.tsv"
        self.episode_events_path = run_dir / "episode_events.jsonl"
        self.deleted_failed_path = run_dir / "deleted_failed_episodes.jsonl"
        self.job_summary_csv_path = run_dir / "job_summary.csv"
        self.status_path = run_dir / "status.json"
        self.report_json_path = run_dir / "run_report.json"
        self.report_md_path = run_dir / "run_report.md"
        self.gpu_samples_path = run_dir / gpu_sampling.output
        self._last_status_write = 0.0
        self._last_event_refresh = 0.0
        self._last_gpu_refresh = 0.0
        self._last_dataset_scan = 0.0
        self._last_gpu_sample = 0.0
        self.gpu_stats: dict[int, dict[str, Any]] = {}
        self.gpu_process_stats: list[dict[str, Any]] = []
        self.cache_cleanup_result: dict[str, Any] = {}

    def append_manifest(self, event: str, **values: Any) -> None:
        record = {"event": event, "time": utc_now(), **values}
        with self.lock:
            with self.manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def write_status(self) -> None:
        with self.lock:
            payload = {
                "generated_at": utc_now(),
                "run_id": self.settings.run_id,
                "jobs": [self.job_status(job) for job in self.jobs],
                "summary": self.summary_locked(),
                "gpu_stats": self.gpu_stats,
                "gpu_process_stats": self.gpu_process_stats,
                "cache_cleanup": self.cache_cleanup_result,
                "paths": {
                    "run_dir": self.run_dir.as_posix(),
                    "run_report_md": self.report_md_path.as_posix(),
                    "run_report_json": self.report_json_path.as_posix(),
                    "job_summary_csv": self.job_summary_csv_path.as_posix(),
                    "gpu_samples_csv": self.gpu_samples_path.as_posix(),
                    "episode_events": self.episode_events_path.as_posix(),
                    "deleted_failed_episodes": self.deleted_failed_path.as_posix(),
                },
            }
            self.status_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._last_status_write = time.time()

    def maybe_write_status(self, interval_sec: float = 5.0) -> None:
        if time.time() - self._last_status_write >= interval_sec:
            self.write_status()

    def job_status(self, job: ConcreteJob) -> dict[str, Any]:
        elapsed = None
        if job.start_time is not None:
            elapsed = (job.end_time or time.time()) - job.start_time
        silent_age = None
        if job.last_log_time is not None:
            silent_age = time.time() - job.last_log_time
        return {
            "queue_idx": job.queue_idx,
            "job_id": job.job_id,
            "task_path": job.task_path,
            "de_config": job.de_config,
            "random_num": job.random_num,
            "dataset_root": job.dataset_root,
            "seed": job.seed,
            "allowed_gpus": sorted(job.allowed_gpus),
            "fixed_gpu": job.fixed_gpu,
            "gpu": job.gpu,
            "worker": job.worker_name,
            "status": job.status,
            "display_status": display_status(job),
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "container_id": job.container_id,
            "container_name": job.container_name,
            "exit_code": job.exit_code,
            "failure_reason": job.failure_reason,
            "fatal_detected": job.fatal_detected,
            "fatal_reason": job.fatal_reason,
            "fatal_log_line": job.fatal_log_line,
            "startup_complete": job.startup_complete,
            "suspect_hang": job.suspect_hang,
            "current_gpu_mem_mib": job.current_gpu_mem_mib,
            "peak_gpu_mem_mib": job.peak_gpu_mem_mib,
            "last_gpu_sample_time": job.last_gpu_sample_time,
            "elapsed_sec": round(elapsed, 3) if elapsed is not None else None,
            "silent_age_sec": round(silent_age, 3) if silent_age is not None else None,
            "log_path": job.log_path.as_posix() if job.log_path else "",
            "task_dir": job.task_dir.as_posix(),
            "episode_event_path": job.episode_event_path.as_posix() if job.episode_event_path else "",
            "progress": self.job_progress_locked(job),
        }

    def job_progress_locked(self, job: ConcreteJob) -> dict[str, Any]:
        return {
            "target": job.random_num,
            "generated": job.generated_count,
            "success": job.success_count,
            "failed": job.failed_count,
            "success_rate": round(percent(job.success_count, max(1, job.generated_count)), 3),
            "target_completion_rate": round(percent(job.generated_count, job.random_num), 3),
            "success_trajectory_sec": round(job.success_trajectory_sec, 6),
            "success_trajectory_hours": round(job.success_trajectory_sec / 3600.0, 9),
            "success_action_steps": job.success_action_steps,
            "success_action_duration_sec": round(job.success_action_duration_sec, 6),
            "success_data_bytes": job.success_data_bytes,
            "failed_data_bytes": job.failed_data_bytes,
            "total_data_bytes": job.success_data_bytes + job.failed_data_bytes,
            "deleted_failed_episode_count": job.deleted_failed_episode_count,
            "freed_failed_episode_bytes": job.freed_failed_episode_bytes,
            "last_episode_time": job.last_episode_time,
        }

    def summary_locked(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        success = counts.get("success", 0)
        failed = counts.get("failed", 0) + counts.get("timeout", 0)
        target = sum(job.random_num for job in self.jobs)
        generated = sum(job.generated_count for job in self.jobs)
        episode_success = sum(job.success_count for job in self.jobs)
        episode_failed = sum(job.failed_count for job in self.jobs)
        trajectory_sec = sum(job.success_trajectory_sec for job in self.jobs)
        action_steps = sum(job.success_action_steps for job in self.jobs)
        success_data_bytes = sum(job.success_data_bytes for job in self.jobs)
        failed_data_bytes = sum(job.failed_data_bytes for job in self.jobs)
        freed_failed_bytes = sum(job.freed_failed_episode_bytes for job in self.jobs)
        return {
            "total_jobs": len(self.jobs),
            "status_counts": counts,
            "success": success,
            "failed": failed,
            "startup_hang": sum(1 for job in self.jobs if job.failure_reason == "isaac_startup_hang"),
            "retrying_or_retried": sum(1 for job in self.jobs if job.attempt > 1),
            "suspect_hang": sum(1 for job in self.jobs if job.suspect_hang),
            "target_episodes": target,
            "generated_episodes": generated,
            "successful_episodes": episode_success,
            "failed_episodes": episode_failed,
            "episode_success_rate": round(percent(episode_success, max(1, generated)), 3),
            "target_completion_rate": round(percent(generated, target), 3),
            "success_trajectory_sec": round(trajectory_sec, 6),
            "success_trajectory_hours": round(trajectory_sec / 3600.0, 9),
            "success_action_steps": action_steps,
            "success_data_bytes": success_data_bytes,
            "failed_data_bytes": failed_data_bytes,
            "total_data_bytes": success_data_bytes + failed_data_bytes,
            "freed_failed_episode_bytes": freed_failed_bytes,
            "deleted_failed_episode_count": sum(job.deleted_failed_episode_count for job in self.jobs),
        }

    def claim_job(self, gpu: int, worker_name: str) -> ConcreteJob | None:
        with self.lock:
            for job in self.jobs:
                if job.status != "pending":
                    continue
                if gpu not in job.allowed_gpus:
                    continue
                job.status = "starting"
                job.gpu = gpu
                job.worker_name = worker_name
                self.append_manifest(
                    "job_claim",
                    queue_idx=job.queue_idx,
                    job_id=job.job_id,
                    gpu=gpu,
                    worker=worker_name,
                )
                return job
        return None

    def mark_job(self, job: ConcreteJob, status: str, **values: Any) -> None:
        with self.lock:
            job.status = status
            for key, value in values.items():
                setattr(job, key, value)
            self.write_status()

    def register_container(self, cid: str) -> None:
        with self.lock:
            if cid:
                self.running_containers.add(cid)

    def unregister_container(self, cid: str) -> None:
        with self.lock:
            if cid:
                self.running_containers.discard(cid)

    def stop_containers(self) -> None:
        with self.lock:
            cids = list(self.running_containers)
        for cid in cids:
            subprocess.run(["docker", "rm", "-f", cid], cwd=self.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_event_episode_path(path_s: str, repo_root: Path) -> Path:
    path = Path(path_s)
    return path if path.is_absolute() else repo_root / path


def docker_visible_path_and_mounts(path: Path, repo_root: Path) -> tuple[str, list[tuple[str, str]]]:
    path_abs = path if path.is_absolute() else repo_root / path
    parent = path_abs.parent
    try:
        parent_resolved = parent.resolve()
    except OSError:
        parent_resolved = parent
    resolved_path = parent_resolved / path_abs.name
    try:
        parent_resolved.relative_to(repo_root.resolve())
        rel_path = path_abs.relative_to(repo_root).as_posix()
        return f"/workspace/{rel_path}", []
    except ValueError:
        return resolved_path.as_posix(), [(parent_resolved.as_posix(), parent_resolved.as_posix())]


def append_docker_mount(job: ConcreteJob, host_path: str, container_path: str) -> None:
    mount = (host_path, container_path)
    if mount not in job.docker_mounts:
        job.docker_mounts.append(mount)


def read_job_events(job: ConcreteJob) -> list[dict[str, Any]]:
    if not job.episode_event_path or not job.episode_event_path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = job.episode_event_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def episode_mtime(episode_dir: Path) -> float:
    meta_path = episode_dir / "meta_info.pkl"
    try:
        return meta_path.stat().st_mtime
    except OSError:
        try:
            return episode_dir.stat().st_mtime
        except OSError:
            return 0.0


def episode_after_run_start(runtime: ParallelRuntime, episode_dir: Path) -> bool:
    if not runtime.failed_cleanup.require_run_time_window:
        return True
    return episode_mtime(episode_dir) >= max(0.0, runtime.run_start_unix - 5.0)


def job_dataset_roots(runtime: ParallelRuntime, job: ConcreteJob) -> list[Path]:
    dataset_root = resolve_host_path(job.dataset_root, runtime.repo_root)
    task_dir = Path(job.output_task_dir)
    roots: list[Path] = []
    robot_names = job.output_robot_names or ["*"]
    for robot_name in [*robot_names, "*"]:
        base = dataset_root / job.task_class / robot_name / task_dir
        if job.output_collect_info:
            roots.append(base / job.output_collect_info)
        roots.append(base)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = root.as_posix()
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def episode_belongs_to_job(runtime: ParallelRuntime, job: ConcreteJob, episode_dir: Path) -> bool:
    return any(path_under(episode_dir, root) for root in job_dataset_roots(runtime, job) if root.exists())


def load_episode_meta(episode_dir: Path) -> dict[str, Any]:
    try:
        with (episode_dir / "meta_info.pkl").open("rb") as f:
            meta = pickle.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def summarize_episode_for_scan(runtime: ParallelRuntime, episode_dir: Path) -> dict[str, Any]:
    meta = load_episode_meta(episode_dir)
    num_steps = int(meta.get("num_steps") or 0)
    trajectory_sec = (num_steps / runtime.progress.video_fps) if runtime.progress.video_fps > 0 and num_steps else 0.0
    return {
        "episode_dir": episode_dir,
        "status": "failed" if episode_dir.name.startswith("fail_") else "success",
        "num_steps": num_steps,
        "trajectory_sec": trajectory_sec,
        "action_duration_sec": (num_steps / runtime.progress.action_fps) if runtime.progress.action_fps > 0 else 0.0,
        "bytes": directory_size(episode_dir),
        "mtime": episode_mtime(episode_dir),
    }


def scan_job_dataset(runtime: ParallelRuntime, job: ConcreteJob) -> dict[str, Any]:
    episode_dirs: set[Path] = set()
    for root in job_dataset_roots(runtime, job):
        candidate_roots = [Path(path) for path in glob.glob(root.as_posix())] if "*" in root.as_posix() else [root]
        for candidate_root in candidate_roots:
            if not candidate_root.exists():
                continue
            for meta_path in candidate_root.rglob("meta_info.pkl"):
                episode_dir = meta_path.parent
                if not (episode_dir / "lmdb" / "data.mdb").exists():
                    continue
                if not episode_after_run_start(runtime, episode_dir):
                    continue
                episode_dirs.add(episode_dir)

    success = failed = 0
    success_bytes = failed_bytes = 0
    success_steps = 0
    success_action_sec = 0.0
    success_traj_sec = 0.0
    last_mtime = 0.0
    failed_summaries: list[dict[str, Any]] = []
    for episode_dir in sorted(episode_dirs):
        item = summarize_episode_for_scan(runtime, episode_dir)
        last_mtime = max(last_mtime, float(item["mtime"]))
        if item["status"] == "failed":
            failed += 1
            failed_bytes += int(item["bytes"])
            failed_summaries.append(item)
        else:
            success += 1
            success_bytes += int(item["bytes"])
            success_steps += int(item["num_steps"])
            success_action_sec += float(item["action_duration_sec"])
            success_traj_sec += float(item["trajectory_sec"])
    return {
        "generated": success + failed,
        "success": success,
        "failed": failed,
        "success_data_bytes": success_bytes,
        "failed_data_bytes": failed_bytes,
        "success_action_steps": success_steps,
        "success_action_duration_sec": success_action_sec,
        "success_trajectory_sec": success_traj_sec,
        "last_episode_time": datetime.fromtimestamp(last_mtime, timezone.utc).isoformat() if last_mtime else "",
        "failed_summaries": failed_summaries,
    }


def cleanup_failed_episode_dir(
    runtime: ParallelRuntime,
    job: ConcreteJob,
    episode_dir: Path,
    *,
    source_event_id: str = "",
    failure_reason: str = "",
    source: str = "dataset_scan",
) -> None:
    if not runtime.failed_cleanup.enabled or not runtime.failed_cleanup.delete_dirs:
        return
    if runtime.failed_cleanup.mode != "conservative":
        return
    if not episode_dir.name.startswith("fail_"):
        return
    if not (episode_dir / "meta_info.pkl").exists() or not (episode_dir / "lmdb" / "data.mdb").exists():
        return
    if not episode_after_run_start(runtime, episode_dir):
        return
    if not episode_belongs_to_job(runtime, job, episode_dir):
        return
    if not episode_dir.exists() or not episode_dir.is_dir():
        return
    size_bytes = directory_size(episode_dir)
    record = {
        "event": "failed_episode_deleted",
        "time": utc_now(),
        "run_id": runtime.settings.run_id,
        "queue_idx": job.queue_idx,
        "job_id": job.job_id,
        "gpu": job.gpu,
        "worker": job.worker_name,
        "episode_dir": episode_dir.as_posix(),
        "size_bytes": size_bytes,
        "source_event_id": source_event_id,
        "failure_reason": failure_reason,
        "source": source,
    }
    try:
        shutil.rmtree(episode_dir)
        with runtime.lock:
            job.deleted_failed_episode_count += 1
            job.freed_failed_episode_bytes += size_bytes
        append_jsonl(runtime.deleted_failed_path, record)
        manifest_record = {key: value for key, value in record.items() if key != "event"}
        runtime.append_manifest("failed_episode_deleted", **manifest_record)
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(runtime.deleted_failed_path, record)
        manifest_record = {key: value for key, value in record.items() if key != "event"}
        runtime.append_manifest("failed_episode_delete_failed", **manifest_record)


def cleanup_failed_episode_dirs(runtime: ParallelRuntime, job: ConcreteJob, event: dict[str, Any]) -> None:
    if not runtime.failed_cleanup.enabled or not runtime.failed_cleanup.delete_dirs:
        return
    if runtime.failed_cleanup.require_finalized_event and not event.get("finalized"):
        return
    episode_dirs = event.get("episode_dirs")
    if not isinstance(episode_dirs, list):
        return
    for episode_dir_s in episode_dirs:
        if not isinstance(episode_dir_s, str) or not episode_dir_s:
            continue
        episode_dir = resolve_event_episode_path(episode_dir_s, runtime.repo_root)
        cleanup_failed_episode_dir(
            runtime,
            job,
            episode_dir,
            source_event_id=str(event.get("event_id", "")),
            failure_reason=str(event.get("failure_reason", "")),
            source="episode_event",
        )


def apply_episode_event(runtime: ParallelRuntime, job: ConcreteJob, event: dict[str, Any]) -> None:
    event_id = str(event.get("event_id") or hashlib.sha1(json.dumps(event, sort_keys=True).encode("utf-8")).hexdigest())
    if event_id in job.processed_event_ids:
        return
    job.processed_event_ids.add(event_id)
    if event.get("event") != "episode_saved":
        return

    status = str(event.get("status") or "")
    num_steps = int(event.get("num_steps") or 0)
    trajectory_sec = float(event.get("trajectory_duration_sec") or 0.0)
    action_duration_sec = float(event.get("action_duration_sec") or 0.0)
    episode_bytes = int(event.get("episode_bytes") or 0)
    job.generated_count += 1
    job.last_episode_time = str(event.get("time") or "")
    if status == "success":
        job.success_count += 1
        job.success_trajectory_sec += trajectory_sec
        job.success_action_steps += num_steps
        job.success_action_duration_sec += action_duration_sec
        job.success_data_bytes += episode_bytes
    elif status == "failed":
        job.failed_count += 1
        job.failed_data_bytes += episode_bytes
        cleanup_failed_episode_dirs(runtime, job, event)

    enriched = dict(event)
    enriched.update(
        {
            "queue_idx": job.queue_idx,
            "launcher_job_id": job.job_id,
            "launcher_worker": job.worker_name,
            "launcher_gpu": job.gpu,
        }
    )
    append_jsonl(runtime.episode_events_path, enriched)
    runtime.append_manifest(
        "episode_event",
        queue_idx=job.queue_idx,
        job_id=job.job_id,
        status=status,
        num_steps=num_steps,
        trajectory_duration_sec=trajectory_sec,
        episode_bytes=episode_bytes,
        source_event_id=event_id,
    )


def refresh_episode_events(runtime: ParallelRuntime, *, force: bool = False) -> None:
    if not runtime.progress.enabled:
        return
    now = time.time()
    if not force and now - runtime._last_event_refresh < runtime.progress.event_poll_interval_sec:
        return
    with runtime.lock:
        for job in runtime.jobs:
            for event in read_job_events(job):
                apply_episode_event(runtime, job, event)
        runtime._last_event_refresh = now


def refresh_dataset_scan(runtime: ParallelRuntime, *, force: bool = False) -> None:
    if not runtime.progress.enabled:
        return
    now = time.time()
    if not force and now - runtime._last_dataset_scan < runtime.progress.dataset_scan_interval_sec:
        return
    for job in runtime.jobs:
        scan = scan_job_dataset(runtime, job)
        for item in scan["failed_summaries"]:
            cleanup_failed_episode_dir(
                runtime,
                job,
                item["episode_dir"],
                failure_reason=job.failure_reason,
                source="dataset_scan",
            )
        with runtime.lock:
            job.generated_count = max(job.generated_count, int(scan["generated"]))
            job.success_count = max(job.success_count, int(scan["success"]))
            job.failed_count = max(job.failed_count, int(scan["failed"]))
            job.success_data_bytes = max(job.success_data_bytes, int(scan["success_data_bytes"]))
            job.failed_data_bytes = max(job.failed_data_bytes, int(scan["failed_data_bytes"]))
            job.success_action_steps = max(job.success_action_steps, int(scan["success_action_steps"]))
            job.success_action_duration_sec = max(
                job.success_action_duration_sec,
                float(scan["success_action_duration_sec"]),
            )
            job.success_trajectory_sec = max(job.success_trajectory_sec, float(scan["success_trajectory_sec"]))
            if scan["last_episode_time"]:
                job.last_episode_time = str(scan["last_episode_time"])
    runtime._last_dataset_scan = now


def refresh_progress(runtime: ParallelRuntime, *, force: bool = False) -> None:
    refresh_episode_events(runtime, force=force)
    refresh_dataset_scan(runtime, force=force)


def refresh_gpu_stats(runtime: ParallelRuntime, *, force: bool = False) -> None:
    now = time.time()
    if not force and now - runtime._last_gpu_refresh < max(1.0, runtime.monitor.refresh_sec):
        return
    if not shutil.which("nvidia-smi"):
        return
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    if result.returncode != 0:
        return
    stats: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            stats[idx] = {
                "uuid": parts[1],
                "util_percent": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "memory_total_mib": float(parts[4]),
            }
        except ValueError:
            continue
    with runtime.lock:
        runtime.gpu_stats = stats
        runtime._last_gpu_refresh = now
    sample_gpu_processes(runtime, force=force)


def sample_gpu_processes(runtime: ParallelRuntime, *, force: bool = False) -> None:
    if not runtime.gpu_sampling.enabled or runtime.settings.dry_run:
        return
    now = time.time()
    if not force and now - runtime._last_gpu_sample < runtime.gpu_sampling.interval_sec:
        return
    if not shutil.which("nvidia-smi") or not shutil.which("docker"):
        return
    uuid_to_gpu = {
        str(stats.get("uuid")): gpu
        for gpu, stats in runtime.gpu_stats.items()
        if stats.get("uuid")
    }
    if not uuid_to_gpu:
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        cwd=runtime.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return

    pid_to_job: dict[int, ConcreteJob] = {}
    with runtime.lock:
        jobs = [job for job in runtime.jobs if job.container_id]
    for job in jobs:
        top = subprocess.run(
            ["docker", "top", job.container_id, "-eo", "pid"],
            cwd=runtime.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if top.returncode != 0:
            continue
        for line in top.stdout.splitlines()[1:]:
            try:
                pid_to_job[int(line.strip().split()[0])] = job
            except (IndexError, ValueError):
                continue

    sample_time = utc_now()
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
            mem_mib = int(parts[2])
        except ValueError:
            continue
        job = pid_to_job.get(pid)
        gpu = uuid_to_gpu.get(parts[0], "")
        row = {
            "time": sample_time,
            "run_id": runtime.settings.run_id,
            "gpu": gpu,
            "pid": pid,
            "process_mem_mib": mem_mib,
            "process_name": ",".join(parts[3:]),
            "container_id": job.container_id if job else "",
            "container_name": job.container_name if job else "",
            "job_id": job.job_id if job else "",
            "worker": job.worker_name if job else "",
        }
        rows.append(row)
        if job:
            with runtime.lock:
                job.current_gpu_mem_mib = mem_mib
                job.peak_gpu_mem_mib = max(job.peak_gpu_mem_mib, mem_mib)
                job.last_gpu_sample_time = sample_time
    if not rows:
        runtime._last_gpu_sample = now
        return

    fields = [
        "time",
        "run_id",
        "gpu",
        "pid",
        "process_mem_mib",
        "process_name",
        "container_id",
        "container_name",
        "job_id",
        "worker",
    ]
    write_header = not runtime.gpu_samples_path.exists() or runtime.gpu_samples_path.stat().st_size == 0
    runtime.gpu_samples_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.gpu_samples_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    with runtime.lock:
        runtime.gpu_process_stats = rows
        runtime._last_gpu_sample = now


def build_configured_jobs(
    *,
    config: dict[str, Any],
    repo_root: Path,
    run_dir: Path,
    settings: ParallelSettings,
    startup_guard: StartupGuard,
) -> list[ConcreteJob]:
    defaults = config.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    raw_jobs = config.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a non-empty list")

    worker_slots = [gpu for _worker_idx in range(settings.workers_per_gpu) for gpu in settings.gpus]
    concrete: list[ConcreteJob] = []

    def expand_job_entry(entry: dict[str, Any], entry_index: int) -> list[dict[str, Any]]:
        if "expand" not in entry:
            return [dict(entry)]
        expand = entry.get("expand")
        if not isinstance(expand, dict):
            raise ValueError(f"jobs[{entry_index}].expand must be a mapping")
        tasks = expand.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"jobs[{entry_index}].expand.tasks must be a non-empty list")
        base = {key: value for key, value in entry.items() if key != "expand"}
        expanded = []
        for idx, task in enumerate(tasks):
            item = dict(base)
            item.update({key: value for key, value in expand.items() if key != "tasks"})
            item["task"] = task
            item["id"] = f"{base.get('id', f'job{entry_index}')}_{idx}"
            expanded.append(item)
        return expanded

    expanded_jobs: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(raw_jobs):
        if not isinstance(entry, dict):
            raise ValueError(f"jobs[{entry_index}] must be a mapping")
        expanded_jobs.extend(expand_job_entry(entry, entry_index))

    queue_idx = 0
    for job_index, job_entry in enumerate(expanded_jobs):
        merged = dict(defaults)
        merged.update(job_entry)
        source_job_id = safe_id(str(merged.get("id") or f"job{job_index}"))
        task_source = str_or_none(merged.get("task"))
        if not task_source:
            raise ValueError(f"Job {source_job_id} is missing task")
        task_paths = discover_task_paths(task_source, repo_root)
        if not task_paths:
            raise ValueError(f"Job {source_job_id} has no task YAMLs from {task_source}")

        de_config = rel_repo_path(str(merged.get("de_config") or "configs/simbox/de_plan_with_render_template.yaml"), repo_root)
        if settings.backend == "docker":
            ensure_repo_visible_for_docker(de_config, repo_root, f"{source_job_id}.de_config")
        if not resolve_host_path(de_config, repo_root).is_file():
            raise FileNotFoundError(f"DataEngine config not found for job {source_job_id}: {de_config}")
        random_num = int_value(merged.get("random_num"), default=10)
        if random_num < 1:
            raise ValueError(f"random_num must be positive for job {source_job_id}")
        scene_info = str_or_none(merged.get("scene_info"))
        seed_base_raw = merged.get("seed_base", merged.get("random_seed_base"))
        seed_base = int(seed_base_raw) if seed_base_raw is not None else None
        dataset_root_raw = str_or_none(merged.get("dataset_root"))
        if dataset_root_raw:
            dataset_root = rel_repo_path(dataset_root_raw, repo_root).rstrip("/")
            dataset_root_source = "job"
        else:
            dataset_root = rel_repo_path(read_de_config_output_dir(de_config, repo_root), repo_root).rstrip("/")
            dataset_root_source = "de_config"
        dataset_root_host = resolve_host_path(dataset_root, repo_root)
        docker_mounts: list[tuple[str, str]] = []
        if settings.backend == "docker" and not path_inside_repo(dataset_root, repo_root):
            docker_mounts.append((dataset_root_host.as_posix(), dataset_root_host.as_posix()))
        dataset_root_host.mkdir(parents=True, exist_ok=True)

        fixed_gpu = None
        if merged.get("gpu") is not None:
            fixed_gpu = int(merged["gpu"])
            allowed_gpus = {fixed_gpu}
        else:
            allowed = merged.get("allowed_gpus")
            allowed_gpus = set(list_of_ints(allowed, name=f"{source_job_id}.allowed_gpus")) if allowed is not None else set(settings.gpus)
        if not allowed_gpus:
            allowed_gpus = set(settings.gpus)
        unknown_gpus = sorted(allowed_gpus - set(settings.gpus))
        if unknown_gpus:
            raise ValueError(f"Job {source_job_id} references GPU(s) not in parallel.gpus: {unknown_gpus}")

        shard_random_num = bool_value(merged.get("shard_random_num"), default=False)
        eligible_slots = [gpu for gpu in worker_slots if gpu in allowed_gpus]
        if not eligible_slots:
            raise ValueError(f"Job {source_job_id} has no eligible worker slots")

        for task_index, task_path in enumerate(task_paths):
            if settings.backend == "docker":
                ensure_repo_visible_for_docker(task_path, repo_root, f"{source_job_id}.task")
            if not resolve_host_path(task_path, repo_root).is_file():
                raise FileNotFoundError(f"Task YAML not found for job {source_job_id}: {task_path}")
            output_meta = read_task_output_metadata(task_path, repo_root)
            shard_count = min(random_num, len(eligible_slots)) if shard_random_num else 1
            counts = split_counts(random_num, shard_count)
            for shard_idx, shard_random in enumerate(counts):
                assigned_gpu = eligible_slots[shard_idx % len(eligible_slots)] if shard_random_num else fixed_gpu
                effective_allowed = {assigned_gpu} if assigned_gpu is not None else set(allowed_gpus)
                job_slug = slugify_task(task_path)
                job_id = source_job_id
                if len(task_paths) > 1:
                    job_id = f"{job_id}_t{task_index}"
                if shard_count > 1:
                    job_id = f"{job_id}_s{shard_idx}"
                job_id = safe_id(job_id)
                worker_fragment = "unassigned"
                task_dir = run_dir / worker_fragment / f"q{queue_idx}_{job_id}_{job_slug}"
                data_engine_name = f"_parallel_runs/{settings.run_id}/{worker_fragment}/q{queue_idx}_{job_id}_{job_slug}"
                seed = stable_seed(settings.run_id, job_id, shard_idx, seed_base, queue_idx)
                launcher_args = [
                    f"--name={data_engine_name}",
                    f"--load_stage.scene_loader.args.cfg_path={task_path}",
                    "--load_stage.scene_loader.args.simulator.active_gpu=0",
                    "--load_stage.scene_loader.args.simulator.physics_gpu=0",
                    "--load_stage.scene_loader.args.simulator.multi_gpu=false",
                    "--load_stage.scene_loader.args.simulator.max_gpu_count=1",
                    f"--load_stage.layout_random_generator.args.random_num={shard_random}",
                    f"--store_stage.writer.args.output_dir={dataset_root}/",
                    f"--random_seed={seed}",
                ]
                if scene_info:
                    launcher_args.append(f"--load_stage.scene_loader.args.scene_info={scene_info}")
                concrete.append(
                    ConcreteJob(
                        queue_idx=queue_idx,
                        job_id=job_id,
                        source_job_id=source_job_id,
                        task_path=task_path,
                        de_config=de_config,
                        random_num=shard_random,
                        dataset_root=dataset_root,
                        dataset_root_source=dataset_root_source,
                        scene_info=scene_info,
                        seed=seed,
                        task_class=str(output_meta["task_class"]),
                        output_task_dir=str(output_meta["task_dir"]),
                        output_collect_info=str(output_meta["collect_info"]),
                        output_robot_names=list(output_meta["robot_names"]),
                        allowed_gpus=effective_allowed,
                        fixed_gpu=assigned_gpu,
                        shard_idx=shard_idx,
                        shard_count=shard_count,
                        task_dir=task_dir,
                        data_engine_name=data_engine_name,
                        launcher_args=launcher_args,
                        docker_mounts=list(docker_mounts),
                        max_attempts=startup_guard.retry + 1,
                    )
                )
                queue_idx += 1
    return concrete


def finalize_job_paths(job: ConcreteJob, runtime: ParallelRuntime, gpu: int, worker_idx: int) -> None:
    worker_name = f"gpu{gpu}_w{worker_idx}"
    job.worker_name = worker_name
    job.gpu = gpu
    job.task_dir = runtime.run_dir / worker_name / f"q{job.queue_idx}_{job.job_id}_{slugify_task(job.task_path)}"
    job.data_engine_name = f"_parallel_runs/{runtime.settings.run_id}/{worker_name}/q{job.queue_idx}_{job.job_id}_{slugify_task(job.task_path)}"
    job.launcher_args = [
        arg if not arg.startswith("--name=") else f"--name={job.data_engine_name}"
        for arg in job.launcher_args
    ]
    job.log_path = job.task_dir / "docker.log"
    job.docker_start_err = job.task_dir / "docker_start.err"
    job.command_path = job.task_dir / "command.sh"
    job.episode_event_path = job.task_dir / "episode_events.jsonl"
    job.task_dir.mkdir(parents=True, exist_ok=True)
    event_container_path, event_mounts = docker_visible_path_and_mounts(job.episode_event_path, runtime.repo_root)
    job.episode_event_container_path = event_container_path
    for host_path, container_path in event_mounts:
        append_docker_mount(job, host_path, container_path)


def build_compose_command(runtime: ParallelRuntime, job: ConcreteJob) -> list[str]:
    assert job.gpu is not None
    container_name = f"interdata-{runtime.settings.run_id}-{job.worker_name}-q{job.queue_idx}-{job.job_id}"
    container_name = safe_id(container_name)[:180]
    job.container_name = container_name
    cmd = [
        "docker",
        "compose",
        "-f",
        runtime.settings.compose_file,
        "-p",
        f"interdata-simbox-{runtime.settings.run_id}",
        "run",
        "-d",
        "--no-deps",
        "--no-TTY",
    ]
    for host_path, container_path in job.docker_mounts:
        cmd.extend(["--volume", f"{host_path}:{container_path}:rw"])
    cmd.extend(
        [
            "--name",
            container_name,
            "isaac",
            "/isaac-sim/python.sh",
            "launcher.py",
            "--config",
            job.de_config,
            *job.launcher_args,
        ]
    )
    return cmd


def write_command_file(job: ConcreteJob, cmd: list[str], env: dict[str, str]) -> None:
    assert job.command_path is not None
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(env[key])}")
    lines.append(" ".join(shlex.quote(part) for part in cmd))
    job.command_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def docker_env(runtime: ParallelRuntime, job: ConcreteJob) -> dict[str, str]:
    assert job.gpu is not None
    cache_root = runtime.repo_root / ".docker" / "isaac-sim" / runtime.settings.run_id / job.worker_name
    for child in ["cache/main", "cache/computecache", "logs", "config", "data", "pkg"]:
        (cache_root / child).mkdir(parents=True, exist_ok=True)
    return {
        "INTERNDATA_ISAAC_GPU_DEVICE_IDS": str(job.gpu),
        "ISAAC_CACHE_MAIN": str(cache_root / "cache/main"),
        "ISAAC_CACHE_COMPUTE": str(cache_root / "cache/computecache"),
        "ISAAC_LOGS": str(cache_root / "logs"),
        "ISAAC_CONFIG": str(cache_root / "config"),
        "ISAAC_DATA": str(cache_root / "data"),
        "ISAAC_PKGS": str(cache_root / "pkg"),
        "INTERNDATA_RUN_IMPORT_CHECKS": os.environ.get("INTERNDATA_RUN_IMPORT_CHECKS", "0"),
        "INTERNDATA_RUN_ID": runtime.settings.run_id,
        "INTERNDATA_JOB_ID": job.job_id,
        "INTERNDATA_WORKER": job.worker_name,
        "INTERNDATA_GPU": str(job.gpu),
        "INTERNDATA_TASK_PATH": job.task_path,
        "INTERNDATA_DATASET_ROOT": job.dataset_root,
        "INTERNDATA_RANDOM_SEED": str(job.seed),
        "INTERNDATA_EPISODE_EVENT_PATH": job.episode_event_container_path,
        "INTERNDATA_ACTION_FPS": str(runtime.progress.action_fps),
        "INTERNDATA_VIDEO_FPS": str(runtime.progress.video_fps),
    }


def poll_container_exit(repo_root: Path, cid: str) -> int | None:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", cid],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return 125
    parts = result.stdout.strip().split()
    if not parts:
        return None
    if parts[0] == "true":
        return None
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 125


def refresh_log_state(job: ConcreteJob, runtime: ParallelRuntime) -> None:
    if not job.log_path or not job.log_path.exists():
        return
    try:
        size = job.log_path.stat().st_size
    except OSError:
        return
    if size != job.last_log_size:
        job.last_log_size = size
        job.last_log_time = time.time()
        if runtime.startup_guard.marker and not job.startup_complete:
            text = "\n".join(tail_lines(job.log_path, 120))
            if runtime.startup_guard.marker in text:
                job.startup_complete = True
                job.status = "running"
                runtime.append_manifest(
                    "startup_complete",
                    queue_idx=job.queue_idx,
                    job_id=job.job_id,
                    gpu=job.gpu,
                    worker=job.worker_name,
                )
    if job.last_log_time and runtime.monitor.silent_warn_sec > 0:
        job.suspect_hang = (time.time() - job.last_log_time) > runtime.monitor.silent_warn_sec


def detect_fatal_log(job: ConcreteJob) -> tuple[str, str]:
    if not job.log_path or not job.log_path.exists():
        return "", ""
    for line in tail_lines(job.log_path, 400):
        for reason, pattern in FATAL_LOG_PATTERNS:
            if pattern.search(line):
                return reason, line.strip()[:500]
    return "", ""


def apply_failure_guard(runtime: ParallelRuntime, job: ConcreteJob, cid: str) -> tuple[bool, str]:
    if not runtime.failure_guard.enabled:
        return False, ""
    if runtime.failure_guard.kill_on_fatal_log:
        reason, line = detect_fatal_log(job)
        if reason:
            job.fatal_detected = True
            job.fatal_reason = reason
            job.fatal_log_line = line
            job.failure_reason = reason
            runtime.append_manifest(
                "fatal_log_detected",
                queue_idx=job.queue_idx,
                job_id=job.job_id,
                container_id=cid,
                reason=reason,
                line=line,
            )
            subprocess.run(["docker", "rm", "-f", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, reason
    if (
        runtime.failure_guard.kill_on_suspect_hang
        and job.last_log_time is not None
        and time.time() - job.last_log_time > runtime.failure_guard.suspect_hang_kill_sec
    ):
        reason = "suspect_hang"
        job.failure_reason = reason
        job.suspect_hang = True
        runtime.append_manifest(
            "suspect_hang_killed",
            queue_idx=job.queue_idx,
            job_id=job.job_id,
            container_id=cid,
            silent_sec=round(time.time() - job.last_log_time, 3),
        )
        subprocess.run(["docker", "rm", "-f", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, reason
    return False, ""


def run_container_attempt(runtime: ParallelRuntime, job: ConcreteJob) -> tuple[str, int]:
    assert job.log_path is not None
    assert job.docker_start_err is not None
    env = docker_env(runtime, job)
    cmd = build_compose_command(runtime, job)
    write_command_file(job, cmd, env)
    with job.log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n[parallel] attempt {job.attempt}/{job.max_attempts} started at {utc_now()}\n")
    with job.docker_start_err.open("a", encoding="utf-8") as f:
        f.write(f"\n\n[parallel] attempt {job.attempt}/{job.max_attempts} started at {utc_now()}\n")

    runtime.append_manifest(
        "task_start",
        queue_idx=job.queue_idx,
        job_id=job.job_id,
        task_path=job.task_path,
        de_config=job.de_config,
        random_num=job.random_num,
        dataset_root=job.dataset_root,
        dataset_root_source=job.dataset_root_source,
        seed=job.seed,
        gpu=job.gpu,
        worker=job.worker_name,
        attempt=job.attempt,
        shard_idx=job.shard_idx,
        shard_count=job.shard_count,
        task_dir=job.task_dir.as_posix(),
    )

    if runtime.settings.dry_run:
        with job.log_path.open("w", encoding="utf-8") as f:
            f.write(f"[dry-run][{job.worker_name}] env INTERNDATA_ISAAC_GPU_DEVICE_IDS={job.gpu}\n")
            f.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
        runtime.append_manifest("task_finish", queue_idx=job.queue_idx, job_id=job.job_id, exit_code=0, dry_run=True)
        return "success", 0

    proc_env = os.environ.copy()
    proc_env.update(env)
    start = subprocess.run(
        cmd,
        cwd=runtime.repo_root,
        env=proc_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if start.stderr:
        with job.docker_start_err.open("a", encoding="utf-8") as f:
            f.write(start.stderr)
    if start.returncode != 0 or not start.stdout.strip():
        with job.log_path.open("a", encoding="utf-8") as f:
            f.write(start.stderr or start.stdout or "")
        runtime.append_manifest(
            "docker_start_failed",
            queue_idx=job.queue_idx,
            job_id=job.job_id,
            exit_code=start.returncode,
            stderr=(start.stderr or "")[-2000:],
        )
        return "failed", start.returncode or 125

    cid = start.stdout.strip().splitlines()[-1].strip()
    job.container_id = cid
    runtime.register_container(cid)
    if not runtime.startup_guard.enabled:
        job.status = "running"
        job.startup_complete = True
    runtime.append_manifest(
        "container_start",
        queue_idx=job.queue_idx,
        job_id=job.job_id,
        gpu=job.gpu,
        worker=job.worker_name,
        container_id=cid,
        container_name=job.container_name,
    )

    log_file = job.log_path.open("ab")
    log_proc = subprocess.Popen(
        ["docker", "logs", "-f", cid],
        cwd=runtime.repo_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    start_time = time.time()
    startup_deadline = start_time + runtime.startup_guard.timeout_sec
    task_deadline = start_time + job_timeout(runtime, job) if job_timeout(runtime, job) > 0 else None
    status = "failed"
    exit_code = 125
    try:
        while True:
            refresh_log_state(job, runtime)
            refresh_progress(runtime)
            refresh_gpu_stats(runtime)
            runtime.maybe_write_status()
            guarded, guard_reason = apply_failure_guard(runtime, job, cid)
            if guarded:
                status = "failed"
                exit_code = FATAL_GUARD_EXIT_CODE
                job.failure_reason = guard_reason
                break
            exit_code_now = poll_container_exit(runtime.repo_root, cid)
            if exit_code_now is not None:
                exit_code = exit_code_now
                status = "success" if exit_code == 0 else "failed"
                break
            now = time.time()
            if runtime.startup_guard.enabled and not job.startup_complete and now > startup_deadline:
                status = "startup_hang"
                exit_code = STARTUP_HANG_EXIT_CODE
                runtime.append_manifest(
                    "startup_hang",
                    queue_idx=job.queue_idx,
                    job_id=job.job_id,
                    container_id=cid,
                    marker=runtime.startup_guard.marker,
                    timeout_sec=runtime.startup_guard.timeout_sec,
                )
                subprocess.run(["docker", "rm", "-f", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            if task_deadline is not None and now > task_deadline:
                status = "timeout"
                exit_code = TASK_TIMEOUT_EXIT_CODE
                runtime.append_manifest(
                    "task_timeout",
                    queue_idx=job.queue_idx,
                    job_id=job.job_id,
                    container_id=cid,
                    timeout_sec=job_timeout(runtime, job),
                )
                subprocess.run(["docker", "rm", "-f", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            time.sleep(1.0)
    finally:
        try:
            log_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log_proc.terminate()
            try:
                log_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_proc.kill()
        log_file.close()
        subprocess.run(["docker", "rm", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        runtime.unregister_container(cid)
        refresh_log_state(job, runtime)
        refresh_progress(runtime, force=True)

    runtime.append_manifest(
        "task_attempt_finish",
        queue_idx=job.queue_idx,
        job_id=job.job_id,
        status=status,
        exit_code=exit_code,
        container_id=cid,
        log_file=job.log_path.as_posix(),
        attempt=job.attempt,
    )
    refresh_progress(runtime, force=True)
    return status, exit_code


def job_timeout(runtime: ParallelRuntime, job: ConcreteJob) -> int:
    return runtime.settings.task_timeout_sec


def run_job(runtime: ParallelRuntime, job: ConcreteJob, gpu: int, worker_idx: int) -> None:
    finalize_job_paths(job, runtime, gpu, worker_idx)
    job.start_time = time.time()
    job.last_log_time = job.start_time
    final_status = "failed"
    final_exit = 125
    for attempt in range(1, job.max_attempts + 1):
        job.attempt = attempt
        job.status = "starting"
        job.startup_complete = False
        job.suspect_hang = False
        runtime.write_status()
        status, exit_code = run_container_attempt(runtime, job)
        final_status = status
        final_exit = exit_code
        if status == "success":
            break
        if status == "startup_hang" and attempt < job.max_attempts:
            job.status = "retrying"
            runtime.append_manifest(
                "task_retry",
                queue_idx=job.queue_idx,
                job_id=job.job_id,
                reason="isaac_startup_hang",
                next_attempt=attempt + 1,
            )
            time.sleep(2)
            continue
        break

    job.end_time = time.time()
    job.exit_code = final_exit
    if final_status == "success":
        job.status = "success"
    elif final_status == "timeout":
        job.status = "timeout"
        job.failure_reason = "task_timeout"
    elif final_status == "startup_hang":
        job.status = "failed"
        job.failure_reason = "isaac_startup_hang"
    else:
        job.status = "failed"
        assert job.log_path is not None and job.docker_start_err is not None
        if not job.failure_reason:
            job.failure_reason = classify_failure(job.log_path, job.docker_start_err, final_status, final_exit)

    runtime.append_manifest(
        "task_finish",
        queue_idx=job.queue_idx,
        job_id=job.job_id,
        status=job.status,
        exit_code=job.exit_code,
        failure_reason=job.failure_reason,
        log_file=job.log_path.as_posix() if job.log_path else "",
        attempts=job.attempt,
    )
    if job.status != "success":
        with runtime.failure_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{job.exit_code}\t{job.gpu}\t{job.worker_name}\t{job.failure_reason}\t"
                f"{job.task_path}\t{job.log_path.as_posix() if job.log_path else ''}\n"
            )
    refresh_progress(runtime, force=True)
    runtime.write_status()


def worker_loop(runtime: ParallelRuntime, gpu: int, worker_idx: int) -> None:
    worker_name = f"gpu{gpu}_w{worker_idx}"
    while not runtime.stop_event.is_set():
        job = runtime.claim_job(gpu, worker_name)
        if job is None:
            return
        run_job(runtime, job, gpu, worker_idx)


def render_plain(runtime: ParallelRuntime) -> str:
    refresh_progress(runtime)
    refresh_gpu_stats(runtime)
    with runtime.lock:
        summary = runtime.summary_locked()
        rows = sorted(runtime.jobs, key=lambda job: job.queue_idx)
        active = [job for job in rows if job.status not in {"pending", "success", "failed", "timeout"}]
        pending = [job for job in rows if job.status == "pending"]
        done = [job for job in rows if job.status in {"success", "failed", "timeout"}]
        shown = active + pending[:10] + done[-runtime.monitor.keep_finished_rows :]
        lines = [
            f"SimBox Parallel V2.6 | run={runtime.settings.run_id} | {utc_now()}",
            f"jobs={summary['total_jobs']} target={summary['target_episodes']} "
            f"generated={summary['generated_episodes']} success={summary['successful_episodes']} "
            f"failed={summary['failed_episodes']} rate={summary['episode_success_rate']:.1f}% "
            f"traj={human_duration(summary['success_trajectory_sec'])} "
            f"data={human_bytes(summary['success_data_bytes'])}",
            "GPU  Worker   State          Job                         Done/Target  S/F    Rate    Traj     Elapsed  Silent",
            "---  -------  -------------  --------------------------  -----------  -----  ------  -------  -------  ------",
        ]
        now = time.time()
        for job in shown:
            elapsed = "-"
            if job.start_time is not None:
                elapsed = f"{int((job.end_time or now) - job.start_time)}s"
            silent = "-"
            if job.last_log_time is not None:
                silent = f"{int(now - job.last_log_time)}s"
            lines.append(
                f"{str(job.gpu if job.gpu is not None else '-'):>3}  "
                f"{job.worker_name or '-':<7}  {display_status(job):<13}  {job.job_id[:26]:<26}  "
                f"{job.generated_count:>4}/{job.random_num:<6}  "
                f"{job.success_count:>2}/{job.failed_count:<2}  "
                f"{percent(job.success_count, max(1, job.generated_count)):>5.1f}%  "
                f"{human_duration(job.success_trajectory_sec):<7}  {elapsed:<7}  {silent:<6}"
            )
        return "\n".join(lines)


def monitor_plain(runtime: ParallelRuntime) -> None:
    while not runtime.stop_event.is_set():
        print("\033[2J\033[H" + render_plain(runtime), flush=True)
        time.sleep(runtime.monitor.refresh_sec)
    print("\033[2J\033[H" + render_plain(runtime), flush=True)


def monitor_rich(runtime: ParallelRuntime) -> None:
    try:
        from rich import box
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        monitor_plain(runtime)
        return

    def status_style(status_text: str) -> str:
        return {
            "success": "bold green",
            "failed": "bold red",
            "timeout": "bold red",
            "startup_hang": "bold yellow",
            "suspect_hang": "bold yellow",
            "retrying": "bold yellow",
            "running": "bold cyan",
            "starting": "bold blue",
            "pending": "white",
        }.get(status_text, "white")

    def build_run_panel(summary: dict[str, Any], elapsed: float) -> Panel:
        table = Table.grid(expand=True)
        for _ in range(8):
            table.add_column(justify="center")
        table.add_row(
            "[bold white]Target[/]",
            "[bold green]Success[/]",
            "[bold red]Failed[/]",
            "[bold cyan]Generated[/]",
            "[bold yellow]Rate[/]",
            "[bold magenta]Traj[/]",
            "[bold blue]Steps[/]",
            "[bold white]Data[/]",
        )
        table.add_row(
            str(summary["target_episodes"]),
            str(summary["successful_episodes"]),
            str(summary["failed_episodes"]),
            str(summary["generated_episodes"]),
            f"{summary['episode_success_rate']:.1f}%",
            human_duration(summary["success_trajectory_sec"]),
            str(summary["success_action_steps"]),
            f"{human_bytes(summary['success_data_bytes'])}/{human_bytes(summary['total_data_bytes'])}",
        )
        subtitle = (
            f"elapsed={human_duration(elapsed)} | jobs={summary['total_jobs']} | "
            f"workers={len(runtime.settings.gpus) * runtime.settings.workers_per_gpu}"
        )
        return Panel(table, title="[bold cyan]Run[/]", subtitle=subtitle, border_style="cyan", box=box.SQUARE)

    def build_gpu_panel(rows: list[ConcreteJob]) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        for col in ["GPU", "Util", "Mem", "Workers", "Run", "OK", "Fail", "Current jobs"]:
            table.add_column(col)
        for gpu in runtime.settings.gpus:
            gpu_jobs = [job for job in rows if job.gpu == gpu]
            active_jobs = [job for job in gpu_jobs if job.status in {"starting", "running", "retrying"}]
            stats = runtime.gpu_stats.get(gpu, {})
            util = stats.get("util_percent")
            mem_used = stats.get("memory_used_mib")
            mem_total = stats.get("memory_total_mib")
            mem_pct = percent(mem_used or 0, mem_total or 0)
            util_text = "-" if util is None else f"{util:.0f}%"
            mem_text = "-" if mem_used is None or mem_total is None else f"{mem_used:.0f}/{mem_total:.0f}MiB {mem_pct:.0f}%"
            mem_style = "green" if mem_pct < 60 else "yellow" if mem_pct < 85 else "red"
            current_jobs = ", ".join(compact_text(job.job_id, 18) for job in active_jobs) or "-"
            table.add_row(
                f"[bold yellow]{gpu}[/]",
                f"[cyan]{util_text}[/]",
                f"[{mem_style}]{mem_text}[/]",
                str(runtime.settings.workers_per_gpu),
                f"[cyan]{len(active_jobs)}[/]",
                f"[green]{sum(job.success_count for job in gpu_jobs)}[/]",
                f"[red]{sum(job.failed_count for job in gpu_jobs)}[/]",
                current_jobs,
            )
        return Panel(table, title="[bold yellow]GPU[/]", border_style="yellow", box=box.SQUARE)

    def build_jobs_panel(rows: list[ConcreteJob], now: float) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        columns = [
            "GPU",
            "Worker",
            "State",
            "Job",
            "Task",
            "Done/Target",
            "S/F",
            "Rate",
            "Traj",
            "Data",
            "Peak",
            "Elapsed",
            "Silent",
            "Reason",
        ]
        for col in columns:
            table.add_column(col, no_wrap=col not in {"Job", "Task"})
        active = [job for job in rows if job.status not in {"pending", "success", "failed", "timeout"}]
        pending = [job for job in rows if job.status == "pending"]
        done = [job for job in rows if job.status in {"success", "failed", "timeout"}]
        shown = active + pending[:10] + done[-runtime.monitor.keep_finished_rows :]
        for job in shown:
            status_text = display_status(job)
            elapsed = "-" if job.start_time is None else human_duration((job.end_time or now) - job.start_time)
            silent = "-" if job.last_log_time is None else human_duration(now - job.last_log_time)
            task_display = compact_path(job.task_path, 36) if runtime.monitor.compact_paths else job.task_path
            job_display = compact_text(job.job_id, 28)
            rate = percent(job.success_count, max(1, job.generated_count))
            table.add_row(
                str(job.gpu if job.gpu is not None else "-"),
                job.worker_name or "-",
                f"[{status_style(status_text)}]{status_text}[/]",
                f"[white]{job_display}[/]",
                f"[dim]{task_display}[/]",
                f"[cyan]{job.generated_count}/{job.random_num}[/]",
                f"[green]{job.success_count}[/]/[red]{job.failed_count}[/]",
                f"[yellow]{rate:.1f}%[/]",
                f"[magenta]{human_duration(job.success_trajectory_sec)}[/]",
                human_bytes(job.success_data_bytes),
                f"{job.peak_gpu_mem_mib}MiB" if job.peak_gpu_mem_mib else "-",
                elapsed,
                silent,
                compact_text(job.failure_reason or job.fatal_reason, 18) or "-",
            )
        return Panel(table, title="[bold white]Jobs[/]", border_style="white", box=box.SQUARE)

    def build_dashboard() -> Group:
        refresh_progress(runtime)
        refresh_gpu_stats(runtime)
        with runtime.lock:
            summary = runtime.summary_locked()
            rows = sorted(runtime.jobs, key=lambda job: job.queue_idx)
            now = time.time()
            starts = [job.start_time for job in rows if job.start_time is not None]
            elapsed = now - min(starts) if starts else 0.0
            roots = ", ".join(compact_path(root, 42) for root in sorted({job.dataset_root for job in rows}))
            header = Text()
            header.append("SimBox Parallel V2.6", style="bold cyan")
            header.append(f" | {runtime.settings.run_id}", style="bold white")
            header.append(f" | dataset={roots or '-'}", style="dim")
            panels: list[Any] = [Panel(header, border_style="magenta", box=box.SQUARE)]
            if runtime.monitor.show_data_panel:
                panels.append(build_run_panel(summary, elapsed))
            if runtime.monitor.show_gpu_panel:
                panels.append(build_gpu_panel(rows))
            panels.append(build_jobs_panel(rows, now))
            return Group(*panels)

    refresh_rate = max(1, int(1 / runtime.monitor.refresh_sec)) if runtime.monitor.refresh_sec > 0 else 1
    with Live(build_dashboard(), refresh_per_second=refresh_rate) as live:
        while not runtime.stop_event.is_set():
            live.update(build_dashboard())
            time.sleep(runtime.monitor.refresh_sec)
        live.update(build_dashboard())


def monitor_loop(runtime: ParallelRuntime) -> None:
    if not runtime.monitor.enabled:
        return
    mode = runtime.monitor.mode
    if mode == "auto":
        mode = "rich" if shutil.which("python3") else "plain"
    if mode == "rich":
        monitor_rich(runtime)
    else:
        monitor_plain(runtime)


def write_job_summary_csv(runtime: ParallelRuntime) -> None:
    fields = [
        "queue_idx",
        "job_id",
        "task_path",
        "gpu",
        "worker",
        "status",
        "target",
        "generated",
        "success",
        "failed",
        "success_rate",
        "success_trajectory_sec",
        "success_action_steps",
        "success_action_duration_sec",
        "success_data_bytes",
        "failed_data_bytes",
        "total_data_bytes",
        "deleted_failed_episode_count",
        "freed_failed_episode_bytes",
        "peak_gpu_mem_mib",
        "last_episode_time",
        "elapsed_sec",
        "reason",
        "fatal_reason",
        "fatal_log_line",
        "log_path",
    ]
    with runtime.job_summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        now = time.time()
        with runtime.lock:
            for job in sorted(runtime.jobs, key=lambda item: item.queue_idx):
                progress = runtime.job_progress_locked(job)
                elapsed = None if job.start_time is None else (job.end_time or now) - job.start_time
                writer.writerow(
                    {
                        "queue_idx": job.queue_idx,
                        "job_id": job.job_id,
                        "task_path": job.task_path,
                        "gpu": job.gpu if job.gpu is not None else "",
                        "worker": job.worker_name,
                        "status": job.status,
                        "target": progress["target"],
                        "generated": progress["generated"],
                        "success": progress["success"],
                        "failed": progress["failed"],
                        "success_rate": progress["success_rate"],
                        "success_trajectory_sec": progress["success_trajectory_sec"],
                        "success_action_steps": progress["success_action_steps"],
                        "success_action_duration_sec": progress["success_action_duration_sec"],
                        "success_data_bytes": progress["success_data_bytes"],
                        "failed_data_bytes": progress["failed_data_bytes"],
                        "total_data_bytes": progress["total_data_bytes"],
                        "deleted_failed_episode_count": progress["deleted_failed_episode_count"],
                        "freed_failed_episode_bytes": progress["freed_failed_episode_bytes"],
                        "peak_gpu_mem_mib": job.peak_gpu_mem_mib,
                        "last_episode_time": progress["last_episode_time"],
                        "elapsed_sec": round(elapsed, 3) if elapsed is not None else "",
                        "reason": job.failure_reason,
                        "fatal_reason": job.fatal_reason,
                        "fatal_log_line": job.fatal_log_line,
                        "log_path": job.log_path.as_posix() if job.log_path else "",
                    }
                )


def write_run_report(runtime: ParallelRuntime) -> None:
    refresh_progress(runtime, force=True)
    write_job_summary_csv(runtime)
    with runtime.lock:
        jobs_payload = [runtime.job_status(job) for job in runtime.jobs]
        summary = runtime.summary_locked()
    payload = {
        "generated_at": utc_now(),
        "run_id": runtime.settings.run_id,
        "summary": summary,
        "jobs": jobs_payload,
        "episode_events_path": runtime.episode_events_path.as_posix(),
        "job_summary_csv_path": runtime.job_summary_csv_path.as_posix(),
        "deleted_failed_episodes_path": runtime.deleted_failed_path.as_posix(),
        "cache_cleanup": runtime.cache_cleanup_result,
    }
    runtime.report_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SimBox Parallel V2.6 Run Report",
        "",
        f"- Run id: `{runtime.settings.run_id}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Total jobs: `{summary['total_jobs']}`",
        f"- Target trajectories: `{summary['target_episodes']}`",
        f"- Generated trajectories: `{summary['generated_episodes']}`",
        f"- Successful trajectories: `{summary['successful_episodes']}`",
        f"- Failed trajectories: `{summary['failed_episodes']}`",
        f"- Episode success rate: `{summary['episode_success_rate']:.2f}%`",
        f"- Successful trajectory time: `{summary['success_trajectory_sec']:.3f}` sec (`{summary['success_trajectory_hours']:.6f}` hours)",
        f"- Successful action steps: `{summary['success_action_steps']}`",
        f"- Successful data bytes: `{summary['success_data_bytes']}` (`{human_bytes(summary['success_data_bytes'])}`)",
        f"- Failed data bytes: `{summary['failed_data_bytes']}` (`{human_bytes(summary['failed_data_bytes'])}`)",
        f"- Total data bytes: `{summary['total_data_bytes']}` (`{human_bytes(summary['total_data_bytes'])}`)",
        f"- Deleted failed episodes: `{summary['deleted_failed_episode_count']}`",
        f"- Freed failed episode bytes: `{summary['freed_failed_episode_bytes']}` (`{human_bytes(summary['freed_failed_episode_bytes'])}`)",
        f"- Startup hang: `{summary['startup_hang']}`",
        f"- Suspect hang: `{summary['suspect_hang']}`",
        f"- Episode events: `{runtime.episode_events_path.as_posix()}`",
        f"- Job summary CSV: `{runtime.job_summary_csv_path.as_posix()}`",
        f"- GPU samples CSV: `{runtime.gpu_samples_path.as_posix()}`",
        f"- Run directory: `{runtime.run_dir.as_posix()}`",
        "",
        "## Cache Cleanup",
        "",
        f"- Enabled: `{runtime.cache_cleanup.enabled}`",
        f"- Removed: `{runtime.cache_cleanup_result.get('removed', False)}`",
        f"- Cache path: `{runtime.cache_cleanup_result.get('path', '')}`",
        f"- Freed bytes: `{runtime.cache_cleanup_result.get('freed_bytes', 0)}` (`{human_bytes(runtime.cache_cleanup_result.get('freed_bytes', 0))}`)",
        f"- Error: `{runtime.cache_cleanup_result.get('error', '')}`",
        "",
        "## Jobs",
        "",
        "| Job | GPU | Worker | Status | Done/Target | Success | Failed | Rate | Traj Sec | Steps | Data | Peak GPU | Last Episode | Attempts | Reason | Log |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for job in runtime.jobs:
        progress = runtime.job_progress_locked(job)
        lines.append(
            f"| `{job.job_id}` | {job.gpu if job.gpu is not None else ''} | `{job.worker_name}` | "
            f"`{display_status(job)}` | {progress['generated']}/{progress['target']} | "
            f"{progress['success']} | {progress['failed']} | {progress['success_rate']:.2f}% | "
            f"{progress['success_trajectory_sec']:.3f} | {progress['success_action_steps']} | "
            f"{human_bytes(progress['success_data_bytes'])} | {job.peak_gpu_mem_mib}MiB | "
            f"`{progress['last_episode_time']}` | {job.attempt} | "
            f"`{job.failure_reason}` | `{job.log_path.as_posix() if job.log_path else ''}` |"
        )
        if job.status != "success" and job.log_path:
            excerpt = tail_lines(job.log_path, 25)
            if excerpt:
                lines.extend(["", f"<details><summary>{job.job_id} last log lines</summary>", "", "```text"])
                lines.extend(excerpt)
                lines.extend(["```", "</details>", ""])
    runtime.report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ffprobe_video_duration(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video_path.exists():
        return None
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
        video_path.as_posix(),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
        if result.returncode != 0:
            return None
        streams = json.loads(result.stdout).get("streams") or []
        stream = streams[0] if streams else {}
        if stream.get("duration") not in (None, "N/A"):
            return float(stream["duration"])
    except Exception:
        return None
    return None


def event_ffprobe_duration(event: dict[str, Any], repo_root: Path) -> float | None:
    episode_dirs = event.get("episode_dirs")
    if not isinstance(episode_dirs, list):
        return None
    durations: list[float] = []
    for episode_dir_s in episode_dirs:
        if not isinstance(episode_dir_s, str):
            continue
        episode_dir = resolve_event_episode_path(episode_dir_s, repo_root)
        for video_path in sorted(episode_dir.glob("images.rgb.*/demo.mp4")):
            duration = ffprobe_video_duration(video_path)
            if duration is not None and duration > 0:
                durations.append(duration)
    return max(durations) if durations else None


def final_ffprobe_verify(runtime: ParallelRuntime) -> None:
    if not runtime.progress.enabled or not runtime.progress.final_ffprobe_verify:
        return
    if not runtime.episode_events_path.exists():
        return
    by_job = {job.job_id: job for job in runtime.jobs}
    verified = 0
    adjusted = 0.0
    try:
        lines = runtime.episode_events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    with runtime.lock:
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "episode_saved" or event.get("status") != "success":
                continue
            job = by_job.get(str(event.get("launcher_job_id") or event.get("job_id") or ""))
            if job is None:
                continue
            verified_duration = event_ffprobe_duration(event, runtime.repo_root)
            if verified_duration is None:
                continue
            old_duration = float(event.get("trajectory_duration_sec") or 0.0)
            delta = verified_duration - old_duration
            if abs(delta) > 1e-6:
                job.success_trajectory_sec += delta
                adjusted += delta
            verified += 1
    runtime.append_manifest("ffprobe_verify_finish", verified_events=verified, trajectory_delta_sec=round(adjusted, 6))


def run_dataset_stats(runtime: ParallelRuntime) -> int:
    roots = sorted({job.dataset_root for job in runtime.jobs})
    status = 0
    for idx, root in enumerate(roots):
        stats_dir = runtime.run_dir / "dataset_stats" / safe_id(root)
        cmd = ["python3", "scripts/simbox/simbox_dataset_stats.py", root, "--output-dir", stats_dir.as_posix()]
        runtime.append_manifest("dataset_stats_start", dataset_root=root, output_dir=stats_dir.as_posix())
        result = subprocess.run(cmd, cwd=runtime.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (stats_dir / "dataset_stats_command.log").parent.mkdir(parents=True, exist_ok=True)
        (stats_dir / "dataset_stats_command.log").write_text(result.stdout, encoding="utf-8")
        runtime.append_manifest(
            "dataset_stats_finish",
            dataset_root=root,
            output_dir=stats_dir.as_posix(),
            exit_code=result.returncode,
        )
        if result.returncode != 0:
            status = result.returncode
    return status


def cleanup_run_cache(runtime: ParallelRuntime, *, run_failed: bool, interrupted: bool) -> None:
    settings = runtime.cache_cleanup
    result: dict[str, Any] = {
        "enabled": settings.enabled,
        "removed": False,
        "skipped": False,
        "path": "",
        "freed_bytes": 0,
        "error": "",
    }
    runtime.cache_cleanup_result = result
    if not settings.enabled:
        result["skipped"] = True
        result["reason"] = "disabled"
        return
    if interrupted and not settings.cleanup_on_interrupt:
        result["skipped"] = True
        result["reason"] = "interrupted"
        return
    if run_failed and not settings.cleanup_on_failure:
        result["skipped"] = True
        result["reason"] = "failure_policy"
        return
    if not run_failed and not settings.cleanup_on_success:
        result["skipped"] = True
        result["reason"] = "success_policy"
        return
    if settings.scope != "current_run":
        result["skipped"] = True
        result["reason"] = f"unsupported_scope:{settings.scope}"
        return

    cache_root = resolve_host_path(settings.root, runtime.repo_root)
    target = cache_root / runtime.settings.run_id
    result["path"] = target.as_posix()
    if not target.exists():
        result["skipped"] = True
        result["reason"] = "missing"
        runtime.append_manifest("cache_cleanup", **result)
        return
    try:
        target_resolved = target.resolve()
        repo_resolved = runtime.repo_root.resolve()
        if target_resolved == repo_resolved or target_resolved == target_resolved.parent:
            raise RuntimeError(f"refusing unsafe cache cleanup target: {target}")
        size_bytes = directory_size(target)
        shutil.rmtree(target)
        result["removed"] = True
        result["freed_bytes"] = size_bytes
        runtime.append_manifest("cache_cleanup", **result)
    except Exception as exc:  # noqa: BLE001 - cleanup should be reported, not mask the run result.
        result["error"] = f"{type(exc).__name__}: {exc}"
        runtime.append_manifest("cache_cleanup_failed", **result)


def preflight(runtime: ParallelRuntime) -> None:
    if runtime.settings.backend != "docker":
        raise ValueError("SimBox parallel v2 currently supports backend=docker")
    compose = resolve_host_path(runtime.settings.compose_file, runtime.repo_root)
    if not compose.is_file():
        raise FileNotFoundError(f"Compose file not found: {runtime.settings.compose_file}")
    if not runtime.settings.dry_run:
        if not shutil.which("docker"):
            raise RuntimeError("docker is not available")
        for cmd in (["docker", "compose", "version"], ["docker", "info"]):
            result = subprocess.run(cmd, cwd=runtime.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
        result = subprocess.run(
            ["docker", "image", "inspect", runtime.settings.isaac_image],
            cwd=runtime.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker image not found: {runtime.settings.isaac_image}. "
                f"Build with: docker compose -f {runtime.settings.compose_file} build isaac"
            )


def build_settings(
    config: dict[str, Any],
) -> tuple[
    ParallelSettings,
    StartupGuard,
    MonitorSettings,
    ProgressSettings,
    FailedEpisodeCleanupSettings,
    CacheCleanupSettings,
    FailureGuardSettings,
    GpuSamplingSettings,
]:
    parallel = config.get("parallel") or {}
    startup = config.get("startup_guard") or {}
    monitor = config.get("monitor") or {}
    progress = config.get("progress") or {}
    failed_cleanup = config.get("failed_episode_cleanup") or {}
    cache_cleanup = config.get("cache_cleanup") or {}
    failure_guard = config.get("failure_guard") or {}
    gpu_sampling = config.get("gpu_sampling") or {}
    run_id = str_or_none(parallel.get("run_id"))
    if not run_id:
        run_id = f"{safe_id(str(config.get('name') or 'simbox_parallel_v2'))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_id = safe_id(run_id)
    settings = ParallelSettings(
        backend=str(parallel.get("backend") or "docker"),
        gpus=list_of_ints(parallel.get("gpus", [0, 1, 2, 3]), name="parallel.gpus"),
        workers_per_gpu=int_value(parallel.get("workers_per_gpu"), default=1),
        run_id=run_id,
        compose_file=str(parallel.get("compose_file") or "docker/docker-compose.simbox.yml"),
        isaac_image=str(os.environ.get("INTERNDATA_ISAAC_IMAGE") or parallel.get("isaac_image") or "local/interdata-isaac-sim-4.1.0-curobo:latest"),
        task_timeout_sec=int_value(parallel.get("task_timeout_sec"), default=0),
        stats_after_run=bool_value(parallel.get("stats_after_run"), default=True),
        dry_run=bool_value(parallel.get("dry_run"), default=False),
        task_preflight=bool_value(parallel.get("task_preflight"), default=True),
    )
    if settings.workers_per_gpu < 1:
        raise ValueError("parallel.workers_per_gpu must be positive")
    if not settings.gpus:
        raise ValueError("parallel.gpus cannot be empty")
    startup_guard = StartupGuard(
        enabled=bool_value(startup.get("enabled"), default=True),
        marker=str(startup.get("marker") or "Simulation App Startup Complete"),
        timeout_sec=int(float_value(startup.get("timeout_min"), default=5.0) * 60),
        retry=int_value(startup.get("retry"), default=1),
    )
    monitor_settings = MonitorSettings(
        enabled=bool_value(monitor.get("enabled"), default=True),
        mode=str(monitor.get("mode") or "auto"),
        refresh_sec=float_value(monitor.get("refresh_sec"), default=2.0),
        silent_warn_sec=int_value(monitor.get("silent_warn_sec"), default=600),
        keep_finished_rows=int_value(monitor.get("keep_finished_rows"), default=20),
        theme=str(monitor.get("theme") or "nvtop_like"),
        compact_paths=bool_value(monitor.get("compact_paths"), default=True),
        show_gpu_panel=bool_value(monitor.get("show_gpu_panel"), default=True),
        show_data_panel=bool_value(monitor.get("show_data_panel"), default=True),
    )
    progress_settings = ProgressSettings(
        enabled=bool_value(progress.get("enabled"), default=True),
        mode=str(progress.get("mode") or "event_hook_first"),
        event_poll_interval_sec=float_value(progress.get("event_poll_interval_sec"), default=2.0),
        dataset_scan_interval_sec=float_value(progress.get("dataset_scan_interval_sec"), default=30.0),
        action_fps=float_value(progress.get("action_fps"), default=30.0),
        video_fps=float_value(progress.get("video_fps"), default=15.0),
        final_ffprobe_verify=bool_value(progress.get("final_ffprobe_verify"), default=True),
    )
    failed_cleanup_settings = FailedEpisodeCleanupSettings(
        enabled=bool_value(failed_cleanup.get("enabled"), default=True),
        mode=str(failed_cleanup.get("mode") or "conservative"),
        require_finalized_event=bool_value(failed_cleanup.get("require_finalized_event"), default=True),
        require_run_time_window=bool_value(failed_cleanup.get("require_run_time_window"), default=True),
        delete_dirs=bool_value(failed_cleanup.get("delete_dirs"), default=True),
        keep_summary=bool_value(failed_cleanup.get("keep_summary"), default=True),
    )
    cache_cleanup_settings = CacheCleanupSettings(
        enabled=bool_value(cache_cleanup.get("enabled"), default=True),
        scope=str(cache_cleanup.get("scope") or "current_run"),
        root=str(cache_cleanup.get("root") or ".docker/isaac-sim"),
        cleanup_on_success=bool_value(cache_cleanup.get("cleanup_on_success"), default=True),
        cleanup_on_failure=bool_value(cache_cleanup.get("cleanup_on_failure"), default=True),
        cleanup_on_interrupt=bool_value(cache_cleanup.get("cleanup_on_interrupt"), default=True),
        keep_summary=bool_value(cache_cleanup.get("keep_summary"), default=True),
    )
    failure_guard_settings = FailureGuardSettings(
        enabled=bool_value(failure_guard.get("enabled"), default=True),
        kill_on_fatal_log=bool_value(failure_guard.get("kill_on_fatal_log"), default=True),
        kill_on_suspect_hang=bool_value(failure_guard.get("kill_on_suspect_hang"), default=True),
        suspect_hang_kill_sec=int_value(failure_guard.get("suspect_hang_kill_sec"), default=1800),
    )
    gpu_sampling_settings = GpuSamplingSettings(
        enabled=bool_value(gpu_sampling.get("enabled"), default=True),
        interval_sec=float_value(gpu_sampling.get("interval_sec"), default=10.0),
        output=str(gpu_sampling.get("output") or "gpu_samples.csv"),
    )
    return (
        settings,
        startup_guard,
        monitor_settings,
        progress_settings,
        failed_cleanup_settings,
        cache_cleanup_settings,
        failure_guard_settings,
        gpu_sampling_settings,
    )


def parse_iso_time(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def stop_run_containers_by_name(runtime: ParallelRuntime) -> None:
    if not shutil.which("docker"):
        return
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
        cwd=runtime.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        cid, name = parts
        if f"interdata-{runtime.settings.run_id}-" in name:
            subprocess.run(["docker", "rm", "-f", cid], cwd=runtime.repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            runtime.append_manifest("recover_container_removed", container_id=cid, container_name=name)


def restore_runtime_from_status(runtime: ParallelRuntime) -> None:
    if not runtime.status_path.exists():
        return
    try:
        payload = json.loads(runtime.status_path.read_text(encoding="utf-8"))
    except Exception:
        return
    jobs_by_idx = {job.queue_idx: job for job in runtime.jobs}
    for item in payload.get("jobs", []):
        if not isinstance(item, dict):
            continue
        job = jobs_by_idx.get(int(item.get("queue_idx", -1)))
        if job is None:
            continue
        job.gpu = item.get("gpu")
        job.worker_name = str(item.get("worker") or job.worker_name)
        job.status = str(item.get("status") or job.status)
        job.container_id = str(item.get("container_id") or "")
        job.container_name = str(item.get("container_name") or "")
        job.exit_code = item.get("exit_code")
        job.failure_reason = str(item.get("failure_reason") or "")
        job.attempt = int(item.get("attempt") or job.attempt or 0)
        job.task_dir = Path(str(item.get("task_dir") or job.task_dir))
        job.docker_start_err = job.task_dir / "docker_start.err"
        log_path = str(item.get("log_path") or "")
        if log_path:
            job.log_path = Path(log_path)
        event_path = str(item.get("episode_event_path") or "")
        if event_path:
            job.episode_event_path = Path(event_path)
        progress = item.get("progress") if isinstance(item.get("progress"), dict) else {}
        job.generated_count = int(progress.get("generated") or 0)
        job.success_count = int(progress.get("success") or 0)
        job.failed_count = int(progress.get("failed") or 0)
        job.success_trajectory_sec = float(progress.get("success_trajectory_sec") or 0.0)
        job.success_action_steps = int(progress.get("success_action_steps") or 0)
        job.success_action_duration_sec = float(progress.get("success_action_duration_sec") or 0.0)
        job.success_data_bytes = int(progress.get("success_data_bytes") or 0)
        job.failed_data_bytes = int(progress.get("failed_data_bytes") or 0)
        job.deleted_failed_episode_count = int(progress.get("deleted_failed_episode_count") or 0)
        job.freed_failed_episode_bytes = int(progress.get("freed_failed_episode_bytes") or 0)
        job.last_episode_time = str(progress.get("last_episode_time") or "")
        if job.log_path and job.docker_start_err:
            reason = classify_failure(job.log_path, job.docker_start_err, job.status, job.exit_code)
            if reason != "unknown_failure":
                job.failure_reason = reason
            fatal_reason, fatal_line = detect_fatal_log(job)
            if fatal_reason:
                job.fatal_detected = True
                job.fatal_reason = fatal_reason
                job.fatal_log_line = fatal_line


def mark_unfinished_jobs(runtime: ParallelRuntime, reason: str) -> None:
    now = time.time()
    with runtime.lock:
        for job in runtime.jobs:
            if job.status in {"starting", "running", "retrying"}:
                job.status = "failed"
                job.exit_code = FATAL_GUARD_EXIT_CODE
                if not job.failure_reason:
                    job.failure_reason = reason
                job.end_time = now
                runtime.append_manifest(
                    "unfinished_job_marked_failed",
                    queue_idx=job.queue_idx,
                    job_id=job.job_id,
                    reason=job.failure_reason,
                )


def finalize_run(
    runtime: ParallelRuntime,
    *,
    run_failed: bool,
    interrupted: bool,
    run_dataset_stats_enabled: bool,
) -> int:
    runtime.stop_event.set()
    runtime.stop_containers()
    if interrupted:
        mark_unfinished_jobs(runtime, "interrupted")
    refresh_progress(runtime, force=True)
    refresh_gpu_stats(runtime, force=True)
    stats_status = 0
    if run_dataset_stats_enabled and runtime.settings.stats_after_run and not runtime.settings.dry_run:
        stats_status = run_dataset_stats(runtime)
    final_ffprobe_verify(runtime)
    has_unsuccessful_job = any(job.status not in {"success", "pending"} for job in runtime.jobs)
    cleanup_run_cache(runtime, run_failed=run_failed or has_unsuccessful_job or stats_status != 0, interrupted=interrupted)
    write_run_report(runtime)
    runtime.write_status()
    return 0 if not run_failed and not has_unsuccessful_job and stats_status == 0 else 1


def run_parallel_config(config_path: str, extras: list[str] | None = None, recover_run_id: str | None = None) -> int:
    repo_root = repo_root_from_here()
    config_file = resolve_host_path(config_path, repo_root)
    config = load_yaml(config_file)
    if extras:
        apply_cli_overrides(config, extras)
    if recover_run_id:
        config.setdefault("parallel", {})["run_id"] = recover_run_id
    settings, startup_guard, monitor, progress, failed_cleanup, cache_cleanup, failure_guard, gpu_sampling = build_settings(config)
    run_dir = repo_root / "output" / "_parallel_runs" / settings.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_start_unix = time.time()
    manifest_path = run_dir / "manifest.jsonl"
    if recover_run_id and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "run_start":
                run_start_unix = parse_iso_time(str(record.get("time") or "")) or run_start_unix
                break
    runtime = ParallelRuntime(
        repo_root=repo_root,
        run_dir=run_dir,
        settings=settings,
        startup_guard=startup_guard,
        monitor=monitor,
        progress=progress,
        failed_cleanup=failed_cleanup,
        cache_cleanup=cache_cleanup,
        failure_guard=failure_guard,
        gpu_sampling=gpu_sampling,
        jobs=[],
        run_start_unix=run_start_unix,
    )
    if not recover_run_id:
        runtime.manifest_path.write_text("", encoding="utf-8")
        runtime.failure_path.write_text("", encoding="utf-8")
        runtime.episode_events_path.write_text("", encoding="utf-8")
        runtime.deleted_failed_path.write_text("", encoding="utf-8")
        runtime.gpu_samples_path.write_text("", encoding="utf-8")
    try:
        jobs = build_configured_jobs(
            config=config,
            repo_root=repo_root,
            run_dir=run_dir,
            settings=settings,
            startup_guard=startup_guard,
        )
        runtime.jobs = jobs
        if recover_run_id:
            restore_runtime_from_status(runtime)
            runtime.append_manifest("recover_start", run_id=settings.run_id, config_path=rel_repo_path(config_file, repo_root))
            stop_run_containers_by_name(runtime)
            mark_unfinished_jobs(runtime, "recovered_stale_container")
            original_status = finalize_run(runtime, run_failed=True, interrupted=True, run_dataset_stats_enabled=False)
            runtime.append_manifest("recover_finish", run_id=settings.run_id, original_exit_code=original_status, exit_code=0)
            print(f"Recovered run report: {runtime.report_md_path.relative_to(repo_root).as_posix()}")
            return 0
        runtime.append_manifest(
            "run_start",
            run_id=settings.run_id,
            config_path=rel_repo_path(config_file, repo_root),
            backend=settings.backend,
            gpus=settings.gpus,
            workers_per_gpu=settings.workers_per_gpu,
            total_jobs=len(jobs),
            dry_run=settings.dry_run,
            progress=progress.__dict__,
            failed_episode_cleanup=failed_cleanup.__dict__,
            cache_cleanup=cache_cleanup.__dict__,
            failure_guard=failure_guard.__dict__,
            gpu_sampling=gpu_sampling.__dict__,
        )
        preflight(runtime)
        runtime.write_status()
        print(f"Run id: {settings.run_id}")
        print(f"Backend: {settings.backend}")
        print(f"GPUs: {settings.gpus}")
        print(f"Workers per GPU: {settings.workers_per_gpu}")
        print(f"Queued jobs: {len(jobs)}")
        print(f"Run records: {run_dir.relative_to(repo_root).as_posix()}")

        monitor_thread = threading.Thread(target=monitor_loop, args=(runtime,), daemon=True)
        if monitor.enabled:
            monitor_thread.start()
        worker_threads = []
        for gpu in settings.gpus:
            for worker_idx in range(settings.workers_per_gpu):
                thread = threading.Thread(target=worker_loop, args=(runtime, gpu, worker_idx), daemon=False)
                worker_threads.append(thread)
                thread.start()
        for thread in worker_threads:
            thread.join()
        runtime.stop_event.set()
        if monitor.enabled:
            monitor_thread.join(timeout=max(2, monitor.refresh_sec + 1))
        has_unsuccessful_job = any(job.status != "success" for job in runtime.jobs)
        overall_status = finalize_run(
            runtime,
            run_failed=has_unsuccessful_job,
            interrupted=False,
            run_dataset_stats_enabled=True,
        )
        runtime.append_manifest("run_finish", run_id=settings.run_id, exit_code=overall_status)
        print(f"Run report: {runtime.report_md_path.relative_to(repo_root).as_posix()}")
        return overall_status
    except KeyboardInterrupt:
        runtime.stop_event.set()
        runtime.stop_containers()
        finalize_run(runtime, run_failed=True, interrupted=True, run_dataset_stats_enabled=True)
        runtime.append_manifest("run_interrupted", run_id=settings.run_id)
        print(f"Interrupted run report: {runtime.report_md_path.relative_to(repo_root).as_posix()}")
        return 130
    except Exception as exc:  # noqa: BLE001 - write report before returning failure.
        runtime.stop_event.set()
        runtime.stop_containers()
        runtime.append_manifest("run_error", run_id=settings.run_id, error=f"{type(exc).__name__}: {exc}")
        error_path = run_dir / "run_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        finalize_run(runtime, run_failed=True, interrupted=False, run_dataset_stats_enabled=True)
        print(f"SimBox parallel v2 failed: {exc}", file=sys.stderr)
        print(f"Error report: {error_path}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    def _handle_stop_signal(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recover-run-id", help="Finalize and clean up an existing run id without starting new jobs.")
    args, extras = parser.parse_known_args(argv)
    return run_parallel_config(args.config, extras, recover_run_id=args.recover_run_id)


if __name__ == "__main__":
    raise SystemExit(main())
