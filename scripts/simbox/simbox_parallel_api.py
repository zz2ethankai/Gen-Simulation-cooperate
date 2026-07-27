#!/usr/bin/env python3
"""Small Python API for SimBox parallel generation.

The shell launcher remains the operational entrypoint. This module provides a
stable wrapper for other code to call without knowing the CLI details.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SimBoxParallelGenerateRequest:
    task_source: str
    backend: str = "docker"
    gpus: str | Sequence[int | str] = "0,1,2,3"
    workers_per_gpu: int = 1
    random_num: int = 10
    split_random_num: bool = False
    random_seed_base: int | None = None
    scene_info: str | None = None
    de_config: str = "configs/simbox/de_plan_with_render_template.yaml"
    dataset_root: str | None = None
    run_id: str | None = None
    compose_file: str = "docker/docker-compose.simbox.yml"
    isaac_python: str = "/home/bld/ykqin/isaacsim/python.sh"
    estimate_mem_gb: int = 16
    min_free_mem_gb: int = 12
    max_gpu_util: int = 70
    task_timeout_sec: int = 0
    stats_after_run: bool = True
    dry_run: bool = False
    repo_root: str | Path | None = None
    script_path: str = "scripts/simbox/simbox_parallel_generate.sh"
    extra_args: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimBoxParallelGenerateResult:
    exit_code: int
    run_id: str
    run_dir: Path
    manifest_path: Path
    failure_path: Path
    stats_dir: Path
    command: tuple[str, ...]
    stdout: str
    stderr: str


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "run"


def _default_run_id() -> str:
    return f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _gpu_list(value: str | Sequence[int | str]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _coerce_request(request: SimBoxParallelGenerateRequest | Mapping[str, Any]) -> SimBoxParallelGenerateRequest:
    if isinstance(request, SimBoxParallelGenerateRequest):
        return request
    return SimBoxParallelGenerateRequest(**dict(request))


def build_simbox_parallel_generate_command(
    request: SimBoxParallelGenerateRequest | Mapping[str, Any],
) -> tuple[tuple[str, ...], Path, str]:
    req = _coerce_request(request)
    repo_root = Path(req.repo_root).expanduser().resolve() if req.repo_root else _repo_root_from_here()
    script_path = Path(req.script_path)
    if not script_path.is_absolute():
        script_path = repo_root / script_path

    run_id = _safe_run_id(req.run_id or _default_run_id())
    cmd: list[str] = [
        "bash",
        str(script_path),
        "--backend",
        req.backend,
        "--gpus",
        _gpu_list(req.gpus),
        "--workers-per-gpu",
        str(req.workers_per_gpu),
        "--random-num",
        str(req.random_num),
        "--de-config",
        req.de_config,
        "--run-id",
        run_id,
        "--compose-file",
        req.compose_file,
        "--isaac-python",
        req.isaac_python,
        "--estimate-mem-gb",
        str(req.estimate_mem_gb),
        "--min-free-mem-gb",
        str(req.min_free_mem_gb),
        "--max-gpu-util",
        str(req.max_gpu_util),
        "--task-timeout-sec",
        str(req.task_timeout_sec),
    ]
    if req.random_seed_base is not None:
        cmd.extend(["--random-seed-base", str(req.random_seed_base)])
    if req.split_random_num:
        cmd.append("--split-random-num")
    if req.scene_info:
        cmd.extend(["--scene-info", req.scene_info])
    if req.dataset_root:
        cmd.extend(["--dataset-root", req.dataset_root])
    if req.stats_after_run:
        cmd.append("--stats-after-run")
    else:
        cmd.append("--no-stats-after-run")
    if req.dry_run:
        cmd.append("--dry-run")
    cmd.extend(req.extra_args)
    cmd.append(req.task_source)
    return tuple(cmd), repo_root, run_id


def run_simbox_parallel_generate(
    request: SimBoxParallelGenerateRequest | Mapping[str, Any],
    *,
    check: bool = False,
    capture_output: bool = True,
) -> SimBoxParallelGenerateResult:
    cmd, repo_root, run_id = build_simbox_parallel_generate_command(request)
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        check=False,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    run_dir = repo_root / "output" / "_parallel_runs" / run_id
    return SimBoxParallelGenerateResult(
        exit_code=completed.returncode,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.jsonl",
        failure_path=run_dir / "failures.tsv",
        stats_dir=run_dir / "dataset_stats",
        command=cmd,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


__all__ = [
    "SimBoxParallelGenerateRequest",
    "SimBoxParallelGenerateResult",
    "build_simbox_parallel_generate_command",
    "run_simbox_parallel_generate",
]
