#!/usr/bin/env python3
"""Build a container-safe launch contract for one SimBox task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = Path("/workspace")


def _repo_path(raw: str, *, must_exist: bool, kind: str) -> tuple[Path, str]:
    value = Path(raw)
    host_path = (value if value.is_absolute() else REPO_ROOT / value).resolve()
    try:
        relative = host_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{kind} must be inside the repository: {host_path}") from exc
    if must_exist and not host_path.is_file():
        raise FileNotFoundError(f"{kind} not found: {host_path}")
    if not must_exist:
        host_path.parent.mkdir(parents=True, exist_ok=True)
    return host_path, str(CONTAINER_ROOT / relative)


def _optional_path(raw: str, *, directory: bool, kind: str) -> tuple[str, str]:
    if not raw:
        return "", ""
    host_path, container_path = _repo_path(raw, must_exist=False, kind=kind)
    if directory:
        host_path.mkdir(parents=True, exist_ok=True)
    return str(host_path), container_path


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    task_host, task_container = _repo_path(
        args.task_config,
        must_exist=True,
        kind="task config",
    )
    launcher_host, launcher_container = _repo_path(
        args.launcher_config,
        must_exist=True,
        kind="launcher config",
    )
    output_host, output_container = _optional_path(
        args.output_dir,
        directory=True,
        kind="output directory",
    )
    seq_host, seq_container = _optional_path(
        args.seq_output_dir,
        directory=True,
        kind="sequence output directory",
    )
    event_host, event_container = _optional_path(
        args.episode_event_path,
        directory=False,
        kind="episode event path",
    )
    debug_host, debug_container = _optional_path(
        args.debug_output_dir,
        directory=True,
        kind="debug output directory",
    )
    launcher_args = [
        f"--name={args.run_name}",
        f"--random_seed={args.random_seed}",
        f"--load_stage.scene_loader.args.cfg_path={task_container}",
        "--load_stage.scene_loader.args.simulator.active_gpu=0",
        "--load_stage.scene_loader.args.simulator.physics_gpu=0",
        f"--load_stage.layout_random_generator.args.random_num={args.random_num}",
    ]
    if output_container:
        launcher_args.append(f"--store_stage.writer.args.output_dir={output_container}")
    if seq_container:
        launcher_args.append(f"--store_stage.writer.args.seq_output_dir={seq_container}")
    if any("\n" in item or "\r" in item for item in launcher_args):
        raise ValueError("launcher arguments may not contain newlines")
    return {
        "task_host": str(task_host),
        "task_container": task_container,
        "launcher_host": str(launcher_host),
        "launcher_container": launcher_container,
        "launcher_args": launcher_args,
        "output_host": output_host,
        "output_container": output_container,
        "seq_output_host": seq_host,
        "seq_output_container": seq_container,
        "episode_event_host": event_host,
        "episode_event_container": event_container,
        "debug_output_host": debug_host,
        "debug_output_container": debug_container,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--launcher-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--random-num", required=True, type=int)
    parser.add_argument("--random-seed", required=True, type=int)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seq-output-dir", default="")
    parser.add_argument("--episode-event-path", default="")
    parser.add_argument("--debug-output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.random_num <= 0:
        raise ValueError("random-num must be positive")
    print(json.dumps(build_contract(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
