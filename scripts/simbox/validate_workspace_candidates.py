#!/usr/bin/env python3
"""Validate workspace candidates with CuRobo probes and bounded Pick runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = Path("/workspace")
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.workspace_planner import (  # noqa: E402
    CuroboCandidateResult,
    PickAttemptResult,
    compile_pick_place_probe_task,
    compile_pick_task,
    compile_probe_task,
    dump_json,
)


CANDIDATE_INDEPENDENT_FAILURES = {
    "ATTACH_PRIM_NOT_IN_CUROBO_WORLD",
    "ATTACH_COLLISION_CONFIG_MISSING",
    "ATTACH_COLLISION_CONFIG_INVALID",
    "ATTACH_COLLISION_CONFIG_CONFLICT",
    "ATTACH_COLLISION_PRIM_NOT_FOUND",
    "ATTACH_COLLISION_PRIM_NOT_COLLIDABLE",
    "ATTACH_COLLISION_PRIM_OUTSIDE_RIGID_ROOT",
    "ATTACH_COLLISION_PRIM_AMBIGUOUS",
    "ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD",
    "NO_GRASP_CANDIDATE",
}
PROBE_INFRASTRUCTURE_FAILURES = {"PROBE_RESULT_MISSING", "PROBE_TIMEOUT"}


def _progress(message: str) -> None:
    print(f"[workspace-validator] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CuRobo and Pick validation for a workspace manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU indices")
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        help="Require this task arm; omit only for legacy dual-arm probes.",
    )
    parser.add_argument(
        "--planning-config",
        type=Path,
        help="YAML containing common_debug.planning or a top-level planning mapping.",
    )
    parser.add_argument(
        "--attach-prim-path-child",
        action="append",
        default=[],
        help="Exact carried-object collision Prim child; may be repeated.",
    )
    parser.add_argument(
        "--max-probe-candidates",
        type=int,
        default=0,
        help="Geometry-safe shortlist budget; 0 probes every feasible candidate.",
    )
    parser.add_argument(
        "--manipulation-preferred-radius-m",
        type=float,
        default=0.65,
        help="Only ranks the shortlist; it never bypasses geometry safety.",
    )
    parser.add_argument("--max-pick-candidates", type=int, default=3)
    parser.add_argument("--probe-timeout-sec", type=int, default=900)
    parser.add_argument("--place-probe-timeout-sec", type=int, default=1200)
    parser.add_argument("--pick-timeout-sec", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--simulator-backend",
        choices=("docker", "conda"),
        default="docker",
        help="Simulator launcher used for every probe and Pick attempt.",
    )
    parser.add_argument("--conda-env", default="interndata-isaac6")

    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Probe only the selected geometry candidate ID; may be repeated.",
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help="Stop after the selected planning gate; never execute Place or record an episode.",
    )
    parser.add_argument(
        "--planning-gate",
        choices=("pick", "pick-place"),
        default="pick",
        help=(
            "pick preserves the planning-only CLI behavior; pick-place additionally "
            "executes the real Pick and plans Place from its attached state."
        ),
    )
    parser.add_argument(
        "--diagnostic-disable-curobo-obstacle-path",
        action="append",
        default=[],
        help=(
            "Planning-only diagnostic: temporarily disable one exact CuRobo obstacle Prim "
            "path during PickPlanProbe; may be repeated. PhysX is unchanged."
        ),
    )
    parser.add_argument(
        "--diagnostic-disable-physics-and-curobo-obstacle-path",
        action="append",
        default=[],
        help=(
            "Planning-only diagnostic: temporarily disable one exact collider in both "
            "PhysX and CuRobo; may be repeated and is always restored."
        ),
    )
    parser.add_argument(
        "--diagnostic-disable-collision-entity",
        action="append",
        default=[],
        help=(
            "Planning-only diagnostic: resolve a registered fixture/object name to all "
            "of its exact colliders and temporarily disable them in PhysX and CuRobo."
        ),
    )
    parser.add_argument(
        "--diagnostic-collision-world",
        choices=("full", "target-only", "empty"),
        default="full",
        help=(
            "Planning-only isolation mode: full keeps the normal CuRobo world, "
            "target-only removes every obstacle except the target, and empty removes "
            "all world obstacles."
        ),
    )
    parser.add_argument(
        "--stop-after-feasible",
        action="store_true",
        help="Stop scheduling new candidates once any runtime Probe is feasible.",
    )
    parser.add_argument(
        "--capture-overview",
        action="store_true",
        help="Capture an independent overview before the planning result is published.",
    )
    parser.add_argument(
        "--capture-trajectory",
        action="store_true",
        help="Export the selected pregrasp and grasp paths as trajectory_debug.usda.",
    )
    parser.add_argument("--camera-eye", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--camera-target", nargs=3, type=float, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--camera-template",
        choices=("robot_target_overhead_v1", "robot_target_diagonal_v1"),
        default="robot_target_overhead_v1",
    )
    parser.add_argument("--camera-template-params-json", default="{}")
    parser.add_argument(
        "--camera-resolution",
        nargs=2,
        type=int,
        default=(1280, 960),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--camera-focal-length-mm", type=float, default=16.0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _candidate_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["candidate_id"]): item for item in manifest.get("geometry_candidates", [])}


def _summarize_probe_failures(
    probe_rows: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    arm_rows = [
        arm
        for row in probe_rows
        for arm in row.get("arms", {}).values()
    ]
    unstable_count = sum(
        str(value.get("failure_code")) == "PROBE_SPAWN_UNSTABLE"
        for value in arm_rows
    )
    stable_count = sum(
        bool((value.get("spawn_check") or {}).get("stable"))
        for value in arm_rows
    )
    failure_counts: dict[str, int] = {}
    for value in arm_rows:
        code = str(value.get("failure_code") or "UNKNOWN_FAILURE")
        failure_counts[code] = failure_counts.get(code, 0) + 1
    summary = {
        "probed_candidate_count": len(probe_rows),
        "spawn_stable_count": stable_count,
        "spawn_unstable_count": unstable_count,
        "failure_counts": dict(sorted(failure_counts.items())),
    }
    if arm_rows and unstable_count == len(arm_rows):
        return "spawn_unstable", "PROBE_SPAWN_UNSTABLE", summary
    return "no_safe_reachable_pose", "NO_CUROBO_CANDIDATE", summary


def _load_planning_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"planning config must be a mapping: {path}")
    planning = (value.get("common_debug") or {}).get("planning")
    if planning is None:
        planning = value.get("planning")
    if not isinstance(planning, dict):
        raise ValueError(
            f"planning config must contain common_debug.planning or planning: {path}"
        )
    return planning


def _stop_process_group(process: subprocess.Popen[Any], timeout_sec: float = 30.0) -> int | None:
    """Stop the whole launcher/Isaac process tree, not only its shell parent."""

    if process.poll() is not None:
        return process.wait()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=timeout_sec)


def _stack_id(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_prefix = "".join(char if char.isalnum() or char in "_.-" else "-" for char in prefix)
    return f"{safe_prefix.strip('-') or 'workspace'}-{digest}"


def _runner_command(simulator_backend: str) -> list[str]:
    if simulator_backend == "docker":
        return ["bash", "scripts/docker/up_simbox_isaac.sh"]
    if simulator_backend == "conda":
        return ["bash", "scripts/simbox/run_simbox_task.sh"]
    raise ValueError(f"unsupported simulator backend: {simulator_backend}")


def _shortlist_candidates(
    candidates: list[dict[str, Any]],
    preferred_radius_m: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank close manipulation stances while retaining approach-side diversity."""

    if not math.isfinite(preferred_radius_m) or preferred_radius_m <= 0.0:
        raise ValueError("manipulation preferred radius must be finite and positive")
    if limit < 0:
        raise ValueError("max probe candidates must be non-negative")
    buckets: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        angle = float(candidate.get("angle_deg", 0.0)) % 360.0
        buckets.setdefault(int(angle // 45.0), []).append(candidate)
    for values in buckets.values():
        values.sort(
            key=lambda item: (
                abs(float(item["radius_m"]) - preferred_radius_m),
                float(item["radius_m"]),
                abs(float(item.get("yaw_offset_deg", 0.0))),
                str(item["candidate_id"]),
            )
        )
    ordered: list[dict[str, Any]] = []
    depth = 0
    while True:
        appended = False
        for bucket in sorted(buckets):
            values = buckets[bucket]
            if depth < len(values):
                ordered.append(values[depth])
                appended = True
        if not appended:
            break
        depth += 1
    if limit:
        return ordered[:limit]
    return ordered


def _run_probe(
    candidate: dict[str, Any],
    gpu: str,
    source_task: Path,
    target: str,
    run_root: Path,
    timeout: int,
    conda_env: str,
    simulator_backend: str,
    seed: int,

    required_arm: str | None,
    planning: dict[str, Any] | None,
    attach_prim_path_children: list[str],
    expected_target_world_xyz: list[float] | None,
    diagnostic_disable_curobo_obstacle_paths: list[str],
    diagnostic_disable_physics_and_curobo_obstacle_paths: list[str],
    diagnostic_disable_collision_entities: list[str],
    diagnostic_collision_world: str,
    diagnostic_capture: dict[str, Any] | None = None,
    spawn_settle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    case_dir = run_root / "probes" / candidate_id / f"seed_{seed}"
    result_dir = case_dir / "results"
    task_path = case_dir / "probe_task.yaml"
    case_dir.mkdir(parents=True, exist_ok=True)
    compile_probe_task(
        source_task,
        candidate,
        target,
        task_path,
        result_dir,
        arm=required_arm,
        planning=planning,
        attach_prim_path_children=attach_prim_path_children,
        expected_target_world_xyz=expected_target_world_xyz,
        spawn_settle=spawn_settle,
        diagnostic_disable_curobo_obstacle_paths=(
            diagnostic_disable_curobo_obstacle_paths
        ),
        diagnostic_disable_physics_and_curobo_obstacle_paths=(
            diagnostic_disable_physics_and_curobo_obstacle_paths
        ),
        diagnostic_disable_collision_entities=diagnostic_disable_collision_entities,
        diagnostic_collision_world=diagnostic_collision_world,
        diagnostic_capture=(
            {**diagnostic_capture, "output_dir": str((case_dir / "diagnostics").resolve())}
            if diagnostic_capture
            else None
        ),
    )
    arms_to_probe = (required_arm,) if required_arm is not None else ("left", "right")
    result_paths = {
        arm: result_dir / f"{candidate_id}.{arm}.json" for arm in arms_to_probe
    }
    for path in result_paths.values():
        path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TASK_CONFIG": str(task_path),
            "LAUNCH_TEMPLATE": "configs/simbox/de_workspace_probe_template.yaml",
            "GPU_ID": gpu,
            "RANDOM_NUM": "1",
            "RANDOM_SEED": str(seed),
            "RUN_NAME": f"workspace_probe/{candidate_id}/seed_{seed}",
            "OUTPUT_DIR": "",
            "SEQ_OUTPUT_DIR": str(case_dir / "unused_seq"),
            "INTERNDATA_STACK_ID": _stack_id(f"workspace-probe-{candidate_id}", case_dir),
            "INTERNDATA_DOCKER_METADATA_PATH": str(case_dir / "docker_runtime.json"),
            "SIMBOX_DEBUG_OUTPUT_DIR": str(case_dir / "simbox_debug"),
            "INTERNDATA_RANDOM_SEED": str(seed),
            "INTERNDATA_GPU": gpu,
            "INTERNDATA_SIMULATOR_BACKEND": simulator_backend,
            "CONDA_ENV": conda_env,
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = case_dir / "stdout.log"
    timed_out = False
    terminated_after_results = False
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            _runner_command(simulator_backend),
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if all(path.is_file() for path in result_paths.values()):
                terminated_after_results = True
                return_code = _stop_process_group(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                return_code = _stop_process_group(process)
                break
            time.sleep(0.25)
        else:
            return_code = process.wait()

    arms: dict[str, dict[str, Any]] = {}
    for arm in arms_to_probe:
        path = result_paths[arm]
        arms[arm] = _load_json(path) if path.is_file() else {
            "feasible": False,
            "arm": arm,
            "failure_code": "PROBE_TIMEOUT" if timed_out else "PROBE_RESULT_MISSING",
        }
        arms[arm]["seed"] = seed
        arms[arm]["artifact"] = str(path)
        if path.is_file():
            dump_json(arms[arm], path)
    feasible_arms = [value for value in arms.values() if value.get("feasible")]
    selected_arm: dict[str, Any] | None = None
    if feasible_arms:
        selected_arm = min(
            feasible_arms,
            key=lambda value: (
                -int(value.get("joint_success_count", 0)),
                float(value.get("selected_grasp_score"))
                if value.get("selected_grasp_score") is not None
                else math.inf,
                str(value.get("arm", "")),
            ),
        )
    row = CuroboCandidateResult(
        candidate_id=candidate_id,
        gpu=gpu,
        return_code=return_code,
        timed_out=timed_out,
        results_complete=all(path.is_file() for path in result_paths.values()),
        terminated_after_results=terminated_after_results,
        arms=arms,
        feasible=selected_arm is not None,
        selected_arm=selected_arm.get("arm") if selected_arm else None,
        joint_success_count=int(selected_arm.get("joint_success_count", 0)) if selected_arm else 0,
        selected_grasp_score=selected_arm.get("selected_grasp_score") if selected_arm else None,
        log=str(log_path),
    ).to_dict()
    row["seed"] = seed
    return row


def run_pick_place_planning_probe(
    candidate: dict[str, Any],
    arm: str,
    gpu: str,
    source_task: Path,
    target: str,
    run_root: Path,
    timeout: int,
    conda_env: str,
    simulator_backend: str,
    planning: dict[str, Any] | None,
    attach_prim_path_children: list[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    case_dir = run_root / "place_probes" / candidate_id / f"seed_{seed}"
    result_path = case_dir / "results" / f"{candidate_id}.{arm}.json"
    task_path = case_dir / "pick_place_probe_task.yaml"
    case_dir.mkdir(parents=True, exist_ok=True)
    compile_pick_place_probe_task(
        source_task,
        candidate,
        target,
        arm,
        task_path,
        result_path,
        planning=planning,
        attach_prim_path_children=attach_prim_path_children,
    )
    result_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TASK_CONFIG": str(task_path),
            "LAUNCH_TEMPLATE": "configs/simbox/de_workspace_probe_template.yaml",
            "GPU_ID": gpu,
            "RANDOM_NUM": "1",
            "RANDOM_SEED": str(seed),
            "RUN_NAME": f"workspace_place_probe/{candidate_id}/seed_{seed}",
            "OUTPUT_DIR": "",
            "SEQ_OUTPUT_DIR": str(case_dir / "unused_seq"),
            "INTERNDATA_RANDOM_SEED": str(seed),
            "INTERNDATA_GPU": gpu,
            "INTERNDATA_SIMULATOR_BACKEND": simulator_backend,
            "CONDA_ENV": conda_env,
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = case_dir / "stdout.log"
    timed_out = False
    terminated_after_result = False
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            _runner_command(simulator_backend),
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if result_path.is_file():
                terminated_after_result = True
                return_code = _stop_process_group(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                return_code = _stop_process_group(process)
                break
            time.sleep(0.25)
        else:
            return_code = process.wait()

    result = _load_json(result_path) if result_path.is_file() else {
        "feasible": False,
        "arm": arm,
        "failure_code": (
            "PLACE_PROBE_TIMEOUT" if timed_out else "PLACE_PROBE_RESULT_MISSING"
        ),
    }
    result["seed"] = seed
    if result_path.is_file():
        dump_json(result, result_path)
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "seed": seed,
        "gpu": gpu,
        "return_code": return_code,
        "timed_out": timed_out,
        "results_complete": result_path.is_file(),
        "terminated_after_result": terminated_after_result,
        "feasible": bool(result.get("feasible")),
        "result": result,
        "artifact": str(result_path),
        "log": str(log_path),
    }


def _run_probe_queue(
    queue: list[dict[str, Any]],
    gpu: str,
    source_task: Path,
    target: str,
    run_root: Path,
    timeout: int,
    conda_env: str,
    simulator_backend: str,
    seed: int,

    stop_event: threading.Event,
    stop_after_feasible: bool,
    required_arm: str | None,
    planning: dict[str, Any] | None,
    attach_prim_path_children: list[str],
    expected_target_world_xyz: list[float] | None,
    spawn_settle: dict[str, Any] | None,
    diagnostic_disable_curobo_obstacle_paths: list[str],
    diagnostic_disable_physics_and_curobo_obstacle_paths: list[str],
    diagnostic_disable_collision_entities: list[str],
    diagnostic_collision_world: str,
    diagnostic_capture: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in queue:
        if stop_event.is_set():
            break
        row = _run_probe(
            candidate,
            gpu,
            source_task,
            target,
            run_root,
            timeout,
            conda_env,
            simulator_backend,
            seed,

            required_arm,
            planning,
            attach_prim_path_children,
            expected_target_world_xyz,
            diagnostic_disable_curobo_obstacle_paths,
            diagnostic_disable_physics_and_curobo_obstacle_paths,
            diagnostic_disable_collision_entities,
            diagnostic_collision_world,
            diagnostic_capture,
            spawn_settle=spawn_settle,
        )
        rows.append(row)
        if stop_after_feasible and row.get("feasible"):
            stop_event.set()
            break
        failures = {str(value.get("failure_code")) for value in row["arms"].values()}
        if failures & (CANDIDATE_INDEPENDENT_FAILURES | PROBE_INFRASTRUCTURE_FAILURES):
            stop_event.set()
    return rows


def _read_episode_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values[-1] if values else None


def _host_artifact_path(value: str) -> Path:
    path = Path(value)
    try:
        relative = path.relative_to(CONTAINER_ROOT)
    except ValueError:
        return path
    return REPO_ROOT / relative


def _run_pick(
    candidate: dict[str, Any],
    arm: str,
    seed: int,
    gpu: str,
    source_task: Path,
    target: str,
    run_root: Path,
    timeout: int,
    conda_env: str,
    simulator_backend: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    case_dir = run_root / "picks" / candidate_id / f"seed_{seed}"
    task_path = case_dir / "pick_task.yaml"
    data_dir = case_dir / "data"
    event_path = case_dir / "episode_events.jsonl"
    case_dir.mkdir(parents=True, exist_ok=True)
    compile_pick_task(source_task, candidate, target, arm, task_path)
    env = os.environ.copy()
    env.update(
        {
            "TASK_CONFIG": str(task_path),
            "LAUNCH_TEMPLATE": "configs/de_plan_with_render_template.yaml",
            "GPU_ID": gpu,
            "RANDOM_NUM": "1",
            "RANDOM_SEED": str(seed),
            "RUN_NAME": f"workspace_pick/{candidate_id}/seed_{seed}",
            "OUTPUT_DIR": str(data_dir),
            "INTERNDATA_EPISODE_EVENT_PATH": str(event_path),
            "INTERNDATA_RANDOM_SEED": str(seed),
            "INTERNDATA_GPU": gpu,
            "INTERNDATA_SIMULATOR_BACKEND": simulator_backend,
            "CONDA_ENV": conda_env,
            "INTERNDATA_STACK_ID": _stack_id(
                f"workspace-pick-{candidate_id}-s{seed}",
                case_dir,
            ),
            "INTERNDATA_DOCKER_METADATA_PATH": str(case_dir / "docker_runtime.json"),
            "SIMBOX_DEBUG_OUTPUT_DIR": str(case_dir / "simbox_debug"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = case_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            _runner_command(simulator_backend),
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _stop_process_group(process)
            return_code = None
            timed_out = True
    stdout = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    event = _read_episode_event(event_path)
    episode_dir = (
        _host_artifact_path(str(event.get("primary_episode_dir", "")))
        if event
        else None
    )
    event_success = bool(event and event.get("status") == "success")
    episode_name_valid = bool(episode_dir and episode_dir.name and not episode_dir.name.startswith("fail_"))
    meta_valid = bool(episode_dir and (episode_dir / "meta_info.pkl").is_file())
    lmdb_valid = bool(episode_dir and (episode_dir / "lmdb/data.mdb").is_file())
    task_success = "Task is successful" in stdout
    success = task_success and event_success and episode_name_valid and meta_valid and lmdb_valid
    return PickAttemptResult(
        candidate_id=candidate_id,
        arm=arm,
        seed=seed,
        gpu=gpu,
        return_code=return_code,
        timed_out=timed_out,
        task_is_successful=task_success,
        event_success=event_success,
        episode_dir=str(episode_dir) if episode_dir else None,
        episode_name_valid=episode_name_valid,
        meta_info_created=meta_valid,
        lmdb_created=lmdb_valid,
        success=success,
        log=str(log_path),
    ).to_dict()


def main() -> int:
    args = parse_args()
    if args.max_pick_candidates <= 0:
        raise ValueError("--max-pick-candidates must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("version") != 4:
        raise ValueError(f"expected workspace manifest version 4: {manifest_path}")
    source_task = Path(str(manifest["source_task"])).resolve()
    target = str(manifest["target"]["name"])
    expected_target_world_xyz = manifest["target"].get("world_xyz")
    if expected_target_world_xyz is not None:
        expected_target_world_xyz = [float(value) for value in expected_target_world_xyz]
    manifest_arm = manifest.get("required_arm")
    if manifest_arm is not None and args.arm != manifest_arm:
        raise ValueError(
            f"workspace manifest requires arm={manifest_arm}, got --arm={args.arm}"
        )
    if args.planning_gate == "pick-place":
        if not args.planning_only:
            raise ValueError("--planning-gate=pick-place requires --planning-only")
        if args.arm is None:
            raise ValueError("--planning-gate=pick-place requires --arm")
    capture_overview = bool(getattr(args, "capture_overview", False))
    capture_trajectory = bool(getattr(args, "capture_trajectory", False))
    capture_requested = bool(capture_overview or capture_trajectory)
    if capture_requested:
        if not args.planning_only or args.planning_gate != "pick" or args.arm is None:
            raise ValueError(
                "diagnostic capture requires --planning-only --planning-gate=pick and --arm"
            )
        raw_camera_eye = getattr(args, "camera_eye", None)
        raw_camera_target = getattr(args, "camera_target", None)
        camera_resolution = getattr(args, "camera_resolution", (1280, 960))
        camera_focal_length_mm = float(
            getattr(args, "camera_focal_length_mm", 16.0)
        )
        if (raw_camera_eye is None) != (raw_camera_target is None):
            raise ValueError("--camera-eye and --camera-target must be provided together")
        camera_eye = raw_camera_eye or [1.0, 1.0, 3.5]
        camera_target = raw_camera_target or [0.0, 0.0, 0.8]
        camera_values = [
            *(camera_eye or [0.0, 0.0, 1.0]),
            *(camera_target or [0.0, 0.0, 0.0]),
            camera_focal_length_mm,
        ]
        if not all(math.isfinite(float(value)) for value in camera_values):
            raise ValueError("diagnostic camera values must be finite")
        if min(camera_resolution) <= 0 or camera_focal_length_mm <= 0.0:
            raise ValueError("diagnostic camera resolution and focal length must be positive")
        try:
            camera_template_params = json.loads(
                getattr(args, "camera_template_params_json", "{}")
            )
        except json.JSONDecodeError as exc:
            raise ValueError("--camera-template-params-json must be valid JSON") from exc
        if not isinstance(camera_template_params, dict):
            raise ValueError("--camera-template-params-json must contain a JSON object")
    diagnostic_capture = (
        {
            "overview": capture_overview,
            "trajectory": capture_trajectory,
            "camera": (
                {
                    "eye": [float(value) for value in camera_eye],
                    "target": [float(value) for value in camera_target],
                    "resolution": [int(value) for value in camera_resolution],
                    "focal_length_mm": camera_focal_length_mm,
                }
                if capture_overview and raw_camera_eye is not None
                else {
                    "template": getattr(
                        args, "camera_template", "robot_target_overhead_v1"
                    ),
                    "template_params": camera_template_params,
                    "resolution": [int(value) for value in camera_resolution],
                    "focal_length_mm": camera_focal_length_mm,
                }
                if capture_overview
                else {}
            ),
        }
        if capture_requested
        else None
    )
    candidates = [item for item in manifest.get("geometry_candidates", []) if item.get("geometry_feasible")]
    if args.candidate_id:
        requested = set(args.candidate_id)
        candidates = [item for item in candidates if str(item.get("candidate_id")) in requested]
        found = {str(item.get("candidate_id")) for item in candidates}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested candidates are missing or geometry-infeasible: {missing}")
    if not candidates:
        manifest.update({"status": "no_geometry_candidate", "failure_code": "NO_GEOMETRY_CANDIDATE"})
        dump_json(manifest, manifest_path)
        return 2
    planning = _load_planning_config(args.planning_config)
    raw_spawn_settle = (planning or {}).get("spawn_settle")
    if raw_spawn_settle is not None and not isinstance(raw_spawn_settle, dict):
        raise ValueError("planning.spawn_settle must be a mapping")
    spawn_settle = dict(raw_spawn_settle or {})
    spawn_settle["target_support"] = str(manifest["support"]["name"])
    diagnostic_paths = [
        str(value).strip() for value in args.diagnostic_disable_curobo_obstacle_path
    ]
    dual_paths = [
        str(value).strip()
        for value in args.diagnostic_disable_physics_and_curobo_obstacle_path
    ]
    collision_entities = [
        str(value).strip() for value in args.diagnostic_disable_collision_entity
    ]
    if (
        diagnostic_paths or dual_paths or collision_entities
    ) and not args.planning_only:
        raise ValueError("diagnostic collision disable options require --planning-only")
    if args.diagnostic_collision_world != "full" and not args.planning_only:
        raise ValueError("--diagnostic-collision-world requires --planning-only")
    if any(not value or not value.startswith("/") for value in diagnostic_paths) or len(
        diagnostic_paths
    ) != len(set(diagnostic_paths)):
        raise ValueError(
            "diagnostic CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    if any(not value or not value.startswith("/") for value in dual_paths) or len(
        dual_paths
    ) != len(set(dual_paths)):
        raise ValueError(
            "diagnostic Physics+CuRobo obstacle paths must be unique, non-empty absolute Prim paths"
        )
    if any(not value for value in collision_entities) or len(collision_entities) != len(
        set(collision_entities)
    ):
        raise ValueError(
            "diagnostic collision entity names must be unique and non-empty"
        )
    if (
        diagnostic_paths or dual_paths or collision_entities
    ) and args.diagnostic_collision_world != "full":
        raise ValueError(
            "exact/entity diagnostic collision isolation cannot be combined with "
            "target-only or empty-world diagnostics"
        )
    if diagnostic_paths and (dual_paths or collision_entities):
        raise ValueError(
            "CuRobo-only and Physics+CuRobo diagnostic isolation modes are mutually exclusive"
        )
    if args.planning_gate == "pick-place" and (
        diagnostic_paths
        or dual_paths
        or collision_entities
        or args.diagnostic_collision_world != "full"
    ):
        raise ValueError(
            "Pick+Place planning gate requires the complete Physics/CuRobo collision world"
        )
    attach_paths = [str(value).strip() for value in args.attach_prim_path_child]
    if any(not value for value in attach_paths) or len(attach_paths) != len(set(attach_paths)):
        raise ValueError("attach Prim child paths must be unique and non-empty")
    if args.arm is not None and (planning is None or not attach_paths):
        raise ValueError(
            "required-arm probes require --planning-config and at least one "
            "--attach-prim-path-child so Probe and execution share a contract"
        )
    if not args.candidate_id:
        candidates = _shortlist_candidates(
            candidates,
            args.manipulation_preferred_radius_m,
            args.max_probe_candidates,
        )
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    run_root = manifest_path.parent
    manifest["probe_contract"] = {
        "simulator_backend": args.simulator_backend,
        "planning_gate": args.planning_gate,
        "seed": args.seed,
        "required_arm": args.arm,
        "planning_config": str(args.planning_config.resolve()) if args.planning_config else None,
        "collision_world_mode": str(
            ((planning or {}).get("collision_world") or {}).get("mode", "")
        ),
        "attach_prim_path_children": attach_paths,
        "max_probe_candidates": int(args.max_probe_candidates),
        "manipulation_preferred_radius_m": float(
            args.manipulation_preferred_radius_m
        ),
        "candidate_ids": [str(value["candidate_id"]) for value in candidates],
        "diagnostic_disable_curobo_obstacle_paths": diagnostic_paths,
        "diagnostic_disable_physics_and_curobo_obstacle_paths": dual_paths,
        "diagnostic_disable_collision_entities": collision_entities,
        "diagnostic_collision_world": args.diagnostic_collision_world,
        "diagnostic_capture": diagnostic_capture,
    }
    _progress(
        f"geometry shortlist={len(candidates)}, GPUs={','.join(gpus)}, "
        f"target={target}, required_arm={args.arm or 'legacy_both'}, "
        f"backend={args.simulator_backend}"
    )
    stop_event = threading.Event()
    queues = {gpu: candidates[index :: len(gpus)] for index, gpu in enumerate(gpus)}
    probe_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(
                _run_probe_queue,
                queue,
                gpu,
                source_task,
                target,
                run_root,
                args.probe_timeout_sec,
                args.conda_env,
                args.simulator_backend,
                args.seed,

                stop_event,
                args.stop_after_feasible,
                args.arm,
                planning,
                attach_paths,
                expected_target_world_xyz,
                spawn_settle,
                diagnostic_paths,
                dual_paths,
                collision_entities,
                args.diagnostic_collision_world,
                diagnostic_capture,
            )
            for gpu, queue in queues.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            probe_rows.extend(future.result())
    probe_rows.sort(key=lambda item: item["candidate_id"])
    manifest["curobo_results"] = probe_rows
    _progress(
        f"CuRobo probes completed={len(probe_rows)}, "
        f"feasible={sum(bool(row.get('feasible')) for row in probe_rows)}"
    )
    fixed_failures = {
        str(arm.get("failure_code"))
        for row in probe_rows
        for arm in row.get("arms", {}).values()
        if arm.get("failure_code")
        in (CANDIDATE_INDEPENDENT_FAILURES | PROBE_INFRASTRUCTURE_FAILURES)
    }
    if fixed_failures:
        _progress(f"blocked by candidate-independent failure: {sorted(fixed_failures)[0]}")
        manifest.update({"status": "blocked", "failure_code": sorted(fixed_failures)[0]})
        dump_json(manifest, manifest_path)
        return 3

    candidate_by_id = _candidate_map(manifest)
    feasible_rows = [row for row in probe_rows if row.get("feasible")]
    feasible_rows.sort(
        key=lambda row: (
            -int(row.get("joint_success_count", 0)),
            float(row["selected_grasp_score"]) if row.get("selected_grasp_score") is not None else math.inf,
            abs(
                float(candidate_by_id[row["candidate_id"]]["radius_m"])
                - float(manifest["sampling"]["preferred_radius_m"])
            ),
            row["candidate_id"],
        )
    )
    if not feasible_rows:
        status, failure_code, failure_summary = _summarize_probe_failures(probe_rows)
        _progress(
            "no safe required-arm pose: "
            f"stable={failure_summary['spawn_stable_count']}, "
            f"spawn_unstable={failure_summary['spawn_unstable_count']}"
        )
        manifest.update(
            {
                "status": status,
                "failure_code": failure_code,
                "failure_summary": failure_summary,
            }
        )
        dump_json(manifest, manifest_path)
        return 4

    if args.planning_only:
        best = feasible_rows[0]
        if args.planning_gate == "pick-place":
            place_rows = []
            best = None
            for index, row in enumerate(feasible_rows[: args.max_pick_candidates]):
                candidate = candidate_by_id[row["candidate_id"]]
                place_row = run_pick_place_planning_probe(
                    candidate,
                    str(row["selected_arm"]),
                    gpus[index % len(gpus)],
                    source_task,
                    target,
                    run_root,
                    args.place_probe_timeout_sec,
                    args.conda_env,
                    args.simulator_backend,
                    planning,
                    attach_paths,
                    seed=args.seed,
                )
                place_rows.append(place_row)
                _progress(
                    f"Place probe candidate={row['candidate_id']} "
                    f"arm={row['selected_arm']} feasible={place_row['feasible']}"
                )
                if place_row["feasible"]:
                    best = row
                    break
            manifest["place_probe_results"] = place_rows
            if best is None:
                manifest.update(
                    {
                        "status": "no_place_plan",
                        "failure_code": "NO_PLACE_PLAN",
                    }
                )
                dump_json(manifest, manifest_path)
                return 6
        selected_arm = str(best["selected_arm"])
        pick_result = best["arms"][selected_arm]
        spawn_check = pick_result.get("spawn_check")
        if not isinstance(spawn_check, dict) or spawn_check.get("stable") is not True:
            manifest.update(
                {
                    "status": "no_stable_spawn",
                    "failure_code": "PROBE_SPAWN_UNSTABLE",
                }
            )
            dump_json(manifest, manifest_path)
            return 7
        spawn_settle_path = run_root / "spawn_settle.json"
        dump_json(
            {
                "schema_version": 1,
                "candidate_id": best["candidate_id"],
                "arm": selected_arm,
                "seed": args.seed,
                "stable": True,
                "spawn_check": spawn_check,
                "source_pick_probe": str(pick_result["artifact"]),
            },
            spawn_settle_path,
        )
        probe_artifacts = {
            "pick": str(best["arms"][selected_arm]["artifact"]),
            "spawn_settle": str(spawn_settle_path),
        }
        if args.planning_gate == "pick-place":
            selected_place_row = next(
                row
                for row in manifest["place_probe_results"]
                if row["candidate_id"] == best["candidate_id"] and row["feasible"]
            )
            probe_artifacts["place"] = str(selected_place_row["artifact"])
        manifest.update(
            {
                "selected_candidate": {
                    **candidate_by_id[best["candidate_id"]],
                    "arm": best["selected_arm"],
                },
                "planning_probe_artifacts": probe_artifacts,
                "status": "planning_success",
                "failure_code": None,
            }
        )
        dump_json(manifest, manifest_path)
        _progress(
            f"planning selected={best['candidate_id']} arm={best['selected_arm']} "
            f"joint_success_count={best['joint_success_count']}"
        )
        return 0

    attempts: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    for index, row in enumerate(feasible_rows[: args.max_pick_candidates]):
        candidate = candidate_by_id[row["candidate_id"]]
        attempt = _run_pick(
            candidate,
            str(row["selected_arm"]),
            0,
            gpus[index % len(gpus)],
            source_task,
            target,
            run_root,
            args.pick_timeout_sec,
            args.conda_env,
            args.simulator_backend,
        )
        attempts.append(attempt)
        _progress(
            f"Pick candidate={row['candidate_id']} arm={row['selected_arm']} "
            f"seed=0 success={attempt['success']}"
        )
        if attempt["success"]:
            winner = {**candidate, "arm": row["selected_arm"]}
            break
    if winner is None:
        _progress("no Pick success among ranked candidates")
        manifest.update(
            {"pick_attempts": attempts, "status": "no_pick_success", "failure_code": "NO_PICK_SUCCESS"}
        )
        dump_json(manifest, manifest_path)
        return 5

    for seed in (1, 2):
        attempt = _run_pick(
            winner,
            str(winner["arm"]),
            seed,
            gpus[seed % len(gpus)],
            source_task,
            target,
            run_root,
            args.pick_timeout_sec,
            args.conda_env,
            args.simulator_backend,
        )
        attempts.append(attempt)
        _progress(
            f"Pick candidate={winner['candidate_id']} arm={winner['arm']} "
            f"seed={seed} success={attempt['success']}"
        )
    successes = sum(bool(item["success"]) for item in attempts if item["candidate_id"] == winner["candidate_id"])
    winner["stability"] = "3/3 stable" if successes == 3 else "2/3 partially_stable" if successes == 2 else "unstable"
    manifest.update(
        {
            "pick_attempts": attempts,
            "selected_candidate": winner,
            "status": "success",
            "failure_code": None,
        }
    )
    dump_json(manifest, manifest_path)
    _progress(f"selected={winner['candidate_id']} arm={winner['arm']} stability={winner['stability']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
