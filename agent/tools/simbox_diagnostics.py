"""User-facing deterministic wrappers for SimBox views and CuRobo probes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent.settings import resolve_debug_camera
from workflows.simbox.core.utils.camera_template import (
    CAMERA_TEMPLATE_DEFAULTS,
    resolve_camera_template_pose,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNING_CONFIG = REPO_ROOT / "agent" / "config.yaml"


def _camera_for_args(args, settings: Mapping[str, Any]) -> dict[str, Any]:
    camera = resolve_debug_camera(settings)
    requested_template = getattr(args, "camera_template", None)
    template = requested_template or camera["template"]
    defaults = CAMERA_TEMPLATE_DEFAULTS[template]
    params = (
        dict(defaults)
        if requested_template and requested_template != camera["template"]
        else dict(camera["template_params"])
    )
    overrides = {
        "height_m": getattr(args, "camera_height_m", None),
        "look_fraction": getattr(args, "camera_look_fraction", None),
        "look_height_m": getattr(args, "camera_look_height_m", None),
        "behind_m": getattr(args, "camera_behind_m", None),
        "side_m": getattr(args, "camera_side_m", None),
    }
    supplied = {key: value for key, value in overrides.items() if value is not None}
    invalid = sorted(set(supplied) - set(defaults))
    if invalid:
        raise ValueError(f"camera template {template} does not support: {invalid}")
    params.update({key: float(value) for key, value in supplied.items()})
    resolve_camera_template_pose(
        template,
        [0.0, 0.0, 0.0],
        0.0,
        [1.0, 0.0, 0.0],
        params,
    )
    camera["template"] = template
    camera["template_params"] = params
    if requested_template or supplied:
        camera["eye"] = None
        camera["target"] = None
    return camera


def _vector(values: Sequence[Any], name: str, length: int) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    result = [float(value) for value in values]
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _task_attach_paths(manifest: Mapping[str, Any]) -> list[str]:
    probe_contract = manifest.get("probe_contract")
    if isinstance(probe_contract, Mapping):
        values = probe_contract.get("attach_prim_path_children")
        if isinstance(values, list) and values:
            return [str(value) for value in values]

    source_task = Path(str(manifest.get("source_task") or "")).expanduser().resolve()
    if not source_task.is_file():
        raise ValueError(f"workspace manifest source_task is missing: {source_task}")
    document = yaml.safe_load(source_task.read_text(encoding="utf-8")) or {}
    tasks = document.get("tasks") or []
    target = str((manifest.get("target") or {}).get("name") or "")
    if not tasks or not target:
        raise ValueError("workspace manifest must declare source_task and target.name")
    target_object = next(
        (item for item in tasks[0].get("objects", []) if item.get("name") == target),
        None,
    )
    if target_object is None:
        raise ValueError(f"target object is missing from source task: {target}")
    values = target_object.get("attach_prim_path_children")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    value = target_object.get("attach_prim_path_child")
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    raise ValueError(f"target object has no attach collision path: {target}")


def _task_camera_subjects(task_path: Path) -> tuple[str, str]:
    document = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    tasks = document.get("tasks") or []
    if not tasks:
        raise ValueError(f"task has no task block: {task_path}")
    task = tasks[0]
    robots = task.get("robots") or []
    if not robots or not robots[0].get("name"):
        raise ValueError(f"task has no robot instance: {task_path}")

    def find_pick(value: Any) -> str | None:
        if isinstance(value, Mapping):
            name = str(value.get("name") or "").lower()
            objects = value.get("objects")
            if name in {"pick", "pick_plan_probe"} and isinstance(objects, list) and objects:
                return str(objects[0])
            for child in value.values():
                target = find_pick(child)
                if target:
                    return target
        elif isinstance(value, list):
            for child in value:
                target = find_pick(child)
                if target:
                    return target
        return None

    target = find_pick(task.get("skills") or [])
    if target is None:
        active = task.get("delivery_active_objects") or []
        target = str(active[0]) if active else None
    if not target:
        raise ValueError(
            "robot-target camera template requires a Pick target or delivery_active_objects"
        )
    return str(robots[0]["name"]), target


def run_view(args, settings: Mapping[str, Any]) -> int:
    camera = _camera_for_args(args, settings)
    requested_view = args.view or ("debug_overview" if args.mode == "physics" else "")
    custom_eye = args.eye is not None or args.target is not None
    if custom_eye and (args.eye is None or args.target is None):
        raise ValueError("--eye and --target must be provided together")
    use_custom_view = custom_eye or requested_view == "debug_overview"
    eye = _vector(args.eye, "--eye", 3) if args.eye is not None else camera["eye"]
    target = _vector(args.target, "--target", 3) if args.target is not None else camera["target"]
    width = int(args.width if args.width is not None else camera["resolution"][0])
    height = int(args.height if args.height is not None else camera["resolution"][1])
    focal_length = float(
        args.focal_length_mm
        if args.focal_length_mm is not None
        else camera["focal_length_mm"]
    )
    if width <= 0 or height <= 0 or focal_length <= 0.0:
        raise ValueError("view resolution and focal length must be positive")
    if args.mode == "layout" and use_custom_view:
        raise ValueError("custom eye/target requires --mode physics")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "agent" / "visual.py"),
        "--task",
        str(args.task.resolve()),
        "--out-dir",
        str(output_dir),
        "--mode",
        args.mode,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if args.include_robot:
        command.append("--include-robot")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["INTERNDATA_ISAAC_ACTIVE_GPU"] = str(args.gpu)
    env["TASK_RENDER_FOCAL_LENGTH_MM"] = str(focal_length)
    if use_custom_view:
        view_name = "debug_overview"
        if eye is None:
            robot_name, target_name = _task_camera_subjects(args.task.resolve())
            view = {
                "name": view_name,
                "template": camera["template"],
                "template_params": camera["template_params"],
                "robot": robot_name,
                "target": target_name,
            }
        else:
            view = {"name": view_name, "eye": eye, "target": target}
        env["TASK_RENDER_EXTRA_VIEWS_JSON"] = json.dumps(
            [view],
            separators=(",", ":"),
        )
        command.extend(["--single-view", view_name])
    elif requested_view:
        command.extend(["--single-view", requested_view])
    log_path = output_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess_return_code = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    render_status_path = output_dir / "render_status.json"
    render_status = None
    if render_status_path.is_file():
        render_status = json.loads(render_status_path.read_text(encoding="utf-8"))
    image_files = sorted(output_dir.glob("**/*.png"))
    if args.mode == "physics":
        return_code = (
            int(render_status["return_code"])
            if isinstance(render_status, Mapping) and "return_code" in render_status
            else int(subprocess_return_code or 1)
        )
        physics_audit_path = output_dir / "physics_audit.json"
        physics_audit = (
            json.loads(physics_audit_path.read_text(encoding="utf-8"))
            if physics_audit_path.is_file()
            else {}
        )
        if return_code == 0 and (
            not image_files or physics_audit.get("physics_enabled") is not True
        ):
            return_code = 1
        visualization_manifest = {
            "task": str(args.task.resolve()),
            "renderer": str((REPO_ROOT / "agent" / "visual_physics.py").resolve()),
            "physics_enabled": physics_audit.get("physics_enabled") is True,
            "physics_audit": str(physics_audit_path.resolve()),
            "render_status": str(render_status_path.resolve()),
            "exit_code": return_code,
            "image_count": len(image_files),
            "images": [
                path.relative_to(output_dir).as_posix() for path in image_files
            ],
        }
        visualization_manifest_path = output_dir / "visualization_manifest.json"
        visualization_manifest_path.write_text(
            json.dumps(visualization_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        return_code = int(subprocess_return_code)
        visualization_manifest_path = None
    summary = {
        "task": str(args.task.resolve()),
        "command": command,
        "return_code": return_code,
        "subprocess_return_code": subprocess_return_code,
        "stdout_log": str(log_path),
        "image_files": [str(path.resolve()) for path in image_files],
        "visualization_manifest": (
            str(visualization_manifest_path.resolve())
            if visualization_manifest_path is not None
            else None
        ),
        "layout_manifest": (
            str((output_dir / "layout_manifest.json").resolve())
            if (output_dir / "layout_manifest.json").is_file()
            else None
        ),
    }
    (output_dir / "view_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return return_code


def _probe_command(
    args,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[str]:
    arm = args.arm or manifest.get("required_arm")
    if arm not in {"left", "right"}:
        raise ValueError("probe requires a manifest required_arm or explicit --arm")
    attach_paths = list(args.attach_prim_path_child or _task_attach_paths(manifest))
    if not attach_paths:
        raise ValueError("probe requires at least one attach collision path")
    planning_config = (args.planning_config or DEFAULT_PLANNING_CONFIG).resolve()
    execution = settings.get("execution", {})
    conda_env = args.conda_env or (
        execution.get("conda_env", "interndata-isaac6")
        if isinstance(execution, Mapping)
        else "interndata-isaac6"
    )
    simulator_backend = getattr(args, "simulator_backend", None) or (
        execution.get("simulator_backend", "docker")
        if isinstance(execution, Mapping)
        else "docker"
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "simbox" / "validate_workspace_candidates.py"),
        "--manifest",
        str(manifest_path),
        "--gpus",
        str(args.gpu),
        "--simulator-backend",
        str(simulator_backend),
        "--arm",
        str(arm),
        "--planning-config",
        str(planning_config),
        "--conda-env",
        str(conda_env),
        "--planning-only",
        "--planning-gate",
        args.gate,
        "--seed",
        str(args.seed),
        "--probe-timeout-sec",
        str(args.timeout_sec),
        "--diagnostic-collision-world",
        args.collision_world,
    ]
    for path in attach_paths:
        command.extend(["--attach-prim-path-child", path])
    if args.candidate_id:
        command.extend(["--candidate-id", args.candidate_id])
    if args.stop_after_feasible:
        command.append("--stop-after-feasible")
    for entity in args.disable_collision_entity:
        command.extend(["--diagnostic-disable-collision-entity", entity])
    for path in args.disable_curobo_obstacle_path:
        command.extend(["--diagnostic-disable-curobo-obstacle-path", path])
    for path in args.disable_physics_and_curobo_obstacle_path:
        command.extend(
            ["--diagnostic-disable-physics-and-curobo-obstacle-path", path]
        )

    if args.capture_overview or args.capture_trajectory:
        if args.gate != "pick":
            raise ValueError("diagnostic capture currently requires --gate pick")
        if (args.camera_eye is None) != (args.camera_target is None):
            raise ValueError("--camera-eye and --camera-target must be provided together")
        camera = _camera_for_args(args, settings)
        eye = _vector(args.camera_eye, "--camera-eye", 3) if args.camera_eye else camera["eye"]
        target = _vector(args.camera_target, "--camera-target", 3) if args.camera_target else camera["target"]
        resolution = (
            [int(value) for value in _vector(args.resolution, "--resolution", 2)]
            if args.resolution
            else camera["resolution"]
        )
        focal_length = float(
            args.focal_length_mm
            if args.focal_length_mm is not None
            else camera["focal_length_mm"]
        )
        if min(resolution) <= 0 or focal_length <= 0.0:
            raise ValueError("capture resolution and focal length must be positive")
        if args.capture_overview:
            command.append("--capture-overview")
            if eye is not None:
                command.extend(
                    [
                        "--camera-eye",
                        *(str(value) for value in eye),
                        "--camera-target",
                        *(str(value) for value in target),
                    ]
                )
            else:
                command.extend(
                    [
                        "--camera-template",
                        camera["template"],
                        "--camera-template-params-json",
                        json.dumps(camera["template_params"], separators=(",", ":")),
                    ]
                )
            command.extend(
                [
                    "--camera-resolution",
                    *(str(value) for value in resolution),
                    "--camera-focal-length-mm",
                    str(focal_length),
                ]
            )
        if args.capture_trajectory:
            command.append("--capture-trajectory")
    return command


def run_probe(args, settings: Mapping[str, Any]) -> int:
    source_manifest = args.manifest.resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("workspace manifest root must be an object")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "candidates.json"
    if source_manifest != manifest_path:
        shutil.copy2(source_manifest, manifest_path)
    command = _probe_command(args, manifest_path, manifest, settings)
    if args.dry_run:
        return_code = 0
    else:
        return_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    summary = {
        "source_manifest": str(source_manifest),
        "manifest": str(manifest_path),
        "command": command,
        "dry_run": bool(args.dry_run),
        "return_code": return_code,
        "result_files": [
            str(path.resolve()) for path in sorted(output_dir.glob("probes/**/results/*.json"))
        ],
        "overview_files": [
            str(path.resolve()) for path in sorted(output_dir.glob("probes/**/diagnostics/overview.png"))
        ],
        "trajectory_files": [
            str(path.resolve())
            for path in sorted(output_dir.glob("probes/**/diagnostics/trajectory_debug.usda"))
        ],
    }
    summary_path = output_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return return_code
