#!/usr/bin/env python3
"""Probe PandaOmron articulation targets inside the real SimBox scene."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import traceback

import cv2  # noqa: F401  # Preload OpenCV before Kit adjusts shared library paths.
import numpy as np
import yaml
from isaacsim import SimulationApp
from omegaconf import DictConfig, ListConfig, OmegaConf

_RUNNER_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
SIMULATION_APP = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
sys.argv = [sys.argv[0], *_RUNNER_ARGS]

sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("workflows/simbox")

from omni.isaac.core import World  # noqa: E402  pylint: disable=wrong-import-position
from omni.isaac.core.utils.prims import get_prim_at_path  # noqa: E402  pylint: disable=wrong-import-position
from omni.physx import acquire_physx_interface  # noqa: E402  pylint: disable=wrong-import-position
from pxr import UsdPhysics  # noqa: E402  pylint: disable=wrong-import-position
from yaml import Loader  # noqa: E402  pylint: disable=wrong-import-position

from nimbus.utils.utils import init_env  # noqa: E402  pylint: disable=wrong-import-position
from workflows import import_extensions  # noqa: E402  pylint: disable=wrong-import-position
from workflows.base import create_workflow  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.core.tasks import get_task_cls  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.core.utils.collision_utils import filter_collisions  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.utils.task_config_parser import TaskConfigParser  # noqa: E402  pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/de_plan_with_render_scene8_validation.yaml")
    parser.add_argument("--output", default="output/debug_panda_omron_scene_articulation/result.json")
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--drive-steps", type=int, default=120)
    parser.add_argument("--arm-delta", type=float, default=0.10)
    parser.add_argument(
        "--arm-target",
        type=float,
        nargs=7,
        default=None,
        help="Absolute 7-DOF arm target. Overrides --arm-delta when provided.",
    )
    parser.add_argument(
        "--include-gripper-target",
        action="store_true",
        help="Send arm and gripper targets together through robot.apply_action.",
    )
    parser.add_argument(
        "--gripper-target",
        type=float,
        nargs=2,
        default=[0.04, -0.04],
        help="Absolute 2-DOF gripper target used with --include-gripper-target.",
    )
    parser.add_argument("--arm-action-shape", choices=("1d", "2d"), default="2d")
    parser.add_argument(
        "--base-hold-during-arm",
        action="store_true",
        help="Also apply a zero base command after each arm target to match the full workflow step order.",
    )
    parser.add_argument(
        "--base-command",
        type=float,
        nargs="+",
        default=None,
        help="Direct base joint velocity command. Use 2 wheel speeds for diff drive or 3 virtual-base velocities.",
    )
    parser.add_argument(
        "--base-before-arm",
        action="store_true",
        help="Drive the base before arm target execution to reproduce post-navigation pick ordering.",
    )
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--manual-task-load",
        dest="manual_task_load",
        action="store_true",
        help="Load the task directly without SimBox workflow/nav bridge setup.",
    )
    load_group.add_argument(
        "--workflow-load",
        dest="manual_task_load",
        action="store_false",
        help="Load through the normal SimBox workflow path.",
    )
    parser.set_defaults(manual_task_load=True)
    parser.add_argument(
        "--keep-bridges",
        action="store_true",
        help="When using --workflow-load, keep ROS base bridges and nav managers active.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _eval_dt(value) -> float:
    if isinstance(value, str):
        return float(eval(value, {"__builtins__": {}}, {}))
    return float(value)


def _world_physics_dt(world: World) -> float:
    get_physics_dt = getattr(world, "get_physics_dt", None)
    if callable(get_physics_dt):
        return float(get_physics_dt())
    return float(getattr(world, "physics_dt", 1.0 / 60.0))


def _apply_base_command(robot, world: World, steering_positions: np.ndarray, wheel_velocities: np.ndarray) -> None:
    signature = inspect.signature(robot.apply_base_command)
    if "step_dt" in signature.parameters:
        robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            step_dt=_world_physics_dt(world),
        )
        return
    robot.apply_base_command(steering_positions, wheel_velocities)


def _build_world(simulator_cfg: dict) -> World:
    return World(
        physics_dt=_eval_dt(simulator_cfg["physics_dt"]),
        rendering_dt=_eval_dt(simulator_cfg["rendering_dt"]),
        stage_units_in_meters=float(simulator_cfg.get("stage_units_in_meters", 1.0)),
    )


def _deep_update_dict(base: dict, override: dict):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value


def _merge_base_configs(base_cfg: dict):
    override_cfg = dict(base_cfg)
    merged_base_cfg = {}
    base_config_file = override_cfg.get("base_config_file")
    nav_config_file = override_cfg.get("nav_config_file")

    if base_config_file:
        with open(base_config_file, "r", encoding="utf-8") as handle:
            loaded_base_cfg = yaml.load(handle, Loader=Loader)
        if isinstance(loaded_base_cfg, dict):
            merged_base_cfg = dict(loaded_base_cfg)

    if nav_config_file:
        nav_config_files = nav_config_file if isinstance(nav_config_file, list) else [nav_config_file]
        for nav_config_path in nav_config_files:
            with open(nav_config_path, "r", encoding="utf-8") as handle:
                loaded_nav_cfg = yaml.load(handle, Loader=Loader)
            if isinstance(loaded_nav_cfg, dict):
                _deep_update_dict(merged_base_cfg, loaded_nav_cfg)

    _deep_update_dict(merged_base_cfg, override_cfg)
    base_cfg.clear()
    base_cfg.update(merged_base_cfg)


def _merge_robot_configs(task_cfg: dict):
    for robot in task_cfg.get("robots", []):
        robot_config_file = robot.get("robot_config_file")
        if not robot_config_file:
            continue
        with open(robot_config_file, "r", encoding="utf-8") as handle:
            robot_base_cfg = yaml.load(handle, Loader=Loader)
        merged_cfg = dict(robot_base_cfg)
        merged_cfg.update(robot)
        base_cfg = merged_cfg.get("base")
        if isinstance(base_cfg, dict):
            _merge_base_configs(base_cfg)
        robot.clear()
        robot.update(merged_cfg)


def _resolve_arena_file_path(task_cfg: dict) -> str:
    arena_file_path = task_cfg.get("arena_file")
    if arena_file_path and os.path.exists(arena_file_path):
        return arena_file_path
    asset_root = task_cfg.get("asset_root")
    if arena_file_path and asset_root:
        candidate = os.path.join(asset_root, arena_file_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"arena_file does not exist: {arena_file_path}")


def _build_manual_task(config_path: str, world: World):
    config = _load_config(config_path)
    scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
    task_cfg_path = scene_loader_cfg["cfg_path"]
    task_cfg = TaskConfigParser(task_cfg_path).parse_tasks()[0]
    _merge_robot_configs(task_cfg)
    arena_file_path = _resolve_arena_file_path(task_cfg)
    with open(arena_file_path, "r", encoding="utf-8") as handle:
        task_cfg["arena"] = yaml.load(handle, Loader=Loader)
    task_cfg.pop("arena_file", None)
    task_cfg.pop("camera_file", None)
    task_cfg.pop("logger_file", None)

    if task_cfg.get("fluid", None):
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)

    task = get_task_cls(task_cfg["task"])(task_cfg)
    stage = world.stage
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    root_prim = stage.GetPrimAtPath(task.root_prim_path)
    if root_prim.IsValid():
        stage.RemovePrim(task.root_prim_path)
    collision_prim = stage.GetPrimAtPath("/World/collisions")
    if collision_prim.IsValid():
        stage.RemovePrim("/World/collisions")
    world.add_task(task)

    prim_paths = []
    global_collision_paths = []
    for robot_cfg in task_cfg["robots"]:
        prim_paths.append(task.root_prim_path + "/" + robot_cfg["name"])
    neglect_collision_names = task_cfg.get("neglect_collision_names", [])
    for candidate in task_cfg["objects"] + task_cfg["arena"]["fixtures"]:
        candidate_prim_path = task.root_prim_path + "/" + candidate["name"]
        global_collision_paths.append(candidate_prim_path)
        for neglect_collision_name in neglect_collision_names:
            if neglect_collision_name in candidate["name"]:
                prim_paths.append(candidate_prim_path)
                global_collision_paths.remove(candidate_prim_path)

    filter_collisions(
        stage,
        world.get_physics_context().prim_path,
        "/World/collisions",
        prim_paths,
        global_collision_paths,
    )
    world.reset()
    world.step(render=False)
    task.set_fixed_robot_start_poses()
    world.step(render=False)
    for _ in range(20):
        world.step(render=False)
    return task, scene_loader_cfg


def _find_robot(workflow):
    robots = workflow.task.robots if hasattr(workflow, "task") else workflow.robots
    for robot in robots.values():
        if robot.__class__.__name__ in {"PandaOmron", "PandaOmronVirtual"}:
            return robot
    available = {name: robot.__class__.__name__ for name, robot in workflow.task.robots.items()}
    raise RuntimeError(f"PandaOmron robot not found; available={available}")


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _joint_snapshot(robot, label: str, step: int) -> dict:
    pos, quat = robot.get_mobile_base_pose()
    nav_getter = getattr(robot, "get_nav_base_pose", None)
    if not callable(nav_getter):
        raise RuntimeError("debug_panda_omron_scene_articulation requires get_nav_base_pose on PandaOmron")
    nav_pos, nav_quat = nav_getter()
    joint_positions = robot._articulation_view.get_joint_positions()[0]
    joint_velocities = robot._articulation_view.get_joint_velocities()[0]
    arm_indices = np.asarray(robot.left_joint_indices, dtype=np.int64)
    gripper_indices = np.asarray(robot.left_gripper_indices, dtype=np.int64)
    base_indices = np.asarray(robot.base_wheel_joint_indices, dtype=np.int64)
    arm_position_targets = getattr(robot, "_debug_last_arm_position_targets", None)
    base_velocity_targets = getattr(robot, "_debug_last_base_velocity_targets", None)
    return {
        "label": str(label),
        "step": int(step),
        "mobile_base_position": [float(v) for v in list(pos)],
        "mobile_base_yaw": float(_yaw_from_wxyz(quat)),
        "nav_base_position": [float(v) for v in list(nav_pos)],
        "nav_base_yaw": float(_yaw_from_wxyz(nav_quat)),
        "nav_minus_mobile_xy": [float(nav_pos[0] - pos[0]), float(nav_pos[1] - pos[1])],
        "arm_positions": [float(v) for v in joint_positions[arm_indices].tolist()],
        "arm_velocities": [float(v) for v in joint_velocities[arm_indices].tolist()],
        "arm_position_targets": (
            None
            if arm_position_targets is None
            else [float(v) for v in np.asarray(arm_position_targets).reshape(-1).tolist()]
        ),
        "gripper_positions": [float(v) for v in joint_positions[gripper_indices].tolist()],
        "base_positions": [float(v) for v in joint_positions[base_indices].tolist()],
        "base_velocities": [float(v) for v in joint_velocities[base_indices].tolist()],
        "base_velocity_targets": (
            None
            if base_velocity_targets is None
            else [float(v) for v in np.asarray(base_velocity_targets).reshape(-1).tolist()]
        ),
    }


def _drive_snapshot(robot) -> dict:
    from omni.isaac.core.utils.prims import get_prim_at_path  # pylint: disable=import-outside-toplevel

    drive_info = {}
    joint_root = f"{robot.robot_prim_path}/robot0_base/joints"
    for joint_name in list(robot.cfg["left_joint_names"]) + list(robot.base_wheel_joint_names):
        joint_path = f"{joint_root}/{joint_name}"
        prim = get_prim_at_path(joint_path)
        if not prim.IsValid():
            drive_info[joint_name] = {"exists": False}
            continue
        entries = {}
        for drive_name in ("X", "linear", "angular"):
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
            if not drive_api:
                continue
            entries[drive_name] = {
                "type": str(drive_api.GetTypeAttr().Get()),
                "stiffness": float(drive_api.GetStiffnessAttr().Get() or 0.0),
                "damping": float(drive_api.GetDampingAttr().Get() or 0.0),
                "max_force": float(drive_api.GetMaxForceAttr().Get() or 0.0),
            }
        drive_info[joint_name] = {"exists": True, "drives": entries}
    return drive_info


def _json_safe(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run() -> int:
    args = parse_args()
    report = {
        "status": "started",
        "config": args.config,
        "settle_steps": int(args.settle_steps),
        "drive_steps": int(args.drive_steps),
        "arm_delta": float(args.arm_delta),
        "arm_target": None if args.arm_target is None else [float(v) for v in args.arm_target],
        "include_gripper_target": bool(args.include_gripper_target),
        "gripper_target": [float(v) for v in args.gripper_target],
        "arm_action_shape": str(args.arm_action_shape),
        "base_hold_during_arm": bool(args.base_hold_during_arm),
        "base_before_arm": bool(args.base_before_arm),
        "base_command": None if args.base_command is None else [float(v) for v in args.base_command],
        "manual_task_load": bool(args.manual_task_load),
        "keep_bridges": bool(args.keep_bridges),
    }
    try:
        init_env()
        config = _load_config(args.config)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        simulator_cfg = scene_loader_cfg["simulator"]

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        if args.manual_task_load:
            task, _ = _build_manual_task(args.config, world)
            robot = _find_robot(task)
        else:
            workflow = create_workflow(workflow_type, world, scene_loader_cfg["cfg_path"])
            workflow.init_task(0)
            if not args.keep_bridges and hasattr(workflow, "_destroy_navigation_session_managers"):
                workflow._destroy_navigation_session_managers()
            if not args.keep_bridges and hasattr(workflow, "_destroy_nav2_clock_publisher"):
                workflow._destroy_nav2_clock_publisher()
            if not args.keep_bridges and hasattr(workflow, "_destroy_ros_base_bridges"):
                workflow._destroy_ros_base_bridges()
            robot = _find_robot(workflow)
        dof_names = list(robot._articulation_view.dof_names)
        report.update(
            {
                "robot_name": robot.name,
                "robot_class": robot.__class__.__name__,
                "dof_names": dof_names,
                "left_joint_indices": [int(v) for v in robot.left_joint_indices],
                "left_joint_names_by_index": [dof_names[int(v)] for v in robot.left_joint_indices],
                "left_gripper_indices": [int(v) for v in robot.left_gripper_indices],
                "left_gripper_names_by_index": [dof_names[int(v)] for v in robot.left_gripper_indices],
                "base_wheel_joint_indices": [int(v) for v in robot.base_wheel_joint_indices],
                "base_wheel_names_by_index": [dof_names[int(v)] for v in robot.base_wheel_joint_indices],
                "base_interface": robot.get_base_interface(),
                "drive_snapshot": _drive_snapshot(robot),
            }
        )

        samples = [_joint_snapshot(robot, "initial", 0)]
        zero_base = np.zeros(len(robot.base_wheel_joint_indices), dtype=np.float32)
        for step in range(max(int(args.settle_steps), 0)):
            _apply_base_command(robot, world, np.zeros(0, dtype=np.float32), zero_base)
            robot._debug_last_base_velocity_targets = zero_base.copy()
            if not args.manual_task_load and args.keep_bridges:
                workflow._step_world(render=False)
            else:
                world.step(render=False)
            if step in {0, 29, int(args.settle_steps) - 1}:
                samples.append(_joint_snapshot(robot, "settle", step + 1))

        if args.base_command is None:
            if len(robot.base_wheel_joint_indices) == 2:
                base_command = np.asarray([2.0, 2.0], dtype=np.float32)
            elif len(robot.base_wheel_joint_indices) == 3:
                base_command = np.asarray([0.16, 0.0, 0.0], dtype=np.float32)
            else:
                raise ValueError("Cannot infer default base command for this base joint count")
            report["base_command"] = [float(v) for v in base_command.tolist()]
        else:
            base_command = np.asarray(args.base_command, dtype=np.float32)
        if base_command.size != len(robot.base_wheel_joint_indices):
            raise ValueError("base-command length must match base wheel/velocity joint count")

        def step_world_once():
            if not args.manual_task_load and args.keep_bridges:
                workflow._step_world(render=False)
            else:
                world.step(render=False)

        def run_base_drive(label: str):
            for step in range(max(int(args.drive_steps), 0)):
                _apply_base_command(robot, world, np.zeros(0, dtype=np.float32), base_command)
                robot._debug_last_base_velocity_targets = base_command.copy()
                step_world_once()
                if step in {0, 29, 59, int(args.drive_steps) - 1}:
                    samples.append(_joint_snapshot(robot, label, step + 1))

        def run_arm_drive(label: str):
            arm_indices = np.asarray(robot.left_joint_indices, dtype=np.int32)
            gripper_indices = np.asarray(robot.left_gripper_indices, dtype=np.int32)
            initial_arm = robot._articulation_view.get_joint_positions()[0][arm_indices].astype(np.float32)
            if args.arm_target is None:
                arm_target = initial_arm.copy()
                arm_target[0] += float(args.arm_delta)
            else:
                arm_target = np.asarray(args.arm_target, dtype=np.float32).reshape(7)
            for step in range(max(int(args.drive_steps), 0)):
                if args.include_gripper_target:
                    target_positions = np.concatenate(
                        [arm_target, np.asarray(args.gripper_target, dtype=np.float32).reshape(2)]
                    )
                    target_indices = np.concatenate([arm_indices, gripper_indices]).astype(np.int32)
                else:
                    target_positions = arm_target
                    target_indices = arm_indices
                arm_target_payload = (
                    target_positions
                    if args.arm_action_shape == "1d"
                    else target_positions.reshape(1, -1)
                )
                robot.apply_action(
                    joint_positions=arm_target_payload,
                    joint_indices=target_indices,
                )
                robot._debug_last_arm_position_targets = arm_target.copy()
                if args.base_hold_during_arm:
                    _apply_base_command(robot, world, np.zeros(0, dtype=np.float32), zero_base)
                    robot._debug_last_base_velocity_targets = zero_base.copy()
                step_world_once()
                if step in {0, 29, 59, int(args.drive_steps) - 1}:
                    samples.append(_joint_snapshot(robot, label, step + 1))

        if args.base_before_arm:
            run_base_drive("base_drive_before_arm")
            _apply_base_command(robot, world, np.zeros(0, dtype=np.float32), base_command)
            robot._debug_last_base_velocity_targets = base_command.copy()
            run_arm_drive("arm_drive_after_base")
        else:
            run_arm_drive("arm_drive")
            run_base_drive("base_drive")

        _apply_base_command(robot, world, np.zeros(0, dtype=np.float32), zero_base)
        robot._debug_last_base_velocity_targets = zero_base.copy()
        for _ in range(12):
            step_world_once()
        samples.append(_joint_snapshot(robot, "final", int(args.drive_steps)))
        report.update({"status": "completed", "samples": samples})
        return_code = 0
    except Exception as exc:  # pylint: disable=broad-except
        report.update({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        return_code = 1
    finally:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(report), handle, indent=2)
        print(json.dumps(_json_safe(report), indent=2), flush=True)
        SIMULATION_APP.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
