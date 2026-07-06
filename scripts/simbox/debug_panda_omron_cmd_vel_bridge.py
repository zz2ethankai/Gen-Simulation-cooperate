#!/usr/bin/env python3
"""Probe PandaOmron /cmd_vel -> mobile-base bridge in a SimBox scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import cv2  # noqa: F401
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
from omni.physx import acquire_physx_interface  # noqa: E402  pylint: disable=wrong-import-position
from yaml import Loader  # noqa: E402  pylint: disable=wrong-import-position

from nimbus.utils.utils import init_env  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.core.mobile import build_mobile_base_bridge  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.core.tasks import get_task_cls  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.core.utils.collision_utils import filter_collisions  # noqa: E402  pylint: disable=wrong-import-position
from workflows.simbox.utils.task_config_parser import TaskConfigParser  # noqa: E402  pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/de_plan_with_render_scene8_validation.yaml")
    parser.add_argument("--output", default="output/debug_panda_omron_wheel_drive/cmd_vel_bridge_result.json")
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--drive-steps", type=int, default=180)
    parser.add_argument("--stop-steps", type=int, default=60)
    parser.add_argument("--vx", type=float, default=-0.0757)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.1913)
    return parser.parse_args()


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _eval_dt(value) -> float:
    if isinstance(value, str):
        return float(eval(value, {"__builtins__": {}}, {}))
    return float(value)


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
        acquire_physx_interface().overwrite_gpu_setting(1)

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

    robot_paths = [task.root_prim_path + "/" + robot_cfg["name"] for robot_cfg in task_cfg["robots"]]
    global_collision_paths = []
    for candidate in task_cfg["objects"] + task_cfg["arena"]["fixtures"]:
        candidate_prim_path = task.root_prim_path + "/" + candidate["name"]
        global_collision_paths.append(candidate_prim_path)
        for neglect_collision_name in task_cfg.get("neglect_collision_names", []):
            if neglect_collision_name in candidate["name"]:
                robot_paths.append(candidate_prim_path)
                global_collision_paths.remove(candidate_prim_path)

    filter_collisions(
        stage,
        world.get_physics_context().prim_path,
        "/World/collisions",
        robot_paths,
        global_collision_paths,
    )
    world.reset()
    world.step(render=False)
    task.set_fixed_robot_start_poses()
    world.step(render=False)
    for _ in range(20):
        world.step(render=False)
    return task


def _find_panda_omron(task):
    available = []
    for robot in task.robots.values():
        class_name = robot.__class__.__name__
        available.append(class_name)
        if class_name in {"PandaOmron", "PandaOmronVirtual"}:
            return robot
    raise RuntimeError(f"PandaOmron robot not found in workflow task; available={available}")


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _record(robot, bridge, label: str, step: int) -> dict:
    pose, quat = robot.get_mobile_base_pose()
    nav_pose, nav_quat = robot.get_nav_base_pose()
    base_state = robot.get_base_joint_state()
    action_snapshot = bridge.get_logging_action_snapshot()
    state_snapshot = bridge.get_logging_state_snapshot()
    return {
        "label": str(label),
        "step": int(step),
        "mobile_base_position": [float(v) for v in list(pose)],
        "mobile_base_yaw": float(_yaw_from_wxyz(quat)),
        "nav_base_position": [float(v) for v in list(nav_pose)],
        "nav_base_yaw": float(_yaw_from_wxyz(nav_quat)),
        "wheel_positions": [float(v) for v in np.asarray(base_state["wheel_positions"]).reshape(-1).tolist()],
        "wheel_velocities": [float(v) for v in np.asarray(base_state["wheel_velocities"]).reshape(-1).tolist()],
        "bridge_action": action_snapshot,
        "bridge_state": state_snapshot,
    }


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


def _make_twist(bridge, *, vx: float, vy: float, wz: float):
    msg = bridge._Twist()  # pylint: disable=protected-access
    msg.linear.x = float(vx)
    msg.linear.y = float(vy)
    msg.angular.z = float(wz)
    return msg


def _step_world_with_bridge(world: World, bridge, *, render: bool = False):
    step_dt = float(world.get_physics_dt()) if callable(getattr(world, "get_physics_dt", None)) else 1.0 / 60.0
    bridge.step(step_dt=step_dt)
    world.step(render=render)


def run() -> int:
    args = parse_args()
    report = {
        "status": "started",
        "config": args.config,
        "cmd_vel": {"vx": float(args.vx), "vy": float(args.vy), "wz": float(args.wz)},
        "warmup_steps": int(args.warmup_steps),
        "drive_steps": int(args.drive_steps),
        "stop_steps": int(args.stop_steps),
    }
    try:
        init_env()
        config = _load_config(args.config)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        simulator_cfg = scene_loader_cfg["simulator"]

        world = _build_world(simulator_cfg)
        task = _build_manual_task(args.config, world)
        robot = _find_panda_omron(task)
        bridge = build_mobile_base_bridge(robot, node_name=f"{robot.name}_cmd_vel_probe_bridge")

        bridge.reset(clear_debug_history=True)
        bridge.prepare_for_navigation()
        command_pub = bridge.node.create_publisher(bridge._Twist, bridge.ros_cfg["cmd_vel_topic"], 10)  # pylint: disable=protected-access

        samples = [_record(robot, bridge, "initial", 0)]
        for step in range(max(int(args.warmup_steps), 0)):
            _step_world_with_bridge(world, bridge, render=False)
            if step in {0, 29, 59, int(args.warmup_steps) - 1}:
                samples.append(_record(robot, bridge, "warmup", step + 1))

        cmd_msg = _make_twist(bridge, vx=args.vx, vy=args.vy, wz=args.wz)
        for step in range(max(int(args.drive_steps), 0)):
            command_pub.publish(cmd_msg)
            _step_world_with_bridge(world, bridge, render=False)
            if step in {0, 1, 4, 9, 29, 59, 119, int(args.drive_steps) - 1}:
                samples.append(_record(robot, bridge, "drive", step + 1))

        zero_msg = _make_twist(bridge, vx=0.0, vy=0.0, wz=0.0)
        for step in range(max(int(args.stop_steps), 0)):
            command_pub.publish(zero_msg)
            _step_world_with_bridge(world, bridge, render=False)
            if step in {0, 9, 29, int(args.stop_steps) - 1}:
                samples.append(_record(robot, bridge, "stop", step + 1))

        report.update(
            {
                "status": "completed",
                "robot_name": robot.name,
                "base_interface": robot.get_base_interface(),
                "received_cmd_vel_count": int(getattr(bridge, "_received_cmd_vel_count", 0)),
                "applied_driver_command_count": int(getattr(bridge, "_applied_driver_command_count", 0)),
                "last_received_cmd_vel": dict(getattr(bridge, "_last_received_cmd_vel", {}) or {}),
                "recent_cmd_vel_history": list(getattr(bridge, "_debug_cmd_vel_history", []))[-20:],
                "recent_command_history": list(getattr(bridge, "_debug_command_history", []))[-20:],
                "samples": samples,
            }
        )
        return_code = 0
    except Exception as exc:  # pylint: disable=broad-except
        report.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    finally:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(report), handle, indent=2)
        print(json.dumps(_json_safe(report), indent=2))
        SIMULATION_APP.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
