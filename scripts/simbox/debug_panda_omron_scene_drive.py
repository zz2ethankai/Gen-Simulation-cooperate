#!/usr/bin/env python3
"""Drive PandaOmron wheel joints inside a real SimBox task scene."""

from __future__ import annotations

import argparse
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

from nimbus.utils.utils import init_env  # noqa: E402  pylint: disable=wrong-import-position
from workflows import import_extensions  # noqa: E402  pylint: disable=wrong-import-position
from workflows.base import create_workflow  # noqa: E402  pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/de_plan_with_render_scene8_validation.yaml")
    parser.add_argument("--output", default="output/debug_panda_omron_wheel_drive/scene_drive_result.json")
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--drive-steps", type=int, default=900)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--keep-bridge", action="store_true")
    parser.add_argument("--trace-reset-only", action="store_true")
    parser.add_argument(
        "--wheel-speeds",
        type=float,
        nargs=2,
        default=[-0.86, -1.04],
        metavar=("LEFT", "RIGHT"),
    )
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


def _find_panda_omron(workflow):
    for robot in workflow.task.robots.values():
        if robot.__class__.__name__ == "PandaOmron":
            return robot
    raise RuntimeError("PandaOmron robot not found in workflow task")


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _record(robot, label: str, step: int) -> dict:
    from omni.isaac.core.utils.xforms import get_world_pose  # pylint: disable=import-outside-toplevel

    pose, quat = robot.get_mobile_base_pose()
    root_pose, root_quat = get_world_pose(robot.robot_prim_path)
    base_state = robot.get_base_joint_state()
    sample = {
        "label": str(label),
        "step": int(step),
        "root_position": [float(v) for v in list(root_pose)],
        "root_yaw": float(_yaw_from_wxyz(root_quat)),
        "mobile_base_position": [float(v) for v in list(pose)],
        "mobile_base_yaw": float(_yaw_from_wxyz(quat)),
        "wheel_positions": [float(v) for v in np.asarray(base_state["wheel_positions"]).reshape(-1).tolist()],
        "wheel_velocities": [float(v) for v in np.asarray(base_state["wheel_velocities"]).reshape(-1).tolist()],
    }
    sample["contact_prim_poses"] = _record_contact_prim_poses(robot)
    return sample


def _record_contact_prim_poses(robot) -> dict:
    from omni.isaac.core.prims import XFormPrim  # pylint: disable=import-outside-toplevel
    from omni.isaac.core.utils.prims import get_prim_at_path  # pylint: disable=import-outside-toplevel

    names = [
        "left_wheel_link",
        "right_wheel_link",
        "mobilebase0_base/collisions/front_left_support",
        "mobilebase0_base/collisions/front_right_support",
        "mobilebase0_base/collisions/rear_left_support",
        "mobilebase0_base/collisions/rear_right_support",
        "mobilebase0_base/collisions/panda_omron_chassis_collision",
    ]
    poses = {}
    for name in names:
        prim_path = f"{robot.robot_prim_path}/robot0_base/{name}"
        if not get_prim_at_path(prim_path).IsValid():
            poses[name] = {"exists": False}
            continue
        prim = XFormPrim(prim_path=prim_path)
        pos, quat = prim.get_world_pose()
        poses[name] = {
            "exists": True,
            "position": [float(v) for v in list(pos)],
            "yaw": float(_yaw_from_wxyz(quat)),
        }
    return poses


def _record_prim_poses(robot) -> dict:
    from omni.isaac.core.prims import XFormPrim  # pylint: disable=import-outside-toplevel
    from omni.isaac.core.utils.prims import get_prim_at_path  # pylint: disable=import-outside-toplevel

    prim_paths = [
        robot.robot_prim_path,
        f"{robot.robot_prim_path}/robot0_base",
        f"{robot.robot_prim_path}/robot0_base/robot0_base",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_base",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_wheeled_base",
        f"{robot.robot_prim_path}/robot0_base/left_wheel_link",
        f"{robot.robot_prim_path}/robot0_base/right_wheel_link",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_base/collisions/front_left_support",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_base/collisions/front_right_support",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_base/collisions/rear_left_support",
        f"{robot.robot_prim_path}/robot0_base/mobilebase0_base/collisions/rear_right_support",
        f"{robot.robot_prim_path}/robot0_base/robot0_link0",
        f"{robot.robot_prim_path}/robot0_base/panda_hand",
    ]
    poses = {}
    for prim_path in prim_paths:
        if not get_prim_at_path(prim_path).IsValid():
            poses[prim_path] = {"exists": False}
            continue
        prim = XFormPrim(prim_path=prim_path)
        pos, quat = prim.get_world_pose()
        poses[prim_path] = {
            "exists": True,
            "position": [float(v) for v in list(pos)],
            "yaw": float(_yaw_from_wxyz(quat)),
            "quat": [float(v) for v in list(quat)],
        }
    return poses


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
        "wheel_speeds": [float(v) for v in args.wheel_speeds],
        "settle_steps": int(args.settle_steps),
        "drive_steps": int(args.drive_steps),
        "randomize": bool(args.randomize),
        "keep_bridge": bool(args.keep_bridge),
    }
    try:
        init_env()
        config = _load_config(args.config)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        simulator_cfg = scene_loader_cfg["simulator"]

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, scene_loader_cfg["cfg_path"])
        workflow.init_task(0)
        robot = _find_panda_omron(workflow)
        trace_samples = [_record(robot, "after_init_task", 0)]
        if args.randomize:
            workflow.randomization()
            trace_samples.append(_record(robot, "after_randomization", 0))

        if args.trace_reset_only:
            trace_points = {
                0,
                1,
                2,
                3,
                4,
                5,
                9,
                14,
                19,
                29,
                59,
                89,
                119,
                179,
                239,
            }
            if hasattr(workflow, "_reset_fixed_robot_start_states_after_physics"):
                workflow._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
                trace_samples.append(_record(robot, "after_fixed_reset", 0))
            for step in range(max(int(args.settle_steps), 0)):
                workflow._step_world(render=False)
                if step in trace_points:
                    trace_samples.append(_record(robot, "post_reset_step", step + 1))
            report.update(
                {
                    "status": "completed",
                    "robot_name": robot.name,
                    "robot_cfg": next(cfg for cfg in workflow.task.cfg["robots"] if cfg["name"] == robot.name),
                    "trace_samples": trace_samples,
                    "base_interface": robot.get_base_interface(),
                    "prim_poses": _record_prim_poses(robot),
                }
            )
            return_code = 0
            return return_code

        dof_names = list(robot._articulation_view.dof_names)
        robot_cfg = next(cfg for cfg in workflow.task.cfg["robots"] if cfg["name"] == robot.name)
        robot_regions = [
            dict(cfg)
            for cfg in workflow.task.cfg.get("regions", [])
            if cfg.get("object") == robot.name
        ]
        samples = []
        zero_wheels = np.zeros(len(robot.base_wheel_joint_indices), dtype=np.float32)
        target_wheels = np.asarray(args.wheel_speeds, dtype=np.float32)
        if target_wheels.shape[0] != len(robot.base_wheel_joint_indices):
            raise ValueError("wheel-speeds length must match PandaOmron wheel joint count")

        samples.append(_record(robot, "initial", 0))
        for step in range(max(int(args.settle_steps), 0)):
            robot.apply_base_command(np.zeros(0, dtype=np.float32), zero_wheels)
            workflow._step_world(render=False)
            if step in {0, 29, 59, 119, int(args.settle_steps) - 1}:
                samples.append(_record(robot, "settle", step + 1))

        samples.append(_record(robot, "drive_start", 0))
        for step in range(max(int(args.drive_steps), 0)):
            robot.apply_base_command(np.zeros(0, dtype=np.float32), target_wheels)
            workflow._step_world(render=False)
            if step in {0, 29, 59, 119, 239, 479, int(args.drive_steps) - 1}:
                samples.append(_record(robot, "drive", step + 1))
        samples.append(_record(robot, "drive_end", int(args.drive_steps)))

        robot.apply_base_command(np.zeros(0, dtype=np.float32), zero_wheels)
        for _ in range(12):
            workflow._step_world(render=False)

        report.update(
            {
                "status": "completed",
                "robot_name": robot.name,
                "robot_cfg": robot_cfg,
                "robot_regions": robot_regions,
                "dof_names": dof_names,
                "base_interface": robot.get_base_interface(),
                "prim_poses": _record_prim_poses(robot),
                "samples": samples,
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        report.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    else:
        return_code = 0
    finally:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(report), handle, indent=2)
        print(json.dumps(_json_safe(report), indent=2))
        SIMULATION_APP.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
