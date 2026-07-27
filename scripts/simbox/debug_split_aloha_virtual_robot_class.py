#!/usr/bin/env python3
"""Validate SplitAloha's virtual base through the real Isaac robot class."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import yaml
from isaacsim import SimulationApp


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_update_dict(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _iter_config_refs(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Config reference must be a string or list, got {type(value).__name__}")


def _load_robot_cfg(robot_config: Path) -> dict:
    repo_root = _repo_root()
    cfg = _load_yaml(robot_config)
    base_override = deepcopy(cfg.get("base", {}))
    merged_base = {}
    for ref in _iter_config_refs(base_override.get("base_config_file")):
        _deep_update_dict(merged_base, _load_yaml((repo_root / ref).resolve()))
    for ref in _iter_config_refs(base_override.get("nav_config_file")):
        _deep_update_dict(merged_base, _load_yaml((repo_root / ref).resolve()))
    _deep_update_dict(merged_base, base_override)
    cfg["base"] = merged_base
    cfg["name"] = "split_aloha_virtual_debug"
    return cfg


def _yaw_from_wxyz(q_wxyz) -> float:
    w, x, y, z = [float(v) for v in q_wxyz[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_from_wxyz(q_wxyz) -> tuple[float, float]:
    w, x, y, z = [float(v) for v in q_wxyz[:4]]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(sin_pitch)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _pose_dict(robot) -> dict:
    position, orientation = robot.get_mobile_base_pose()
    roll, pitch = _roll_pitch_from_wxyz(orientation)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(_yaw_from_wxyz(orientation)),
    }


def _pose_delta(start: dict, end: dict) -> dict:
    return {
        "x": float(end["x"] - start["x"]),
        "y": float(end["y"] - start["y"]),
        "z": float(end["z"] - start["z"]),
        "yaw": float(_wrap_angle(end["yaw"] - start["yaw"])),
    }


def _drive(robot, world, command, *, steps: int, step_dt: float) -> tuple[dict, dict, dict]:
    start = _pose_dict(robot)
    commanded_joint_state = None
    for _ in range(max(int(steps), 0)):
        robot.apply_base_command([], command, step_dt=step_dt)
        world.step(render=False)
        state = robot.get_base_joint_state()
        commanded_joint_state = {
            "positions": [float(value) for value in state["wheel_positions"].reshape(-1).tolist()],
            "velocities": [float(value) for value in state["wheel_velocities"].reshape(-1).tolist()],
        }
    robot.apply_base_command([], [0.0, 0.0, 0.0], step_dt=step_dt)
    for _ in range(5):
        world.step(render=False)
    end = _pose_dict(robot)
    delta = _pose_delta(start, end)
    delta["commanded_joint_state"] = commanded_joint_state
    return start, end, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("workflows/simbox/core/configs/robots/split_aloha.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/debug_split_aloha_virtual_robot_class/result.json"),
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--drive-steps", type=int, default=90)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--ground", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = SimulationApp({"headless": bool(args.headless)})
    result = {"status": "started"}
    try:
        repo_root = _repo_root()
        sys.path.insert(0, str(repo_root / "workflows" / "simbox"))

        import numpy as np
        from core.robots.split_aloha import SplitAloha
        from omni.isaac.core import World
        from omni.isaac.core.utils.xforms import get_world_pose as get_prim_world_pose
        from workflows.simbox.core.mobile.platforms import get_mobile_base_platform

        cfg = _load_robot_cfg((repo_root / args.robot_config).resolve())
        if cfg.get("target_class") != "SplitAloha":
            raise ValueError(f"Probe requires target_class SplitAloha, got {cfg.get('target_class')!r}")

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(args.physics_dt),
            rendering_dt=float(args.physics_dt),
        )
        if args.ground:
            world.scene.add_default_ground_plane()
        robot = world.scene.add(
            SplitAloha(
                asset_root=str((repo_root / "InternDataAssets" / "assets").resolve()),
                root_prim_path="/World",
                cfg=cfg,
            )
        )
        world.reset()

        step_dt = float(args.physics_dt)
        robot.set_mobile_base_world_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        for _ in range(max(int(args.settle_steps), 0)):
            robot.apply_base_command([], [0.0, 0.0, 0.0], step_dt=step_dt)
            world.step(render=False)

        dof_names = list(robot._articulation_view.dof_names)
        base_interface = robot.get_base_interface()
        base_names = list(base_interface["wheel_joint_names"])
        base_indices = list(base_interface["wheel_joint_indices"])
        manip_indices = (
            list(robot.left_joint_indices)
            + list(robot.right_joint_indices)
            + list(robot.left_gripper_indices)
            + list(robot.right_gripper_indices)
        )
        arm_before = robot._articulation_view.get_joint_positions()[0][manip_indices].copy()
        root_pose_before = get_prim_world_pose(robot.robot_prim_path)

        x_start, x_end, x_delta = _drive(
            robot,
            world,
            [0.16, 0.0, 0.0],
            steps=args.drive_steps,
            step_dt=step_dt,
        )
        y_start, y_end, y_delta = _drive(
            robot,
            world,
            [0.0, 0.14, 0.0],
            steps=args.drive_steps,
            step_dt=step_dt,
        )
        yaw_start, yaw_end, yaw_delta = _drive(
            robot,
            world,
            [0.0, 0.0, 0.35],
            steps=args.drive_steps,
            step_dt=step_dt,
        )
        body_start, body_end, body_delta = _drive(
            robot,
            world,
            [0.16, 0.0, 0.0],
            steps=args.drive_steps,
            step_dt=step_dt,
        )

        heading = float(body_start["yaw"])
        body_forward = math.cos(heading) * body_delta["x"] + math.sin(heading) * body_delta["y"]
        body_lateral = -math.sin(heading) * body_delta["x"] + math.cos(heading) * body_delta["y"]
        arm_after = robot._articulation_view.get_joint_positions()[0][manip_indices].copy()
        root_pose_after = get_prim_world_pose(robot.robot_prim_path)
        observations = robot.get_observations()
        observed_world_base = np.asarray(observations["T_world_base"], dtype=np.float64)
        observed_base_translation = observed_world_base[:3, 3]

        reset_position = np.asarray([0.30, -0.20, 0.0], dtype=np.float32)
        reset_yaw = -0.30
        reset_orientation = [math.cos(0.5 * reset_yaw), 0.0, 0.0, math.sin(0.5 * reset_yaw)]
        robot.reset_mobile_base_world_state(reset_position, reset_orientation)
        for _ in range(5):
            world.step(render=False)
        reset_pose = _pose_dict(robot)
        base_state_after_reset = robot.get_base_joint_state()

        checks = {
            "class_is_split_aloha": robot.__class__.__name__ == "SplitAloha",
            "platform_is_virtual": get_mobile_base_platform(robot.base_cfg).__class__.__name__ == "VirtualBasePlatform",
            "base_names_match": base_names == ["mobile_translate_x", "mobile_translate_y", "mobile_rotate"],
            "base_indices_are_distinct": len(base_indices) == 3 and len(set(base_indices)) == 3,
            "base_manipulator_disjoint": not (set(base_indices) & set(manip_indices)),
            "physical_wheels_removed_from_dofs": not any(
                name in dof_names
                for name in (
                    "fl_wheel",
                    "fr_wheel",
                    "rl_wheel",
                    "rr_wheel",
                    "fl_steering_joint",
                    "fr_steering_joint",
                    "rl_steering_joint",
                    "rr_steering_joint",
                )
            ),
            "x_command_moves_positive_x": x_delta["x"] > 0.08 and abs(x_delta["y"]) < 0.04,
            "y_command_moves_positive_y": y_delta["y"] > 0.07 and abs(y_delta["x"]) < 0.04,
            "yaw_command_rotates_positive": yaw_delta["yaw"] > 0.20,
            "body_x_follows_rotated_heading": body_forward > 0.08 and abs(body_lateral) < 0.04,
            "base_height_stable": max(abs(pose["z"]) for pose in (x_end, y_end, yaw_end, body_end)) < 0.02,
            "base_roll_pitch_stable": max(
                abs(value)
                for pose in (x_end, y_end, yaw_end, body_end)
                for value in (pose["roll"], pose["pitch"])
            ) < 0.02,
            "manipulator_targets_stable": float(np.max(np.abs(arm_after - arm_before))) < 0.03,
            "articulation_root_xy_fixed": float(
                np.linalg.norm(np.asarray(root_pose_after[0][:2]) - np.asarray(root_pose_before[0][:2]))
            ) < 0.01,
            "observation_tracks_mobile_base": float(
                np.linalg.norm(
                    observed_base_translation
                    - np.asarray([body_end["x"], body_end["y"], body_end["z"]], dtype=np.float64)
                )
            ) < 0.01,
            "reset_pose_restored": (
                abs(reset_pose["x"] - float(reset_position[0])) < 0.02
                and abs(reset_pose["y"] - float(reset_position[1])) < 0.02
                and abs(_wrap_angle(reset_pose["yaw"] - reset_yaw)) < 0.02
            ),
            "reset_virtual_joints_zero": float(
                np.max(np.abs(base_state_after_reset["wheel_positions"]))
            ) < 0.01,
        }
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "dof_names": dof_names,
            "base_joint_names": base_names,
            "base_joint_indices": base_indices,
            "manipulator_joint_indices": manip_indices,
            "segments": {
                "x": {"start": x_start, "end": x_end, "delta": x_delta},
                "y": {"start": y_start, "end": y_end, "delta": y_delta},
                "yaw": {"start": yaw_start, "end": yaw_end, "delta": yaw_delta},
                "body_x_after_yaw": {
                    "start": body_start,
                    "end": body_end,
                    "delta": body_delta,
                    "forward_projection": float(body_forward),
                    "lateral_projection": float(body_lateral),
                },
            },
            "max_manipulator_drift": float(np.max(np.abs(arm_after - arm_before))),
            "observed_T_world_base": observed_world_base.tolist(),
            "reset_pose": reset_pose,
            "reset_base_joint_positions": [
                float(value) for value in base_state_after_reset["wheel_positions"].reshape(-1).tolist()
            ],
        }
        return_code = 0 if result["status"] == "passed" else 1
    except Exception as exc:  # pylint: disable=broad-except
        import traceback

        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 2
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        app.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
