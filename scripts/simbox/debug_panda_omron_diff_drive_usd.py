#!/usr/bin/env python3
"""Validate PandaOmron USD differential drive in a minimal Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default="InternDataAssets/assets/panda_omron/robot.usd")
    parser.add_argument("--output", default="output/debug_panda_omron_wheel_drive/usd_diff_drive_result.json")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--drive-steps", type=int, default=180)
    parser.add_argument("--stop-steps", type=int, default=45)
    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--track-width", type=float, default=0.56)
    parser.add_argument(
        "--command-profile",
        choices=("full", "slow_right_only", "slow_sweep"),
        default="full",
        help="Wheel-speed command set to execute in the minimal scene.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def yaw_to_quat_wxyz(yaw: float) -> list[float]:
    half = 0.5 * float(yaw)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def quat_wxyz_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def body_delta(start_pose: dict, end_pose: dict) -> dict:
    dx = float(end_pose["x"]) - float(start_pose["x"])
    dy = float(end_pose["y"]) - float(start_pose["y"])
    yaw0 = float(start_pose["yaw"])
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    return {
        "x": c * dx + s * dy,
        "y": -s * dx + c * dy,
        "yaw": wrap_angle(float(end_pose["yaw"]) - yaw0),
    }


def wheel_speeds_from_twist(vx: float, wz: float, *, wheel_radius: float, track_width: float) -> list[float]:
    left_linear = float(vx) - 0.5 * float(wz) * float(track_width)
    right_linear = float(vx) + 0.5 * float(wz) * float(track_width)
    return [left_linear / float(wheel_radius), right_linear / float(wheel_radius)]


def command_profile(name: str) -> list[dict]:
    if name == "slow_right_only":
        return [
            {"name": "slow_rotate_right", "vx": 0.0, "wz": -0.18},
        ]
    if name == "slow_sweep":
        return [
            {"name": "slow_rotate_right_010", "vx": 0.0, "wz": -0.10},
            {"name": "slow_rotate_right_018", "vx": 0.0, "wz": -0.18},
            {"name": "slow_rotate_right_025", "vx": 0.0, "wz": -0.25},
            {"name": "slow_rotate_right_035", "vx": 0.0, "wz": -0.35},
            {"name": "slow_rotate_left_018", "vx": 0.0, "wz": 0.18},
        ]
    return [
        {"name": "forward", "vx": 0.16, "wz": 0.0},
        {"name": "backward", "vx": -0.16, "wz": 0.0},
        {"name": "rotate_left", "vx": 0.0, "wz": 0.55},
        {"name": "rotate_right", "vx": 0.0, "wz": -0.55},
        {"name": "slow_rotate_right", "vx": 0.0, "wz": -0.18},
    ]


def main() -> int:
    args = parse_args()
    app = SimulationApp({"headless": bool(args.headless), "renderer": "RayTracedLighting"})

    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0, physics_dt=float(args.physics_dt), rendering_dt=float(args.physics_dt))
    world.scene.add_default_ground_plane()

    usd_path = str(Path(args.usd).resolve())
    prim_path = "/World/panda_omron"
    add_reference_to_stage(usd_path, prim_path)

    root = XFormPrim(prim_path=prim_path, name="panda_omron_root")
    root.set_world_pose(
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        orientation=np.asarray(yaw_to_quat_wxyz(0.0), dtype=np.float32),
    )
    robot = world.scene.add(Robot(prim_path=prim_path, name="panda_omron"))
    world.reset()

    dof_names = list(robot.dof_names)
    wheel_names = ["left_wheel_joint", "right_wheel_joint"]
    wheel_indices = [dof_names.index(name) for name in wheel_names]
    wheel_indices_array = np.asarray(wheel_indices, dtype=np.int32)
    base = XFormPrim(prim_path=f"{prim_path}/robot0_base/mobilebase0_base", name="mobilebase0_base")

    def read_pose() -> dict:
        position, quat = base.get_world_pose()
        roll, pitch, yaw = quat_wxyz_to_rpy(quat)
        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }

    def read_wheel_velocities() -> list[float]:
        values = robot.get_joint_velocities()[wheel_indices]
        return [float(v) for v in list(values)]

    def apply_wheels(values: list[float]) -> None:
        target = np.asarray(values, dtype=np.float32).reshape(1, -1)
        robot._articulation_view.set_joint_velocity_targets(target, joint_indices=wheel_indices_array)

    zero = [0.0, 0.0]
    for _ in range(max(int(args.settle_steps), 0)):
        apply_wheels(zero)
        world.step(render=False)

    commands = command_profile(args.command_profile)
    segments = []
    for command in commands:
        wheel_target = wheel_speeds_from_twist(
            command["vx"],
            command["wz"],
            wheel_radius=float(args.wheel_radius),
            track_width=float(args.track_width),
        )
        start_pose = read_pose()
        samples = []
        for step in range(max(int(args.drive_steps), 0)):
            apply_wheels(wheel_target)
            world.step(render=False)
            if step in {0, 29, 59, 119, int(args.drive_steps) - 1}:
                samples.append(
                    {
                        "step": int(step + 1),
                        "pose": read_pose(),
                        "wheel_velocities": read_wheel_velocities(),
                    }
                )
        end_pose = read_pose()
        end_wheel_velocities_before_stop = read_wheel_velocities()
        for _ in range(max(int(args.stop_steps), 0)):
            apply_wheels(zero)
            world.step(render=False)
        delta = body_delta(start_pose, end_pose)
        segments.append(
            {
                "name": command["name"],
                "command": {"vx": float(command["vx"]), "wz": float(command["wz"])},
                "wheel_target": [float(v) for v in wheel_target],
                "start_pose": start_pose,
                "end_pose": end_pose,
                "body_delta": delta,
                "end_wheel_velocities_before_stop": end_wheel_velocities_before_stop,
                "end_wheel_velocities_after_stop": read_wheel_velocities(),
                "samples": samples,
            }
        )

    checks = []
    for segment in segments:
        name = segment["name"]
        command = segment["command"]
        delta = segment["body_delta"]
        if abs(command["wz"]) > 0.0:
            passed = bool(delta["yaw"] * command["wz"] > 0.0 and abs(delta["yaw"]) > 0.65 * abs(command["wz"]) * args.drive_steps * args.physics_dt)
        elif command["vx"] > 0.0:
            passed = bool(delta["x"] > 0.65 * command["vx"] * args.drive_steps * args.physics_dt and abs(delta["yaw"]) < 0.20)
        elif command["vx"] < 0.0:
            passed = bool(delta["x"] < 0.65 * command["vx"] * args.drive_steps * args.physics_dt and abs(delta["yaw"]) < 0.20)
        else:
            passed = True
        checks.append({"name": name, "passed": passed, "body_delta": segment["body_delta"]})

    report = {
        "usd": usd_path,
        "physics_dt": float(args.physics_dt),
        "settle_steps": int(args.settle_steps),
        "drive_steps": int(args.drive_steps),
        "stop_steps": int(args.stop_steps),
        "wheel_radius": float(args.wheel_radius),
        "track_width": float(args.track_width),
        "command_profile": str(args.command_profile),
        "dof_names": dof_names,
        "wheel_names": wheel_names,
        "wheel_indices": wheel_indices,
        "segments": segments,
        "checks": checks,
        "passed": bool(all(item["passed"] for item in checks)),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print("passed", report["passed"])
    for item in checks:
        delta = item["body_delta"]
        print(
            item["name"],
            "passed",
            item["passed"],
            "dx",
            f"{delta['x']:.4f}",
            "dyaw",
            f"{delta['yaw']:.4f}",
        )
    app.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
