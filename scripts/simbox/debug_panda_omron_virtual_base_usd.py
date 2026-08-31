#!/usr/bin/env python3
"""Validate PandaOmron virtual mobile-base DOFs in a minimal Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


BASE_DOF_NAMES = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default="InternDataAssets/robots/panda_omron_virtual/robot.usd")
    parser.add_argument("--output", default="output/debug_panda_omron_virtual/virtual_base_result.json")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--drive-steps", type=int, default=120)
    parser.add_argument("--stop-steps", type=int, default=30)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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


def command_profile() -> list[dict]:
    return [
        {"name": "forward", "target": [0.16, 0.0, 0.0]},
        {"name": "backward", "target": [-0.16, 0.0, 0.0]},
        {"name": "left", "target": [0.0, 0.16, 0.0]},
        {"name": "right", "target": [0.0, -0.16, 0.0]},
        {"name": "rotate_left", "target": [0.0, 0.0, 0.55]},
        {"name": "rotate_right", "target": [0.0, 0.0, -0.55]},
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
    prim_path = "/World/panda_omron_virtual"
    add_reference_to_stage(usd_path, prim_path)

    robot = world.scene.add(Robot(prim_path=prim_path, name="panda_omron_virtual"))
    base = XFormPrim(
        prim_path=f"{prim_path}/robot0_base/mobilebase0_base",
        name="panda_omron_virtual_mobilebase0_base",
    )
    world.reset()

    dof_names = list(robot.dof_names)
    missing = [name for name in BASE_DOF_NAMES if name not in dof_names]
    if missing:
        raise RuntimeError(f"Missing virtual base DOF names {missing}; available={dof_names}")
    base_indices = np.asarray([dof_names.index(name) for name in BASE_DOF_NAMES], dtype=np.int32)

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

    def read_base_velocities() -> list[float]:
        values = robot.get_joint_velocities()[base_indices]
        return [float(v) for v in list(values)]

    def apply_base(values: list[float]) -> None:
        target = np.asarray(values, dtype=np.float32).reshape(1, -1)
        robot._articulation_view.set_joint_velocity_targets(target, joint_indices=base_indices)

    zero = [0.0, 0.0, 0.0]
    for _ in range(max(int(args.settle_steps), 0)):
        apply_base(zero)
        world.step(render=False)

    segments = []
    for command in command_profile():
        target = [float(v) for v in command["target"]]
        start_pose = read_pose()
        samples = []
        for step in range(max(int(args.drive_steps), 0)):
            apply_base(target)
            world.step(render=False)
            if step in {0, 29, 59, int(args.drive_steps) - 1}:
                samples.append(
                    {
                        "step": int(step + 1),
                        "pose": read_pose(),
                        "base_velocities": read_base_velocities(),
                    }
                )
        end_pose = read_pose()
        end_base_velocities_before_stop = read_base_velocities()
        for _ in range(max(int(args.stop_steps), 0)):
            apply_base(zero)
            world.step(render=False)
        segments.append(
            {
                "name": command["name"],
                "target": target,
                "start_pose": start_pose,
                "end_pose": end_pose,
                "body_delta": body_delta(start_pose, end_pose),
                "end_base_velocities_before_stop": end_base_velocities_before_stop,
                "end_base_velocities_after_stop": read_base_velocities(),
                "samples": samples,
            }
        )

    min_expected = 0.55 * float(args.drive_steps) * float(args.physics_dt)
    checks = []
    for segment in segments:
        name = segment["name"]
        target = segment["target"]
        delta = segment["body_delta"]
        if name == "forward":
            passed = bool(delta["x"] > min_expected * target[0])
        elif name == "backward":
            passed = bool(delta["x"] < min_expected * target[0])
        elif name == "left":
            passed = bool(delta["y"] > min_expected * target[1])
        elif name == "right":
            passed = bool(delta["y"] < min_expected * target[1])
        elif name == "rotate_left":
            passed = bool(delta["yaw"] > min_expected * target[2])
        elif name == "rotate_right":
            passed = bool(delta["yaw"] < min_expected * target[2])
        else:
            passed = False
        checks.append({"name": name, "passed": passed, "body_delta": delta})

    report = {
        "usd": usd_path,
        "physics_dt": float(args.physics_dt),
        "settle_steps": int(args.settle_steps),
        "drive_steps": int(args.drive_steps),
        "stop_steps": int(args.stop_steps),
        "dof_names": dof_names,
        "base_dof_names": list(BASE_DOF_NAMES),
        "base_dof_indices": [int(v) for v in base_indices.tolist()],
        "segments": segments,
        "checks": checks,
        "passed": bool(all(item["passed"] for item in checks)),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)
    print("passed", report["passed"], flush=True)
    for item in checks:
        delta = item["body_delta"]
        print(
            item["name"],
            "passed",
            item["passed"],
            "dx",
            f"{delta['x']:.4f}",
            "dy",
            f"{delta['y']:.4f}",
            "dyaw",
            f"{delta['yaw']:.4f}",
            flush=True,
        )
    app.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
