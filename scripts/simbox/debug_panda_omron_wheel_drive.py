#!/usr/bin/env python3
"""Run a minimal PandaOmron wheel-drive physics check in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default="InternDataAssets/robots/panda_omron/robot.usd")
    parser.add_argument("--output", default="output/debug_panda_omron_wheel_drive/result.json")
    parser.add_argument("--settle-steps", type=int, default=90)
    parser.add_argument("--drive-steps", type=int, default=180)
    parser.add_argument("--wheel-speed", type=float, default=4.0)
    return parser.parse_args()


def quat_wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    args = parse_args()
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    usd_path = str(Path(args.usd).resolve())
    prim_path = "/World/panda_omron"
    add_reference_to_stage(usd_path, prim_path)
    xform = XFormPrim(prim_path=prim_path, name="panda_omron_xform")
    xform.set_world_pose(position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    robot = world.scene.add(Robot(prim_path=prim_path, name="panda_omron"))
    world.reset()

    dof_names = list(robot.dof_names)
    wheel_names = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    wheel_indices = [dof_names.index(name) for name in wheel_names]
    base = XFormPrim(prim_path=f"{prim_path}/robot0_base/mobilebase0_base", name="mobilebase0_base")
    wheel_links = {
        name: XFormPrim(prim_path=f"{prim_path}/robot0_base/{name.replace('_joint', '_link')}", name=name)
        for name in wheel_names
    }

    samples = []

    def record(label: str, step: int) -> None:
        pos, quat = base.get_world_pose()
        joint_vel = robot.get_joint_velocities()[wheel_indices]
        wheel_pose = {}
        for wheel_name, wheel_prim in wheel_links.items():
            wpos, _wquat = wheel_prim.get_world_pose()
            wheel_pose[wheel_name] = [float(v) for v in list(wpos)]
        samples.append(
            {
                "label": label,
                "step": int(step),
                "base_position": [float(v) for v in list(pos)],
                "base_yaw": float(quat_wxyz_to_yaw(quat)),
                "wheel_velocities": [float(v) for v in list(joint_vel)],
                "wheel_positions": wheel_pose,
            }
        )

    record("initial", 0)
    wheel_indices_array = np.asarray(wheel_indices, dtype=np.int32)
    zero = np.zeros((1, len(wheel_indices)), dtype=np.float32)
    for step in range(int(args.settle_steps)):
        robot._articulation_view.set_joint_velocity_targets(zero, joint_indices=wheel_indices_array)
        world.step(render=False)
        if step in {0, int(args.settle_steps) - 1}:
            record("settle", step + 1)

    commands = [
        ("all_positive", [args.wheel_speed, args.wheel_speed]),
        ("all_negative", [-args.wheel_speed, -args.wheel_speed]),
        (
            "diff_left_negative_right_positive",
            [-args.wheel_speed, args.wheel_speed],
        ),
        (
            "diff_left_positive_right_negative",
            [args.wheel_speed, -args.wheel_speed],
        ),
    ]
    for label, command in commands:
        target = np.asarray(command, dtype=np.float32).reshape(1, -1)
        record(f"{label}_start", 0)
        for step in range(int(args.drive_steps)):
            robot._articulation_view.set_joint_velocity_targets(target, joint_indices=wheel_indices_array)
            world.step(render=False)
        record(f"{label}_end", int(args.drive_steps))
        robot._articulation_view.set_joint_velocity_targets(zero, joint_indices=wheel_indices_array)
        for _ in range(30):
            world.step(render=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "usd": usd_path,
                "dof_names": dof_names,
                "wheel_names": wheel_names,
                "wheel_indices": wheel_indices,
                "wheel_speed": float(args.wheel_speed),
                "samples": samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    app.close()


if __name__ == "__main__":
    main()
