#!/usr/bin/env python3
"""Check PandaOmron wheel response under local differential-drive commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default="InternDataAssets/assets/panda_omron/robot.usd")
    parser.add_argument("--output", default="output/debug_panda_omron_wheel_drive/nav_command_result.json")
    parser.add_argument("--floor", choices=("default", "simbox"), default="simbox")
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--drive-steps", type=int, default=900)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--yaw-deg", type=float, default=157.5)
    parser.add_argument(
        "--wheel-speeds",
        type=float,
        nargs=2,
        default=[-1.52, -0.26],
        metavar=("LEFT", "RIGHT"),
    )
    return parser.parse_args()


def quat_wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_wxyz(yaw: float):
    half = 0.5 * float(yaw)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def main() -> None:
    args = parse_args()
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    world = World(stage_units_in_meters=1.0, physics_dt=float(args.physics_dt), rendering_dt=float(args.physics_dt))
    if args.floor == "default":
        world.scene.add_default_ground_plane()
    else:
        stage = get_current_stage()
        floor = UsdGeom.Cube.Define(stage, "/World/simbox_floor_collision")
        floor.CreateSizeAttr().Set(1.0)
        xform = UsdGeom.Xformable(floor.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3f(2.0, 1.5, -0.01))
        xform.AddScaleOp().Set(Gf.Vec3f(4.0, 3.0, 0.02))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim()).CreateCollisionEnabledAttr().Set(True)

    usd_path = str(Path(args.usd).resolve())
    prim_path = "/World/panda_omron"
    add_reference_to_stage(usd_path, prim_path)
    root = XFormPrim(prim_path=prim_path, name="panda_omron_xform")
    yaw = math.radians(float(args.yaw_deg))
    root.set_world_pose(
        position=np.asarray([2.0, 1.5, 0.0], dtype=np.float32),
        orientation=np.asarray(yaw_to_quat_wxyz(yaw), dtype=np.float32),
    )
    robot = world.scene.add(Robot(prim_path=prim_path, name="panda_omron"))
    world.reset()

    wheel_names = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    dof_names = list(robot.dof_names)
    wheel_indices = [dof_names.index(name) for name in wheel_names]
    wheel_indices_array = np.asarray(wheel_indices, dtype=np.int32)
    base = XFormPrim(prim_path=f"{prim_path}/robot0_base/mobilebase0_base", name="mobilebase0_base")

    samples = []

    def record(label: str, step: int) -> None:
        pos, quat = base.get_world_pose()
        joint_vel = robot.get_joint_velocities()[wheel_indices]
        samples.append(
            {
                "label": str(label),
                "step": int(step),
                "base_position": [float(v) for v in list(pos)],
                "base_yaw": float(quat_wxyz_to_yaw(quat)),
                "wheel_velocities": [float(v) for v in list(joint_vel)],
            }
        )

    zero = np.zeros((1, len(wheel_indices)), dtype=np.float32)
    target = np.asarray(args.wheel_speeds, dtype=np.float32).reshape(1, -1)

    record("initial", 0)
    for step in range(int(args.settle_steps)):
        robot._articulation_view.set_joint_velocity_targets(zero, joint_indices=wheel_indices_array)
        world.step(render=False)
        if step in {0, 29, 59, 119, int(args.settle_steps) - 1}:
            record("settle", step + 1)

    record("drive_start", 0)
    for step in range(int(args.drive_steps)):
        robot._articulation_view.set_joint_velocity_targets(target, joint_indices=wheel_indices_array)
        world.step(render=False)
        if step in {0, 29, 59, 119, 239, 479, int(args.drive_steps) - 1}:
            record("drive", step + 1)
    record("drive_end", int(args.drive_steps))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "usd": usd_path,
                "floor": args.floor,
                "physics_dt": float(args.physics_dt),
                "yaw_deg": float(args.yaw_deg),
                "wheel_names": wheel_names,
                "wheel_indices": wheel_indices,
                "wheel_speeds": [float(v) for v in args.wheel_speeds],
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
