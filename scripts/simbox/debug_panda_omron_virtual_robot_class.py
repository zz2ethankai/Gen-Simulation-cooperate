#!/usr/bin/env python3
"""Validate PandaOmronVirtual base motion through the real robot class."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml
from isaacsim import SimulationApp


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_update_dict(base: dict, override: dict):
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
    raise TypeError(f"config reference must be string or list, got {type(value).__name__}")


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
    cfg["name"] = "panda_omron_virtual_debug"
    return cfg


def _yaw_from_wxyz(q_wxyz) -> float:
    import math

    w, x, y, z = [float(v) for v in q_wxyz[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pose_dict(robot) -> dict:
    pos, quat = robot.get_mobile_base_pose()
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "yaw": float(_yaw_from_wxyz(quat)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("workflows/simbox/core/configs/robots/panda_omron_virtual.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/debug_panda_omron_virtual_robot_class/result.json"))
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--drive-steps", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = SimulationApp({"headless": bool(args.headless)})
    try:
        repo_root = _repo_root()
        sys.path.insert(0, str(repo_root / "workflows" / "simbox"))

        import numpy as np
        from core.robots.panda_omron_virtual import PandaOmronVirtual
        from omni.isaac.core import World

        world = World(stage_units_in_meters=1.0, physics_dt=float(args.physics_dt), rendering_dt=float(args.physics_dt))
        world.scene.add_default_ground_plane()

        cfg = _load_robot_cfg((repo_root / args.robot_config).resolve())
        robot = world.scene.add(
            PandaOmronVirtual(
                asset_root=str((repo_root / "InternDataAssets" / "assets").resolve()),
                root_prim_path="/World",
                cfg=cfg,
            )
        )
        world.reset()
        robot.set_mobile_base_world_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        step_dt = float(args.physics_dt)
        for _ in range(max(int(args.settle_steps), 0)):
            robot.apply_base_command([], [0.0, 0.0, 0.0], step_dt=step_dt)
            world.step(render=False)

        start_pose = _pose_dict(robot)
        command = np.asarray([0.16, 0.0, 0.0], dtype=np.float32)
        for _ in range(max(int(args.drive_steps), 0)):
            robot.apply_base_command([], command, step_dt=step_dt)
            world.step(render=False)
        end_pose = _pose_dict(robot)
        joint_state = robot.get_base_joint_state()

        result = {
            "start_pose": start_pose,
            "end_pose": end_pose,
            "delta": {
                "x": float(end_pose["x"] - start_pose["x"]),
                "y": float(end_pose["y"] - start_pose["y"]),
                "yaw": float(end_pose["yaw"] - start_pose["yaw"]),
            },
            "base_joint_positions": [float(v) for v in joint_state["wheel_positions"].reshape(-1).tolist()],
            "base_joint_velocities": [float(v) for v in joint_state["wheel_velocities"].reshape(-1).tolist()],
            "dof_names": list(robot._articulation_view.dof_names),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0 if abs(result["delta"]["x"]) > 0.05 else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
