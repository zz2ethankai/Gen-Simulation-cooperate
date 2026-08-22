#!/usr/bin/env python3
"""Short Isaac check for PandaOmronVirtual base hold and arm partitioning.

The check drives the virtual base under its navigation drives, resumes the
manipulation hold, then sends an arm-only action for a short physics window.
It reports base-joint drift and fails if any arm/gripper index overlaps a base
index.  The script is intentionally independent of a task YAML.
"""

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


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _deep_update(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _load_robot_cfg(path: Path) -> dict:
    repo_root = _repo_root()
    cfg = _load_yaml(path)
    base_override = deepcopy(cfg.get("base", {}))
    base = {}
    for key in ("base_config_file", "local_navigation_config_file"):
        refs = base_override.get(key, [])
        refs = [refs] if isinstance(refs, str) else list(refs or [])
        for ref in refs:
            _deep_update(base, _load_yaml((repo_root / str(ref)).resolve()))
    _deep_update(base, base_override)
    cfg["base"] = base
    cfg["name"] = "panda_omron_virtual_base_hold_debug"
    return cfg


def _yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = [float(value) for value in quaternion[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("workflows/simbox/core/configs/robots/panda_omron_virtual.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/debug_panda_omron_virtual_base_hold/result.json"))
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--navigation-steps", type=int, default=120)
    parser.add_argument("--manipulation-steps", type=int, default=240)
    parser.add_argument("--command-vx", type=float, default=0.16)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-wz", type=float, default=0.0)
    parser.add_argument("--hold-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = SimulationApp({"headless": True})
    report = {"status": "started"}
    try:
        repo_root = _repo_root()
        sys.path.insert(0, str(repo_root / "workflows" / "simbox"))
        import numpy as np
        from core.robots.panda_omron_virtual import PandaOmronVirtual
        from isaacsim.core.api import World

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(args.physics_dt),
            rendering_dt=float(args.physics_dt),
        )
        world.scene.add_default_ground_plane()
        robot = world.scene.add(
            PandaOmronVirtual(
                asset_root=str((repo_root / "InternDataAssets" / "assets").resolve()),
                root_prim_path="/World",
                cfg=_load_robot_cfg((repo_root / args.robot_config).resolve()),
            )
        )
        world.reset()

        dof_names = list(robot._articulation_view.dof_names)
        arm_indices = np.asarray(robot.left_joint_indices, dtype=np.int64)
        gripper_indices = np.asarray(robot.left_gripper_indices, dtype=np.int64)
        base_indices = np.asarray(robot.base_wheel_joint_indices, dtype=np.int64)
        action_indices = np.concatenate([arm_indices, gripper_indices])
        overlap = sorted(set(action_indices.tolist()) & set(base_indices.tolist()))
        if overlap:
            raise AssertionError(
                f"arm/gripper action overlaps base indices {overlap}; dofs={dof_names}"
            )

        robot.enable_manipulation_base_hold()
        robot.suspend_manipulation_base_hold()
        command = np.asarray([args.command_vx, args.command_vy, args.command_wz], dtype=np.float32)
        dt = float(args.physics_dt)
        for _ in range(max(int(args.navigation_steps), 0)):
            robot.apply_base_command([], command, step_dt=dt)
            world.step(render=False)
        robot.apply_base_command([], np.zeros(3, dtype=np.float32), step_dt=dt)
        world.step(render=False)
        robot.resume_manipulation_base_hold()

        target_arm = np.asarray(robot._articulation_view.get_joint_positions()[0][arm_indices], dtype=np.float32)
        target_arm[0] += 0.10
        gripper = np.asarray(robot._articulation_view.get_joint_positions()[0][gripper_indices], dtype=np.float32)
        hold_state = robot.get_base_joint_state()
        hold_target = np.asarray(hold_state["wheel_positions"], dtype=np.float64).reshape(-1)
        for _ in range(max(int(args.manipulation_steps), 0)):
            action_positions = np.concatenate([target_arm, gripper]).reshape(1, -1)
            robot.apply_action(joint_positions=action_positions, joint_indices=action_indices)
            robot.reapply_manipulation_base_hold()
            world.step(render=False)

        final_state = robot.get_base_joint_state()
        final_positions = np.asarray(final_state["wheel_positions"], dtype=np.float64).reshape(-1)
        drift = np.abs(final_positions - hold_target)
        report.update(
            {
                "status": "completed",
                "dof_names": dof_names,
                "arm_indices": [int(value) for value in arm_indices.tolist()],
                "gripper_indices": [int(value) for value in gripper_indices.tolist()],
                "base_indices": [int(value) for value in base_indices.tolist()],
                "action_indices": [int(value) for value in action_indices.tolist()],
                "navigation_steps": int(args.navigation_steps),
                "manipulation_steps": int(args.manipulation_steps),
                "hold_target": [float(value) for value in hold_target.tolist()],
                "final_base_positions": [float(value) for value in final_positions.tolist()],
                "base_joint_abs_drift": [float(value) for value in drift.tolist()],
                "max_base_joint_abs_drift": float(drift.max(initial=0.0)),
                "base_yaw_drift_deg": float(math.degrees(drift[2])) if drift.size >= 3 else 0.0,
                "hold_tolerance": float(args.hold_tolerance),
            }
        )
        if drift.size and float(drift.max()) > float(args.hold_tolerance):
            report["status"] = "failed"
            report["failure"] = "base hold drift exceeded tolerance"
            return 1
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        report.update({"status": "error", "error": str(exc)})
        return 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
