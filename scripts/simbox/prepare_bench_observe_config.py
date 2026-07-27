#!/usr/bin/env python3
"""Replace a task's Skill chain with two-arm ObserveHold for recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an observation-only task from a compiled robot pose.")
    parser.add_argument("--input", required=True, type=Path, help="Compiled task containing the desired robot pose")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-steps", default=300, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wait_steps <= 0:
        raise ValueError("--wait-steps must be greater than zero")
    cfg = OmegaConf.load(args.input)
    if "tasks" not in cfg or len(cfg.tasks) != 1:
        raise ValueError("observation runner requires exactly one task")
    task = cfg.tasks[0]
    if not task.get("robots"):
        raise ValueError("task must contain a robot")
    robot_name = str(task.robots[0].name)

    for camera in task.get("cameras", []):
        camera_name = str(camera.name)
        camera.record_to = robot_name
        if not camera.get("record_mode"):
            camera.record_mode = "lmdb_and_video"
        if not camera.get("save_name"):
            if camera_name == "navigate_global":
                camera.save_name = "global"
            else:
                camera.save_name = camera_name.removeprefix(f"{robot_name}_")
    task.skills = [
        {
            robot_name: [
                {
                    "base": [],
                    "left": [
                        {
                            "name": "observe_hold",
                            "hold_steps": args.wait_steps,
                        }
                    ],
                    "right": [
                        {
                            "name": "observe_hold",
                            "hold_steps": args.wait_steps,
                        }
                    ],
                }
            ]
        }
    ]
    if not task.get("metadata"):
        task.metadata = {}
    task.metadata.observation_config = {
        "generated_from": str(args.input.resolve()),
        "mode": "observe_hold",
        "hold_steps": args.wait_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output)
    print(f"generated observation config: {args.output}")


if __name__ == "__main__":
    main()
