from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "agent/codex_skills/place-robot-relative-to-object/scripts/place_robot_relative.py"
)


def test_generated_robot_block_matches_virtual_canonical_profile(tmp_path: Path):
    arena_path = tmp_path / "arena.yaml"
    arena_path.write_text(
        yaml.safe_dump(
            {
                "coordinate_frame": {"up_axis": "+Z", "floor_plane": "XY"},
                "fixtures": [
                    {
                        "name": "floor",
                        "translation": [0.0, 0.0, 0.0],
                        "size": [10.0, 10.0, 0.1],
                    },
                    {
                        "name": "table",
                        "translation": [0.0, 0.0, 0.5],
                        "size": [1.0, 1.0, 1.0],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "placement_test",
                        "arena_file": str(arena_path),
                        "robots": [],
                        "regions": [],
                        "positions": {},
                        "metadata": {
                            "robot_placement": {
                                "reference_image": "visual/reference.png"
                            }
                        },
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "split_aloha",
                "interdata_config_file": "workflows/simbox/core/configs/robots/split_aloha.yaml",
                "usd_asset": "InternDataAssets/robots/split_aloha_mid_360_virtual/robot.usd",
                "coordinate_frame": "isaac_z_up_x_front",
                "base": {
                    "approach_offset_m": 0.6,
                    "footprint_xz_m": [
                        [0.2, 0.2],
                        [0.2, -0.2],
                        [-0.2, -0.2],
                        [-0.2, 0.2],
                    ],
                },
                "manipulator": {"horizontal_reach_from_base_center_m": 1.2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            str(task_path),
            "--target",
            "table",
            "--relation=-x",
            "--facing=+x",
            "--robot-profile",
            str(profile_path),
            "--execute",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))["tasks"][0]
    robot = task["robots"][0]
    assert robot["path"] == (
        "InternDataAssets/robots/split_aloha_mid_360_virtual/robot.usd"
    )
    assert not {
        "translation",
        "initial_pose",
        "spawn_region",
        "placement",
    } & robot.keys()

    region = task["regions"][0]
    assert region["placement_mode"] == "fixed_from_region_pose"
    assert region["world_translation"] == [-1.1, 0.0, 0.0]
    assert region["world_euler"] == [0.0, 0.0, 0.0]
    assert task["positions"]["wp_robot_start"] == {
        "x": -1.1,
        "y": 0.0,
        "yaw": 0.0,
    }
    assert task["metadata"]["robot_placement"]["reference_image"] == (
        "visual/reference.png"
    )
