from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.visual import render_layout


def _write_yaml(path: Path, value: dict) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_render_layout_resolves_each_region_against_its_target_fixture(
    tmp_path: Path,
) -> None:
    arena_path = _write_yaml(
        tmp_path / "arena.yaml",
        {
            "fixtures": [
                {
                    "name": "floor",
                    "translation": [0.0, 0.0, 0.0],
                    "size": [8.0, 6.0, 0.1],
                },
                {
                    "name": "sink_counter",
                    "translation": [1.6, 0.33, 0.45],
                    "size": [1.36, 0.56, 0.9],
                    "role": "support_surface",
                    "support_surface": True,
                    "support_surface_z": 0.882,
                },
                {
                    "name": "storage_counter",
                    "translation": [3.4, 0.53, 0.45],
                    "size": [0.71, 0.96, 0.9],
                    "role": "support_surface",
                    "support_surface": True,
                },
                {
                    "name": "storage_counter__support_plane",
                    "translation": [3.4, 0.53, 0.841],
                    "size": [0.71, 0.96, 0.01],
                    "role": "support_collision_plane",
                    "parent_fixture": "storage_counter",
                },
            ]
        },
    )
    task_path = _write_yaml(
        tmp_path / "task.yaml",
        {
            "tasks": [
                {
                    "name": "region_frame_test",
                    "arena_file": str(arena_path),
                    "robots": [
                        {"name": "franka", "euler": [0.0, 0.0, -180.0]}
                    ],
                    "regions": [
                        {
                            "name": "apple_region",
                            "object": "apple",
                            "B": "storage_counter__support_plane",
                            "random_config": {
                                "pos_range": [
                                    [0.12, 0.42, 0.0],
                                    [0.12, 0.42, 0.0],
                                ]
                            },
                        },
                        {
                            "name": "robot_region",
                            "object": "franka",
                            "target": "storage_counter",
                            "random_config": {
                                "pos_range": [
                                    [0.175, 0.0125, 0.0],
                                    [0.175, 0.0125, 0.0],
                                ],
                                "yaw_rotation": [0.0, 0.0],
                            },
                        },
                    ],
                }
            ]
        },
    )

    output, manifest = render_layout(task_path, tmp_path / "output", dpi=40)

    apple = next(region for region in manifest["regions"] if region["object"] == "apple")
    assert output.is_file()
    assert apple["world_xy"] == pytest.approx([3.52, 0.95])
    assert manifest["robot"]["spawn_xy"] == pytest.approx([3.575, 0.5425])
