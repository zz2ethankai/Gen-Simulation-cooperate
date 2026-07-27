import os
from pathlib import Path

os.environ.setdefault("GRASPGENX_DISABLE_AUTO_SETUP", "1")

import numpy as np
import pytest

from graspgenx.exporters.interndata import (
    R_GRASPGENX_FROM_GRASPNET,
    RobotProfile,
    export_interndata_grasps,
)


GRASPGENX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GRASPGENX_ROOT.parents[3]
ROBOT_CONFIG_DIR = PROJECT_ROOT / "workflows/simbox/core/configs/robots"


@pytest.mark.parametrize(
    "config_name,expected_name,expected_ee,expected_gripper",
    [
        ("panda_omron.yaml", "PandaOmron", "panda_hand", "franka_panda"),
        ("panda_omron_virtual.yaml", "PandaOmronVirtual", "panda_hand", "franka_panda"),
        ("split_aloha.yaml", "SplitAloha", "link6", "piper_hand"),
        ("split_aloha_actual.yaml", "SplitAlohaActual", "link6", "piper_hand"),
        ("lift2.yaml", "Lift2", "link6", "arx_x5"),
        ("genie1.yaml", "Genie1", "arm_l_end_link", "galaxea_g1"),
        ("franka_robotiq85.yaml", "FrankaRobotiq85", "panda_link8", "robotiq_2f_85"),
        ("fr3.yaml", "FR3", "panda_hand", "franka_panda"),
        ("tracer2_franka.yaml", "Tracer2Franka", "panda_hand", "franka_panda"),
    ],
)
def test_project_robot_profile_uses_urdf_and_selects_descriptor(
    config_name: str,
    expected_name: str,
    expected_ee: str,
    expected_gripper: str,
) -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / config_name, project_root=PROJECT_ROOT
    )
    assert profile.name == expected_name
    assert profile.gripper_name == expected_gripper
    assert profile.gripper_selection == "target_class_map"
    assert profile.arms[0].ee_link == expected_ee
    assert all(Path(arm.urdf_path).is_file() for arm in profile.arms)


def test_export_contract_full_tcp_transform_and_runtime_round_trip() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "split_aloha.yaml", project_root=PROJECT_ROOT
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, :3, 3] = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    confidences = np.array([0.2, 0.9, 0.5], dtype=np.float64)
    tool_tcp = np.eye(4, dtype=np.float64)
    tool_tcp[:3, 3] = [0.01, -0.02, 0.13]

    output = export_interndata_grasps(
        poses,
        confidences,
        profile,
        tool_tcp_transform=tool_tcp,
        count=2,
    )

    assert output.shape == (2, 17)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.all(np.diff(output[:, 0]) >= 0.0)
    np.testing.assert_allclose(output[:, 0], [0.19, 0.55], atol=1e-6)
    np.testing.assert_allclose(output[:, 1], profile.gripper_max_width)
    np.testing.assert_allclose(
        output[:, 4:13].reshape(-1, 3, 3),
        np.broadcast_to(R_GRASPGENX_FROM_GRASPNET, (2, 3, 3)),
    )

    selected = poses[[1, 2]]
    expected_tcp = (selected @ tool_tcp)[:, :3, 3]
    np.testing.assert_allclose(output[:, 13:16], expected_tcp, atol=1e-6)

    # Reproduce TemplateRobot.pose_post_process_fn without importing Isaac Sim.
    stored_rot = output[:, 4:13].reshape(-1, 3, 3)
    runtime_rot = stored_rot @ np.asarray(profile.r_ee_graspnet).T
    axis_index = {"x": 0, "y": 1, "z": 2}[profile.ee_axis]
    runtime_center = (
        output[:, 13:16]
        + runtime_rot[:, :, axis_index]
        * (output[:, 3:4] - profile.tcp_offset)
    )
    expected_rot = (
        selected[:, :3, :3]
        @ R_GRASPGENX_FROM_GRASPNET
        @ np.asarray(profile.r_ee_graspnet).T
    )
    expected_center = expected_tcp - expected_rot[:, :, axis_index] * profile.tcp_offset
    np.testing.assert_allclose(runtime_rot, expected_rot, atol=1e-6)
    np.testing.assert_allclose(runtime_center, expected_center, atol=1e-6)


def test_export_rejects_invalid_rotation() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "panda_omron.yaml", project_root=PROJECT_ROOT
    )
    pose = np.eye(4)[None]
    pose[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="non-orthonormal"):
        export_interndata_grasps(
            pose,
            np.array([0.5]),
            profile,
            tool_tcp_transform=np.eye(4),
        )


def test_export_requires_requested_candidate_count() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "panda_omron.yaml", project_root=PROJECT_ROOT
    )
    with pytest.raises(ValueError, match="fewer than requested"):
        export_interndata_grasps(
            np.eye(4)[None],
            np.array([0.5]),
            profile,
            tool_tcp_transform=np.eye(4),
            count=2,
        )


def test_export_rejects_invalid_tool_transform() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "panda_omron.yaml", project_root=PROJECT_ROOT
    )
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="proper rotation"):
        export_interndata_grasps(
            np.eye(4)[None],
            np.array([0.5]),
            profile,
            tool_tcp_transform=invalid,
        )
