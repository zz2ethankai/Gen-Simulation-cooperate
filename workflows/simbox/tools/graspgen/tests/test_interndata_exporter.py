from pathlib import Path

import numpy as np
import pytest

from grasp_gen.exporters.interndata import (
    R_GRASPGEN_FROM_GRASPNET,
    RobotProfile,
    export_interndata_grasps,
    load_source_gripper_geometry,
)


GRASPGEN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GRASPGEN_ROOT.parents[3]
ROBOT_CONFIG_DIR = PROJECT_ROOT / "workflows/simbox/core/configs/robots"


@pytest.mark.parametrize(
    "config_name,expected_name,expected_ee",
    [
        ("panda_omron.yaml", "PandaOmron", "panda_hand"),
        ("split_aloha.yaml", "SplitAloha", "link6"),
        ("lift2.yaml", "Lift2", "link6"),
        ("genie1.yaml", "Genie1", "arm_l_end_link"),
        ("franka_robotiq85.yaml", "FrankaRobotiq85", "panda_link8"),
        ("fr3.yaml", "FR3", "panda_hand"),
        ("tracer2_franka.yaml", "Tracer2Franka", "panda_hand"),
    ],
)
def test_project_robot_profile_uses_existing_urdf(
    config_name: str, expected_name: str, expected_ee: str
) -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / config_name, project_root=PROJECT_ROOT
    )
    assert profile.name == expected_name
    assert profile.arms[0].ee_link == expected_ee
    assert all(Path(arm.urdf_path).is_file() for arm in profile.arms)


def test_export_contract_and_runtime_round_trip() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "split_aloha.yaml", project_root=PROJECT_ROOT
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, :3, 3] = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    confidences = np.array([0.2, 0.9, 0.5], dtype=np.float64)
    source_depth = 0.105

    output = export_interndata_grasps(
        poses,
        confidences,
        profile,
        source_gripper_depth=source_depth,
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
        np.broadcast_to(R_GRASPGEN_FROM_GRASPNET, (2, 3, 3)),
    )

    selected = poses[[1, 2]]
    expected_tcp = selected[:, :3, 3] + selected[:, :3, 2] * source_depth
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
        @ R_GRASPGEN_FROM_GRASPNET
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
            source_gripper_depth=0.1,
        )


def test_source_gripper_geometry_resolves_from_vendored_root() -> None:
    geometry = load_source_gripper_geometry("franka_panda")
    assert geometry["depth"] > 0.1
    assert geometry["width"] > 0.1


@pytest.mark.parametrize(
    "config_name,expected_source",
    [
        ("panda_omron.yaml", "franka_panda"),
        ("split_aloha.yaml", "franka_panda"),
        ("lift2.yaml", "franka_panda"),
        ("genie1.yaml", "franka_panda"),
        ("franka_robotiq85.yaml", "robotiq_2f_140"),
    ],
)
def test_robot_profile_recommends_official_checkpoint(
    config_name: str, expected_source: str
) -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / config_name, project_root=PROJECT_ROOT
    )
    assert profile.recommended_source_grippers[0] == expected_source


def test_export_requires_requested_candidate_count() -> None:
    profile = RobotProfile.from_project_config(
        ROBOT_CONFIG_DIR / "panda_omron.yaml", project_root=PROJECT_ROOT
    )
    with pytest.raises(ValueError, match="fewer than requested"):
        export_interndata_grasps(
            np.eye(4)[None],
            np.array([0.5]),
            profile,
            source_gripper_depth=0.1,
            count=2,
        )
