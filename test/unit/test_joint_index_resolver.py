from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.joint_index_resolver import (  # noqa: E402
    JointIndexResolutionError,
    resolve_configured_joint_groups,
    resolve_joint_names,
)
from core.robots.profile import load_robot_profile, project_runtime_config  # noqa: E402
import core.robots.profile as profile_module  # noqa: E402


BENCH21_SPLIT_ALOHA_DOF_NAMES = [
    "lifting_joint",
    "fl_joint1",
    "fr_joint1",
    "fl_steering_joint",
    "fr_steering_joint",
    "rl_steering_joint",
    "rr_steering_joint",
    "mobile_rotate",
    "fl_joint2",
    "fr_joint2",
    "fl_wheel",
    "fr_wheel",
    "rl_wheel",
    "rr_wheel",
    "mobile_translate_y",
    "fl_joint3",
    "fr_joint3",
    "mobile_translate_x",
    "fl_joint4",
    "fr_joint4",
    "fl_joint5",
    "fr_joint5",
    "fl_joint6",
    "fr_joint6",
    "fl_joint7",
    "fl_joint8",
    "fr_joint7",
    "fr_joint8",
]


def _project_without_local_asset(profile, monkeypatch):
    monkeypatch.setattr(
        profile_module,
        "resolve_robot_asset_path",
        lambda value: REPO_ROOT / value.path,
    )
    return project_runtime_config(profile)


def _split_aloha_config(monkeypatch):
    path = REPO_ROOT / "workflows" / "simbox" / "core" / "configs" / "robots" / "split_aloha.yaml"
    return _project_without_local_asset(load_robot_profile(path), monkeypatch)


def test_split_aloha_actual_named_groups_match_bench21_runtime_asset(monkeypatch):
    path = (
        REPO_ROOT
        / "workflows"
        / "simbox"
        / "core"
        / "configs"
        / "robots"
        / "split_aloha_actual.yaml"
    )
    config = _project_without_local_asset(load_robot_profile(path), monkeypatch)
    groups = resolve_configured_joint_groups(BENCH21_SPLIT_ALOHA_DOF_NAMES, config)

    assert groups["left_joint"] == [1, 8, 15, 18, 20, 22]
    assert groups["right_joint"] == [2, 9, 16, 19, 21, 23]
    assert groups["left_gripper"] == [24]
    assert groups["right_gripper"] == [26]


def test_split_aloha_compatibility_indices_match_active_virtual_asset(monkeypatch):
    config = _split_aloha_config(monkeypatch)

    assert config["left_joint_indices"] == [4, 6, 8, 10, 12, 14]
    assert config["right_joint_indices"] == [5, 7, 9, 11, 13, 15]
    assert config["left_gripper_indices"] == [16]
    assert config["right_gripper_indices"] == [18]


def test_split_aloha_virtual_named_groups_resolve_authoritative_names(monkeypatch):
    config = _split_aloha_config(monkeypatch)
    dof_names = [f"unused_{index}" for index in range(19)]
    for key in (
        "lift",
        "left_joint",
        "right_joint",
        "left_gripper",
        "right_gripper",
    ):
        for index, name in zip(config[f"{key}_indices"], config[f"{key}_names"]):
            dof_names[index] = name

    groups = resolve_configured_joint_groups(dof_names, config)

    assert groups["left_joint"] == [4, 6, 8, 10, 12, 14]
    assert groups["right_joint"] == [5, 7, 9, 11, 13, 15]
    assert groups["left_gripper"] == [16]
    assert groups["right_gripper"] == [18]


def test_named_resolution_tracks_reordered_asset_instead_of_legacy_indices():
    dof_names = ["unrelated", "arm_2", "gripper", "arm_1"]
    config = {
        "left_joint_names": ["arm_1", "arm_2"],
        "left_joint_indices": [0, 1],
        "left_gripper_names": ["gripper"],
        "left_gripper_indices": [2],
    }

    groups = resolve_configured_joint_groups(dof_names, config)

    assert groups["left_joint"] == [3, 1]
    assert groups["left_gripper"] == [2]


def test_missing_runtime_joint_fails_before_execution():
    with pytest.raises(JointIndexResolutionError, match="missing"):
        resolve_joint_names(["joint1"], ["joint1", "joint2"], group="right_arm")


def test_duplicate_runtime_joint_name_is_rejected_as_ambiguous():
    with pytest.raises(JointIndexResolutionError, match="ambiguous"):
        resolve_joint_names(["joint1", "joint1"], ["joint1"], group="right_arm")


def test_cross_group_overlap_is_rejected():
    config = {
        "left_joint_names": ["joint1"],
        "left_gripper_indices": [0],
    }
    with pytest.raises(JointIndexResolutionError, match="assigned to both"):
        resolve_configured_joint_groups(["joint1"], config)


def test_all_builtin_robot_configs_declare_authoritative_arm_names():
    config_dir = REPO_ROOT / "workflows" / "simbox" / "core" / "configs" / "robots"
    for path in sorted(config_dir.glob("*.yaml")):
        profile = load_robot_profile(path)
        for arm_id, arm in profile.arms.items():
            assert arm.command_joint_names, (
                f"{path.name} arm {arm_id} is missing command_joint_names"
            )
            assert arm.trajectory_joint_names, (
                f"{path.name} arm {arm_id} is missing trajectory_joint_names"
            )
