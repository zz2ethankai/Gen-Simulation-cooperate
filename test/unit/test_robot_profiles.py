"""Offline contracts for canonical robot model profiles."""

from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.robots.profile import (  # noqa: E402
    PlacementFamily,
    RobotProfileError,
    load_robot_profile,
    project_runtime_config,
    resolve_fixed_robot_start_pose,
    resolve_robot_asset_path,
)


ROBOT_CONFIG_DIR = ROOT / "workflows/simbox/core/configs/robots"
SCENE4_SHARED_SPLIT_ALOHA = (
    ROOT
    / "InternDataAssets/Bench_2.1_isaacsim/scene_4/shared_assets/split_aloha_mid_360/robot.usd"
)


def test_profile_module_imports_without_isaac_runtime():
    before = set(sys.modules)
    importlib.import_module("core.robots.profile")
    imported = set(sys.modules) - before
    assert not any(name == "omni" or name.startswith("omni.") for name in imported)


def test_workspace_facade_exports_canonical_profiles_without_isaac_runtime():
    facade = importlib.import_module("core.utils.workspace_planner")
    assert facade.RobotModelProfile.__name__ == "RobotModelProfile"
    assert facade.PlacementFamily.FLOOR_STANDING.value == "floor_standing"


@pytest.mark.parametrize(
    ("filename", "profile_id", "family", "arms"),
    [
        (
            "split_aloha.yaml",
            "split_aloha_floor_standing_v1",
            PlacementFamily.FLOOR_STANDING,
            {"left", "right"},
        ),
        (
            "lift2.yaml",
            "lift2_floor_standing_v1",
            PlacementFamily.FLOOR_STANDING,
            {"left", "right"},
        ),
        (
            "fr3.yaml",
            "fr3_support_mounted_v1",
            PlacementFamily.SUPPORT_MOUNTED,
            {"left"},
        ),
        (
            "franka_robotiq85.yaml",
            "franka_robotiq85_support_mounted_v1",
            PlacementFamily.SUPPORT_MOUNTED,
            {"left"},
        ),
        (
            "genie1.yaml",
            "genie1_floor_standing_v1",
            PlacementFamily.FLOOR_STANDING,
            {"left", "right"},
        ),
    ],
)
def test_canonical_profiles_are_strict_and_hashable(filename, profile_id, family, arms):
    profile = load_robot_profile(ROBOT_CONFIG_DIR / filename)
    assert profile.profile_id == profile_id
    assert profile.placement.family is family
    assert set(profile.arms) == arms
    assert profile.base.operation_mode == "locked"
    assert len(profile.profile_hash) == 64
    assert profile.profile_hash == load_robot_profile(
        ROBOT_CONFIG_DIR / filename
    ).profile_hash
    assert profile.capabilities == {"pick", "place"}
    assert profile.collision_world_modes == {"physics_schema"}


def test_unknown_placement_family_fails_instead_of_defaulting(tmp_path):
    raw = yaml.safe_load((ROBOT_CONFIG_DIR / "fr3.yaml").read_text(encoding="utf-8"))
    raw["placement"]["family"] = "tabletop"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RobotProfileError, match="placement.family"):
        load_robot_profile(path)


def test_unimplemented_mount_contact_exemptions_are_rejected(tmp_path):
    raw = yaml.safe_load((ROBOT_CONFIG_DIR / "fr3.yaml").read_text(encoding="utf-8"))
    raw["placement"]["allowed_mount_contacts"] = ["base_link->table"]
    path = tmp_path / "invalid_mount_contacts.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RobotProfileError, match="placement fields must be exactly"):
        load_robot_profile(path)


def test_asset_variant_contract_rejects_invalid_hash(tmp_path):
    raw = yaml.safe_load((ROBOT_CONFIG_DIR / "split_aloha.yaml").read_text(encoding="utf-8"))
    raw["asset_variants"] = [{"variant_id": "invalid", "sha256": "not-a-sha256"}]
    path = tmp_path / "invalid_asset_variant.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(RobotProfileError, match="64-character hex digest"):
        load_robot_profile(path)


def test_runtime_projection_uses_canonical_arms_and_rejects_schema_overrides():
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "split_aloha.yaml")
    runtime = project_runtime_config(
        profile,
        {
            "name": "robot_0",
            "robot_config_file": str(ROBOT_CONFIG_DIR / "split_aloha.yaml"),
            "euler": [0.0, 0.0, 90.0],
        },
    )
    assert runtime["arms"]["left"]["controller"] == "SplitAloha"
    assert "robot_file" not in runtime
    assert runtime["arms"]["left"]["command_joint_names"] == [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
    ]
    assert runtime["arms"]["left"]["trajectory_joint_names"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert runtime["manipulation_base_hold"]["joint_names"] == [
        "mobile_translate_x",
        "mobile_translate_y",
        "mobile_rotate",
    ]
    with pytest.raises(RobotProfileError, match="canonical fields"):
        project_runtime_config(profile, {"ignore_roles": ["fixture"]})


def test_legacy_skill_placement_fields_are_not_projected_into_robot_config(
    tmp_path,
):
    asset_path = tmp_path / "robot.usd"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    raw_profile = yaml.safe_load(
        (ROBOT_CONFIG_DIR / "split_aloha.yaml").read_text(encoding="utf-8")
    )
    raw_profile["path"] = str(asset_path)
    profile_path = tmp_path / "split_aloha.yaml"
    profile_path.write_text(
        yaml.safe_dump(raw_profile, sort_keys=False), encoding="utf-8"
    )
    profile = load_robot_profile(profile_path)
    runtime = project_runtime_config(
        profile,
        {
            "name": "split_aloha",
            "euler": [0.0, 0.0, 0.0],
            "translation": [-1.03, -0.055, 0.0],
            "initial_pose": {
                "rotation": [0.0, 0.0, 0.0],
                "keep_upright": True,
            },
            "spawn_region": "split_aloha_start_region",
            "placement": {
                "defined_by": "regions",
                "spawn_region": "split_aloha_start_region",
            },
        },
    )

    assert runtime["euler"] == [0.0, 0.0, 0.0]
    assert not {"translation", "initial_pose", "spawn_region"} & runtime.keys()
    assert runtime["placement"]["family"] == "floor_standing"
    assert "defined_by" not in runtime["placement"]


def test_fixed_robot_start_pose_is_owned_by_region():
    region = {
        "placement_mode": "fixed_from_region_pose",
        "world_translation": [-1.03, -0.055, 0.0],
        "world_euler": [0.0, 0.0, 0.0],
    }
    translation, euler, quaternion = resolve_fixed_robot_start_pose(
        region,
        {"euler": [0.0, 0.0, 90.0]},
    )

    assert translation == [-1.03, -0.055, 0.0]
    assert euler == [0.0, 0.0, 0.0]
    assert quaternion is None


def test_legacy_fixed_region_center_preserves_generated_world_pose():
    region = {
        "placement_mode": "fixed_from_robot_start_position",
        "center": [-1.03, -0.055],
        "support_surface_z": 0.0,
    }
    translation, euler, quaternion = resolve_fixed_robot_start_pose(
        region,
        {"euler": [0.0, 0.0, 0.0]},
    )

    assert translation == [-1.03, -0.055, 0.0]
    assert euler == [0.0, 0.0, 0.0]
    assert quaternion is None


def test_runtime_projection_validates_instance_assertions(tmp_path):
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "split_aloha.yaml")
    runtime = project_runtime_config(
        profile,
        {
            "target_class": "SplitAloha",
            "path": profile.path,
            "name": "robot_0",
        },
    )
    assert runtime["target_class"] == "SplitAloha"
    assert (ROOT / runtime["path"]).samefile(resolve_robot_asset_path(profile))
    with pytest.raises(RobotProfileError, match="PROFILE_TARGET_CLASS_MISMATCH"):
        project_runtime_config(profile, {"target_class": "FR3"})
    different_asset = tmp_path / "robot.usd"
    different_asset.write_bytes(b"not the canonical asset")
    with pytest.raises(RobotProfileError, match="PROFILE_ASSET_MISMATCH"):
        project_runtime_config(
            profile,
            {"path": str(different_asset)},
            task_path=tmp_path / "task.yaml",
        )


def test_runtime_projection_accepts_primary_hash_at_resolved_instance_path(tmp_path):
    primary_asset = tmp_path / "primary.usd"
    equivalent_asset = tmp_path / "equivalent.usd"
    primary_asset.write_bytes(b"equivalent robot asset")
    equivalent_asset.write_bytes(primary_asset.read_bytes())
    raw = yaml.safe_load((ROBOT_CONFIG_DIR / "split_aloha.yaml").read_text(encoding="utf-8"))
    raw["path"] = str(primary_asset)
    raw["asset_variants"] = []
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    profile = load_robot_profile(profile_path)

    runtime = project_runtime_config(profile, {"path": str(equivalent_asset)})

    assert Path(runtime["path"]).samefile(equivalent_asset)


def test_split_aloha_registered_asset_variant_is_used_at_runtime():
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "split_aloha.yaml")
    expected_hash = hashlib.sha256(SCENE4_SHARED_SPLIT_ALOHA.read_bytes()).hexdigest()
    assert [(variant.variant_id, variant.sha256) for variant in profile.asset_variants] == [
        ("collision_enhanced_v1", expected_hash)
    ]
    task_path = (
        ROOT
        / "InternDataAssets/Bench_2.1_isaacsim/scene_4/01_kitchen/assets/basic"
        / "kitchen_cup_transfer/simbox_task.yaml"
    )
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))["tasks"][0]
    robot = task["robots"][0]

    runtime = project_runtime_config(
        profile,
        robot,
        task_path=task_path,
        asset_root=task["asset_root"],
    )

    assert Path(runtime["path"]).is_absolute()
    assert Path(runtime["path"]).samefile(SCENE4_SHARED_SPLIT_ALOHA)


def test_split_aloha_primary_and_variant_satisfy_usd_profile_contract():
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "split_aloha.yaml")
    required_joint_names = set(profile.base.locked_joint_names)
    required_prim_paths: set[str] = set()
    collision_link_paths: set[str] = set()
    for arm in profile.arms.values():
        required_joint_names.update(arm.command_joint_names)
        required_joint_names.update(arm.gripper.joint_names)
        required_prim_paths.update(
            (arm.base_path, arm.ee_path, *arm.filter_paths, *arm.forbid_collision_paths)
        )
        collision_link_paths.update(arm.forbid_collision_paths)
    required_prim_paths.update(
        camera.parent_suffix for camera in profile.camera_rig if camera.parent_suffix
    )

    for asset_path in (resolve_robot_asset_path(profile), SCENE4_SHARED_SPLIT_ALOHA):
        stage = Usd.Stage.Open(str(asset_path))
        assert stage is not None
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid()
        default_prefix = f"{default_prim.GetPath()}/"
        relative_prims = {
            str(prim.GetPath())[len(default_prefix) :]: prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(default_prefix)
        }
        joint_names = {
            prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        }
        collision_prim_paths = {
            path
            for path, prim in relative_prims.items()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }

        assert required_joint_names <= joint_names
        assert required_prim_paths <= set(relative_prims)
        assert all(
            any(
                collision_path == link_path
                or collision_path.startswith(f"{link_path}/")
                for collision_path in collision_prim_paths
            )
            for link_path in collision_link_paths
        )


def test_runtime_projection_does_not_rebase_canonical_asset_to_scene_root(tmp_path):
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "split_aloha.yaml")
    scene_root = tmp_path / "scene"
    conflicting = scene_root / profile.path
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"different robot asset")

    runtime = project_runtime_config(profile, {"name": "robot_0"}, asset_root=scene_root)
    assert (ROOT / runtime["path"]).samefile(resolve_robot_asset_path(profile))
    with pytest.raises(RobotProfileError, match="PROFILE_ASSET_MISMATCH"):
        project_runtime_config(
            profile,
            {"name": "robot_0", "path": profile.path},
            asset_root=scene_root,
        )


def test_lift2_locks_planar_base_and_lift_joint():
    profile = load_robot_profile(ROBOT_CONFIG_DIR / "lift2.yaml")
    assert profile.base.locked_joint_names == (
        "mobile_translate_x",
        "mobile_translate_y",
        "mobile_rotate",
        "joint4",
    )


def test_profile_hash_changes_with_canonical_content(tmp_path):
    raw = yaml.safe_load(
        (ROBOT_CONFIG_DIR / "split_aloha.yaml").read_text(encoding="utf-8")
    )
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    changed = copy.deepcopy(raw)
    changed["placement"]["base_contact_offset_m"] = 0.001
    second_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert load_robot_profile(first_path).profile_hash != load_robot_profile(
        second_path
    ).profile_hash
