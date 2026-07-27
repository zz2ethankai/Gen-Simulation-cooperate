"""Pure offline tests for target-annulus workspace planning."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.workspace.geometry import (  # noqa: E402
    colliding_fixture_layer,
    inside_floor,
    rectangles_overlap,
    sample_target_annulus,
    yaw_to_align_arm_base_deg,
)
from core.workspace.models import (  # noqa: E402
    DEFAULT_ROBOT_PROFILES,
    GeometryCandidate,
    RobotCollisionLayer,
    SamplingConfig,
    WorkspacePlanningError,
)
from core.workspace.planner import (  # noqa: E402
    _fixtures_with_asset_extents,
    apply_candidate_to_document,
    build_manifest,
    load_yaml,
)
from core.workspace.task_compiler import compile_pick_task, compile_probe_task  # noqa: E402


PHONE_TASK = (
    ROOT
    / "InternDataAssets/Bench_2.1_isaacsim/scene_4/04_bedroom/assets/basic/bedroom_phone_placement/simbox_task.yaml"
)
KITCHEN_CUP_TASK = (
    ROOT
    / "InternDataAssets/Bench_2.1_isaacsim/scene_4/01_kitchen/assets/basic/kitchen_cup_transfer/simbox_task.yaml"
)


def test_annulus_sampling_is_deterministic_and_has_stable_ids():
    config = SamplingConfig()
    first = sample_target_annulus([3.0, 2.0], config)
    second = sample_target_annulus([3.0, 2.0], config)
    assert len(first) == 96
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert [item.candidate_id for item in first] == [f"annulus_{index:03d}" for index in range(96)]


def test_annulus_sampling_is_uniform_in_area_and_inside_bounds():
    config = SamplingConfig()
    candidates = sample_target_annulus([0.0, 0.0], config)
    squared_radii = [item.radius_m**2 for item in candidates]
    steps = [right - left for left, right in zip(squared_radii, squared_radii[1:])]
    assert all(config.min_radius_m < item.radius_m < config.max_radius_m for item in candidates)
    assert max(steps) - min(steps) < 1e-12


def test_polar_grid_samples_every_radius_for_every_approach_direction():
    config = SamplingConfig(
        planner="target_annulus_v2",
        min_radius_m=0.45,
        max_radius_m=1.20,
        preferred_radius_m=0.65,
        sequence="polar_grid",
        radial_count=16,
        angular_count=72,
        candidate_count=1152,
    )
    candidates = sample_target_annulus([0.0, 0.0], config)
    assert len(candidates) == 1152
    angle_90 = [value for value in candidates if value.angle_deg == pytest.approx(90.0)]
    assert len(angle_90) == 16
    assert [value.radius_m for value in angle_90] == pytest.approx(
        [0.45 + index * 0.05 for index in range(16)]
    )
    assert angle_90[0].candidate_id == "annulus_v2_a018_r000_y00"
    assert angle_90[-1].candidate_id == "annulus_v2_a018_r015_y00"


def test_polar_grid_yaw_variants_do_not_remove_xy_candidates():
    config = SamplingConfig(
        planner="target_annulus_v2",
        min_radius_m=0.45,
        max_radius_m=1.20,
        preferred_radius_m=0.65,
        sequence="polar_grid",
        radial_count=2,
        angular_count=4,
        yaw_offsets_deg=(-20.0, 0.0, 20.0),
        candidate_count=24,
    )
    candidates = sample_target_annulus([0.0, 0.0], config)
    first_xy = [value for value in candidates if value.world_xy == candidates[0].world_xy]
    assert [value.yaw_offset_deg for value in first_xy] == [-20.0, 0.0, 20.0]
    assert [value.yaw_deg for value in first_xy] == pytest.approx(
        [candidates[0].yaw_deg, candidates[0].yaw_deg + 20.0, candidates[0].yaw_deg + 40.0]
    )


def test_split_aloha_floor_envelope_matches_enabled_wheel_colliders_with_margin():
    profile = DEFAULT_ROBOT_PROFILES["split_aloha_tabletop_v1"]
    assert profile.footprint_m == pytest.approx((0.70, 0.47))
    assert profile.footprint_m[0] > 0.674
    assert profile.footprint_m[1] > 0.4442


def test_every_candidate_faces_the_target():
    target = [1.2, -0.4]
    for candidate in sample_target_annulus(target, SamplingConfig()):
        yaw = math.radians(candidate.yaw_deg)
        forward = [math.cos(yaw), math.sin(yaw)]
        delta = [target[0] - candidate.world_xy[0], target[1] - candidate.world_xy[1]]
        cross = forward[0] * delta[1] - forward[1] * delta[0]
        assert cross == pytest.approx(0.0, abs=1e-9)
        assert forward[0] * delta[0] + forward[1] * delta[1] > 0.0


def test_required_arm_yaw_places_target_on_that_arm_base_forward_ray():
    base = (0.0, 0.0)
    target = (0.0, 0.85)
    arm_base = (0.36848, -0.306)
    yaw = math.radians(yaw_to_align_arm_base_deg(base, target, arm_base))
    dx, dy = target[0] - base[0], target[1] - base[1]
    target_local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    assert target_local_y == pytest.approx(arm_base[1])


def test_rotated_footprints_use_obb_collision():
    assert rectangles_overlap((0, 0), (1.0, 0.4), 90, (0, 0.3), (0.3, 0.3), 0)
    assert not rectangles_overlap((0, 0), (1.0, 0.4), 90, (2, 0), (0.3, 0.3), 0)
    assert not rectangles_overlap((0, 0), (1, 1), 0, (1, 0), (1, 1), 0)


def test_floor_gate_checks_all_rotated_footprint_corners():
    floor = {"translation": [0.0, 0.0, 0.0], "size": [2.0, 2.0]}
    inside = GeometryCandidate("inside", (0.0, 0.0), 45.0, 0.5, 0.0)
    outside = GeometryCandidate("outside", (0.9, 0.9), 45.0, 0.5, 0.0)
    assert inside_floor(inside, (0.7, 0.4), floor)
    assert not inside_floor(outside, (0.7, 0.4), floor)


def test_phone_manifest_contains_only_annulus_candidates():
    document = load_yaml(PHONE_TASK)
    manifest = build_manifest(document, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    assert manifest.version == 3
    assert manifest.status == "geometry_ready"
    assert len(manifest.geometry_candidates) == 96
    assert any(item["geometry_feasible"] for item in manifest.geometry_candidates)
    assert all(item["candidate_id"].startswith("annulus_") for item in manifest.geometry_candidates)
    assert not any("waypoint" in str(item).lower() for item in manifest.geometry_candidates)


def test_asset_metadata_fixture_extent_rejects_fridge_overlap():
    document = load_yaml(KITCHEN_CUP_TASK)
    manifest = build_manifest(document, KITCHEN_CUP_TASK, target_name="white_mug_a_0_id9000")
    candidate = next(
        item for item in manifest.geometry_candidates if item["candidate_id"] == "annulus_009"
    )
    assert candidate["geometry_feasible"] is False
    fridge_side = next(
        fixture
        for fixture in manifest.fixture_audit
        if fixture["name"] == "fridge_side_counter_0_id6"
    )
    assert fridge_side["size_xy"] == pytest.approx([0.36179806, 0.369141695])
    assert fridge_side["size_xyz"][2] == pytest.approx(0.75)
    assert fridge_side["size_source"].startswith("source_metadata:")


def test_home_pose_layer_only_rejects_height_overlapping_fixture():
    candidate = GeometryCandidate("candidate", (0.0, 0.0), 0.0, 0.5, 0.0)
    layer = RobotCollisionLayer("arms", (0.0, 0.0), (1.0, 1.0), 1.5, 2.0)
    low_fixture = {
        "name": "low_counter",
        "translation": [0.0, 0.0, 0.4],
        "size": [0.5, 0.5],
        "size_xyz": [0.5, 0.5, 0.8],
        "collision_enabled": True,
    }
    tall_fixture = {
        **low_fixture,
        "name": "tall_cabinet",
        "translation": [0.0, 0.0, 1.0],
        "size_xyz": [0.5, 0.5, 2.0],
    }
    assert colliding_fixture_layer(candidate, layer, [low_fixture]) is None
    assert colliding_fixture_layer(candidate, layer, [tall_fixture]) == "tall_cabinet"


def _geometry_fixture(**overrides):
    fixture = {
        "name": "fixture_0",
        "target_class": "GeometryObject",
        "path": "asset/Aligned_obj.usda",
        "translation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "collision_enabled": True,
    }
    fixture.update(overrides)
    return fixture


def _write_extent_metadata(path: Path, extent):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"geometry_alignment": {"usd_size_xyz_m": '
        + str(list(extent)).replace("'", '"')
        + "}}\n",
        encoding="utf-8",
    )


def test_fixture_extent_prefers_inline_size_over_all_asset_sources(tmp_path):
    _write_extent_metadata(tmp_path / "explicit.json", [9.0, 9.0, 9.0])
    fixture = _geometry_fixture(size=[1.25, 0.75], source_metadata="explicit.json")
    actual = _fixtures_with_asset_extents({"asset_root": str(tmp_path)}, [fixture])[0]
    assert actual["size"] == [1.25, 0.75]
    assert actual["size_source"] == "arena_inline_size"


def test_fixture_extent_prefers_explicit_source_metadata_over_adjacent_file(tmp_path):
    _write_extent_metadata(tmp_path / "explicit.json", [1.2, 0.8, 0.5])
    _write_extent_metadata(tmp_path / "asset/metadata.json", [8.0, 7.0, 6.0])
    fixture = _geometry_fixture(source_metadata="explicit.json", scale=[2.0, 0.5, 1.0])
    actual = _fixtures_with_asset_extents({"asset_root": str(tmp_path)}, [fixture])[0]
    assert actual["size"] == pytest.approx([2.4, 0.4])
    assert actual["size_source"].startswith("source_metadata:")
    assert actual["size_source"].endswith("#geometry_alignment.usd_size_xyz_m")


def test_fixture_extent_uses_adjacent_metadata_as_compatibility_fallback(tmp_path):
    _write_extent_metadata(tmp_path / "asset/metadata.json", [1.4, 0.6, 0.9])
    actual = _fixtures_with_asset_extents(
        {"asset_root": str(tmp_path)}, [_geometry_fixture()]
    )[0]
    assert actual["size"] == pytest.approx([1.4, 0.6])
    assert actual["size_source"].startswith("adjacent_metadata:")


def test_fixture_extent_uses_actual_usd_bbox_as_final_offline_fallback(tmp_path):
    from pxr import Usd, UsdGeom

    asset_path = tmp_path / "asset/Aligned_obj.usda"
    asset_path.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(asset_path))
    cube = UsdGeom.Cube.Define(stage, "/Asset")
    stage.SetDefaultPrim(cube.GetPrim())
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set((1.6, 0.7, 0.4))
    stage.GetRootLayer().Save()

    actual = _fixtures_with_asset_extents(
        {"asset_root": str(tmp_path)}, [_geometry_fixture()]
    )[0]
    assert actual["size"] == pytest.approx([1.6, 0.7])
    assert actual["size_source"].startswith("usd_bbox:")


def test_fixture_extent_strictly_rejects_unresolvable_geometry_fixture(tmp_path):
    with pytest.raises(WorkspacePlanningError) as error:
        _fixtures_with_asset_extents(
            {"asset_root": str(tmp_path)}, [_geometry_fixture(path="missing.usda")]
        )
    assert error.value.code == "FIXTURE_EXTENT_MISSING"
    assert error.value.details["fixtures"][0]["name"] == "fixture_0"


def test_old_edge_and_waypoint_fields_do_not_change_candidates():
    document = load_yaml(PHONE_TASK)
    baseline = build_manifest(document, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    legacy = copy.deepcopy(document)
    workspace = legacy["tasks"][0].setdefault("manipulation_workspace", {})
    workspace.setdefault("anchor", {}).update(
        {"preferred_edges": ["north_edge"], "preferred_waypoints": ["stale_waypoint"]}
    )
    workspace.setdefault("robot", {})["initial_pose_policy"] = "waypoint_only"
    actual = build_manifest(legacy, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    assert actual.geometry_candidates == baseline.geometry_candidates


def test_candidate_runtime_region_is_floor_relative():
    document = load_yaml(PHONE_TASK)
    manifest = build_manifest(document, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    candidate = next(item for item in manifest.geometry_candidates if item["geometry_feasible"])
    compiled = apply_candidate_to_document(document, PHONE_TASK, candidate)
    task = compiled["tasks"][0]
    region = next(item for item in task["regions"] if item["object"] == "split_aloha")
    shift = region["random_config"]["pos_range"][0]
    assert shift[:2] == pytest.approx([candidate["world_xy"][0] - 2.25, candidate["world_xy"][1] - 1.8])
    assert task["robots"][0]["euler"][2] == pytest.approx(candidate["yaw_deg"])
    assert region["candidate_id"] == candidate["candidate_id"]


def test_compiled_pick_uses_minimal_official_skill_fields(tmp_path):
    document = load_yaml(PHONE_TASK)
    manifest = build_manifest(document, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    candidate = next(item for item in manifest.geometry_candidates if item["geometry_feasible"])
    compiled = compile_pick_task(
        PHONE_TASK,
        candidate,
        "bedroom_phone_0_id9003",
        "right",
        tmp_path / "pick_task.yaml",
    )
    stage = compiled["tasks"][0]["skills"][0]["split_aloha"][0]
    assert stage["left"] == []
    assert stage["right"] == [
        {
            "name": "pick",
            "objects": ["bedroom_phone_0_id9003"],
            "filter_y_dir": ["forward", 90],
            "filter_z_dir": ["downward", 140],
        }
    ]


def test_required_arm_probe_uses_execution_planning_contract_without_cameras(tmp_path):
    document = load_yaml(PHONE_TASK)
    manifest = build_manifest(
        document, PHONE_TASK, target_name="bedroom_phone_0_id9003"
    )
    candidate = next(
        item for item in manifest.geometry_candidates if item["geometry_feasible"]
    )
    planning = {
        "collision_world": {
            "mode": "physics_schema",
            "strict": True,
            "exact_exclusions": [],
        }
    }
    compiled = compile_probe_task(
        PHONE_TASK,
        candidate,
        "bedroom_phone_0_id9003",
        tmp_path / "probe_task.yaml",
        tmp_path / "results",
        arm="left",
        planning=planning,
        attach_prim_path_children=["Aligned/Normalize/Source/base_link/collisions"],
    )
    task = compiled["tasks"][0]
    stage = task["skills"][0]["split_aloha"][0]
    assert stage["right"] == []
    assert [skill["name"] for skill in stage["left"]] == ["pick_plan_probe"]
    assert task["planning"] == planning
    assert task["cameras"] == []
    assert task["render"] is False
    target = next(
        value
        for value in task["objects"]
        if value["name"] == "bedroom_phone_0_id9003"
    )
    assert target["attach_prim_path_children"] == [
        "Aligned/Normalize/Source/base_link/collisions"
    ]
    metadata = task["metadata"]["workspace_probe"]
    assert metadata["required_arm"] == "left"
    assert metadata["collision_world_mode"] == "physics_schema"


def test_movable_floor_target_can_generate_initial_pose_candidates():
    document = load_yaml(PHONE_TASK)
    task = document["tasks"][0]
    region = next(item for item in task["regions"] if item["object"] == "bedroom_phone_0_id9003")
    region["target"] = "floor"
    manifest = build_manifest(document, PHONE_TASK, target_name="bedroom_phone_0_id9003")
    assert manifest.target["support"] == "floor"
    assert any(item["geometry_feasible"] for item in manifest.geometry_candidates)


def test_target_without_grasp_annotation_is_rejected_offline(tmp_path):
    document = load_yaml(PHONE_TASK)
    target_name = "bedroom_phone_0_id9003"
    target = next(item for item in document["tasks"][0]["objects"] if item["name"] == target_name)
    target_usd = tmp_path / "Aligned_obj.usd"
    target_usd.write_text("#usda 1.0\n", encoding="utf-8")
    target["path"] = str(target_usd)
    with pytest.raises(WorkspacePlanningError) as error:
        build_manifest(document, PHONE_TASK, target_name=target_name)
    assert error.value.code == "TARGET_GRASP_ANNOTATION_MISSING"


def test_offline_workspace_package_has_no_runtime_imports():
    package = ROOT / "workflows/simbox/core/workspace"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("import omni", "import isaacsim", "import curobo", "from curobo"):
        assert forbidden not in source
