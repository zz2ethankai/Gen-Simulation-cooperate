"""Offline geometry, USD-schema, and lifecycle tests for Skill target overlays."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Usd, UsdGeom, UsdShade


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows/simbox"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.visualization.skill_target_math import (  # noqa: E402
    dashed_line_curves,
    gripper_line_curves,
    plane_from_region_points,
    pose_matrix,
    ratio_box_corners,
)
from core.visualization.skill_targets import (  # noqa: E402
    SkillTargetVisualizer,
    create_skill_target_visualizer,
)
import core.visualization.skill_targets as target_module  # noqa: E402
from workflows.base import NimbusWorkFlow  # noqa: E402


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/task_0")
    return stage


def _skill(arm="right"):
    keypoints = {
        "tool_head": [0.0, 0.0, 0.12, 1.0],
        "tool_tail": [0.0, 0.0, 0.06, 1.0],
        "tool_side": [-0.05, 0.0, 0.12, 1.0],
    }
    robot = SimpleNamespace(
        name="split_aloha",
        gripper_max_width=0.1,
        fl_gripper_keypoints=keypoints,
        fr_gripper_keypoints=keypoints,
    )
    controller = SimpleNamespace(
        lr_name=arm,
        robot_base_path=f"/World/task_0/split_aloha/{arm}/arm_base",
    )
    return SimpleNamespace(
        robot=robot,
        controller=controller,
        _target_visualization_context={
            "robot": "split_aloha",
            "arm": arm,
            "skill": "pick",
            "skill_index": 0,
        },
    )


def _patch_identity_transforms(monkeypatch):
    monkeypatch.setattr(target_module, "get_prim_at_path", lambda path: path)
    monkeypatch.setattr(
        target_module,
        "get_relative_transform",
        lambda source, target: np.eye(4),
    )


def test_gripper_outline_and_dashed_approach_follow_pose_and_robot_dimensions():
    transform = pose_matrix([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    curves = gripper_line_curves(
        transform,
        [0.0, 0.0, 0.12],
        [0.0, 0.0, 0.06],
        [-0.05, 0.0, 0.12],
        0.1,
    )
    assert len(curves) == 4
    points = np.concatenate(curves)
    np.testing.assert_allclose(points[:, 0].min(), 0.95)
    np.testing.assert_allclose(points[:, 0].max(), 1.05)
    np.testing.assert_allclose(points[:, 2].min(), 3.06)
    np.testing.assert_allclose(points[:, 2].max(), 3.12)
    dashes = dashed_line_curves([0, 0, 0], [0, 0, 0.1])
    assert len(dashes) >= 3
    np.testing.assert_allclose(dashes[0][0], [0, 0, 0])
    np.testing.assert_allclose(dashes[-1][-1], [0, 0, 0.1])


def test_vertical_and_horizontal_regions_keep_true_extents_and_pad_only_display():
    ratio_points = ratio_box_corners(
        [0, 0, 0],
        [1, 2, 3],
        ((0.4, 0.6), (0.25, 0.75), (0.5, 0.5)),
    )
    vertical = plane_from_region_points(
        ratio_points,
        [0, 0, 1],
        min_display_extent_m=0.08,
        normal_offset_m=0.003,
        tangent_hint=[1, 0, 0],
    )
    np.testing.assert_allclose(vertical["true_extents"], [0.2, 1.0])
    assert vertical["display_padded"] is False
    np.testing.assert_allclose(vertical["corners"][:, 2], 1.503)

    collapsed = np.repeat([[0.5, 0.0, 1.0]], 8, axis=0)
    horizontal = plane_from_region_points(
        collapsed,
        [1, 0, 0],
        min_display_extent_m=0.08,
        tangent_hint=[0, 0, 1],
    )
    assert horizontal["display_padded"] is True
    np.testing.assert_allclose(horizontal["true_extents"], [0.0, 0.0])
    np.testing.assert_allclose(horizontal["display_extents"], [0.08, 0.08])
    np.testing.assert_allclose(horizontal["corners"][:, 0], 0.5)


def test_disabled_config_does_not_create_skill_target_visualizer():
    assert create_skill_target_visualizer(None, "/World/task_0", {}) is None
    assert (
        create_skill_target_visualizer(
            None,
            "/World/task_0",
            {"visualization": {"skill_targets": {"enabled": False}}},
        )
        is None
    )


def test_workflow_async_copy_snapshots_skill_target_layer():
    snapshot = object()

    class Visualizer:
        def clone_for_save(self):
            return snapshot

    class Workflow:
        __copy__ = NimbusWorkFlow.__copy__

    workflow = Workflow()
    workflow.skill_target_visualizer = Visualizer()
    copied = copy.copy(workflow)
    assert copied.skill_target_visualizer is snapshot


def test_pick_target_records_selected_and_pregrasp_curves_then_fades(monkeypatch):
    _patch_identity_transforms(monkeypatch)
    stage = _stage()
    visualizer = SkillTargetVisualizer(stage, "/World/task_0", {})
    skill = _skill()
    handle = visualizer.record_target(
        skill,
        {
            "kind": "pick",
            "objects": ["white_mug"],
            "selected_index": 7,
            "selected_score": 0.12,
            "constraints": {"filter_z_dir": ["downward", 140]},
            "pregrasp_position": [0.4, 0.1, 0.9],
            "pregrasp_orientation": [1, 0, 0, 0],
            "grasp_position": [0.4, 0.1, 0.8],
            "grasp_orientation": [1, 0, 0, 0],
        },
    )
    assert handle == "skill_000_pick"
    root = "/World/task_0/__debug_skill_targets__/Skills/skill_000_pick"
    prim = stage.GetPrimAtPath(root)
    assert prim.GetCustomDataByKey("selected_index") == 7
    assert prim.GetCustomDataByKey("status") == "active"
    assert '"filter_z_dir"' in prim.GetCustomDataByKey("constraints_json")
    for child in ("grasp", "pregrasp", "approach"):
        assert stage.GetPrimAtPath(f"{root}/{child}").IsA(UsdGeom.BasisCurves)

    visualizer.finish_target(handle, True)
    assert prim.GetCustomDataByKey("status") == "completed"
    targets = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath(f"{root}/grasp")
    ).GetDirectBindingRel().GetTargets()
    assert targets and str(targets[0]).endswith("pick_completed")


def test_infeasible_pick_records_metadata_without_fake_gripper(monkeypatch):
    _patch_identity_transforms(monkeypatch)
    stage = _stage()
    visualizer = SkillTargetVisualizer(stage, "/World/task_0", {})
    handle = visualizer.record_target(
        _skill(),
        {
            "kind": "pick",
            "objects": ["white_mug"],
            "has_target": False,
            "failure_reason": "no_feasible_grasp",
            "candidate_count": 20,
            "constraints": {"filter_y_dir": ["forward", 90]},
        },
    )
    root = f"/World/task_0/__debug_skill_targets__/Skills/{handle}"
    prim = stage.GetPrimAtPath(root)
    assert prim.GetCustomDataByKey("status") == "failed"
    assert prim.GetCustomDataByKey("candidate_count") == 20
    assert '"filter_y_dir"' in prim.GetCustomDataByKey("constraints_json")
    assert not list(prim.GetChildren())


def test_place_target_uses_zero_thickness_plane_selected_pose_and_failed_material(
    monkeypatch, tmp_path
):
    _patch_identity_transforms(monkeypatch)
    stage = _stage()
    visualizer = SkillTargetVisualizer(stage, "/World/task_0", {})
    skill = _skill()
    skill._target_visualization_context.update({"skill": "place", "skill_index": 1})
    handle = visualizer.record_target(
        skill,
        {
            "kind": "place",
            "objects": ["white_mug", "sink"],
            "selected_index": 4,
            "place_direction": "vertical",
            "position_constraint": "object",
            "constraints": {"align_place_obj_axis": [0, 1, 0]},
            "bbox_world": [0, 0, 0, 1, 1, 1],
            "ratio_ranges": [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]],
            "region_points_world": [
                [0.4, 0.4, 1.1],
                [0.6, 0.4, 1.1],
                [0.6, 0.6, 1.1],
                [0.4, 0.6, 1.1],
            ],
            "region_normal_world": [0, 0, 1],
            "region_tangent_hint_world": [1, 0, 0],
            "selected_reference_world": [0.52, 0.48, 1.1],
            "preplace_position": [0.52, 0.48, 1.3],
            "preplace_orientation": [1, 0, 0, 0],
            "place_position": [0.52, 0.48, 1.1],
            "place_orientation": [1, 0, 0, 0],
        },
    )
    root = "/World/task_0/__debug_skill_targets__/Skills/skill_000_place"
    plane = UsdGeom.Mesh.Get(stage, f"{root}/target_region")
    assert '"align_place_obj_axis"' in stage.GetPrimAtPath(root).GetCustomDataByKey(
        "constraints_json"
    )
    assert plane.GetPrim().IsValid()
    assert plane.GetDoubleSidedAttr().Get() is True
    points = np.asarray(plane.GetPointsAttr().Get(), dtype=float)
    assert len(np.unique(points[:, 2])) == 1
    assert stage.GetPrimAtPath(f"{root}/selected_reference").IsA(
        UsdGeom.BasisCurves
    )
    assert stage.GetPrimAtPath(f"{root}/place_gripper").IsA(UsdGeom.BasisCurves)
    assert stage.GetPrimAtPath(f"{root}/preplace_gripper").IsA(
        UsdGeom.BasisCurves
    )

    visualizer.finish_target(handle, False, reason="place_failed")
    prim = stage.GetPrimAtPath(root)
    assert prim.GetCustomDataByKey("status") == "failed"
    assert prim.GetCustomDataByKey("failure_reason") == "place_failed"
    targets = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath(f"{root}/target_region")
    ).GetDirectBindingRel().GetTargets()
    assert targets and str(targets[0]).endswith("failed")
    failed_shader = UsdShade.Shader.Get(
        stage, "/World/task_0/__debug_skill_targets__/Materials/failed/PreviewSurface"
    )
    assert np.isclose(failed_shader.GetInput("opacity").Get(), 0.9)

    output = visualizer.export(tmp_path)
    assert output is not None and output.is_file()
    text = output.read_text(encoding="utf-8")
    assert "BasisCurves" in text and "Mesh" in text and "doubleSided = 1" in text
    forbidden = ("CollisionAPI", "RigidBodyAPI", "MassAPI", "Physx", "ParticleSystem")
    assert not any(token in text for token in forbidden)
    standalone = Usd.Stage.Open(str(output))
    assert sum(prim.IsA(UsdGeom.Mesh) for prim in standalone.Traverse()) == 1

    snapshot = visualizer.clone_for_save()
    visualizer.clear()
    assert not stage.GetPrimAtPath(root).IsValid()
    snapshot_output = snapshot.export(tmp_path / "snapshot")
    assert snapshot_output is not None
    assert "target_count = 1" in snapshot_output.read_text(encoding="utf-8")
