"""Offline sampling, transform, and USD-schema tests for trajectory overlays."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows/simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.visualization.curobo_trajectory import (  # noqa: E402
    CuroboTrajectoryVisualizer,
    create_curobo_trajectory_visualizer,
)
import core.visualization.curobo_trajectory as trajectory_module  # noqa: E402
from core.visualization.trajectory_math import (  # noqa: E402
    distance_sample_indices,
    transform_points,
    uniform_sample_indices,
    valid_sphere_arrays,
)


def test_uniform_sampling_covers_short_long_and_single_point_trajectories():
    assert uniform_sample_indices(0, 64).tolist() == []
    assert uniform_sample_indices(1, 64).tolist() == [0]
    assert uniform_sample_indices(4, 64).tolist() == [0, 1, 2, 3]
    indices = uniform_sample_indices(1000, 64)
    assert len(indices) == 64
    assert indices[0] == 0 and indices[-1] == 999
    assert len(np.unique(indices)) == len(indices)


def test_distance_sampling_keeps_endpoints_and_one_radius_surface_gap():
    points = np.zeros((11, 3), dtype=float)
    points[:, 0] = np.linspace(0.0, 0.2, num=11)
    indices = distance_sample_indices(points, min_spacing_m=0.06, max_count=64)
    assert indices.tolist() == [0, 3, 6, 10]
    distances = np.linalg.norm(np.diff(points[indices], axis=0), axis=1)
    assert np.all(distances >= 0.06 - 1e-9)

    short = distance_sample_indices(points[:2], min_spacing_m=0.06, max_count=64)
    assert short.tolist() == [0, 1]


def test_arm_base_points_are_rotated_and_translated_into_task_root():
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    actual = transform_points(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]]), transform)
    np.testing.assert_allclose(actual, [[1.0, 3.0, 3.0], [-1.0, 2.0, 2.0]])


def test_collision_spheres_drop_non_positive_placeholders_and_keep_real_radii():
    spheres = [
        SimpleNamespace(pose=[1, 2, 3, 1, 0, 0, 0], radius=0.04),
        SimpleNamespace(pose=[4, 5, 6, 1, 0, 0, 0], radius=0.0),
        SimpleNamespace(pose=[7, 8, 9, 1, 0, 0, 0], radius=-1.0),
    ]
    centers, radii = valid_sphere_arrays(spheres)
    np.testing.assert_allclose(centers, [[1, 2, 3]])
    np.testing.assert_allclose(radii, [0.04])


def test_disabled_config_does_not_create_visualizer():
    assert create_curobo_trajectory_visualizer(None, "/World/task_0", {}) is None
    assert (
        create_curobo_trajectory_visualizer(
            None,
            "/World/task_0",
            {"visualization": {"curobo_trajectory": {"enabled": False}}},
        )
        is None
    )


def test_exported_overlay_contains_only_rendering_schemas(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/task_0")
    visualizer = CuroboTrajectoryVisualizer(
        stage,
        "/World/task_0",
        {
            "export_usd": True,
            "ee_color": [1.0, 0.35, 0.0, 1.0],
            "robot_color": [0.1, 0.85, 0.25, 0.28],
        },
    )
    with visualizer._edit():
        visualizer._add_instancer(
            f"{visualizer.root_path}/test/ee_path",
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([0.01, 0.02]),
            "ee",
            visualizer.ee_color,
        )
    output = visualizer.export(tmp_path)
    assert output is not None and output.is_file()
    text = output.read_text(encoding="utf-8")
    assert "PointInstancer" in text and "Sphere" in text and "UsdPreviewSurface" in text
    standalone_stage = Usd.Stage.Open(str(output))
    assert (
        sum(prim.IsA(UsdGeom.PointInstancer) for prim in standalone_stage.Traverse())
        == 1
    )
    forbidden = ("CollisionAPI", "RigidBodyAPI", "MassAPI", "Physx", "ParticleSystem")
    assert not any(token in text for token in forbidden)


class _ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, index):
        return _ArrayTensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_selected_plan_records_custom_data_and_both_marker_types(monkeypatch, tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/task_0")
    visualizer = CuroboTrajectoryVisualizer(stage, "/World/task_0", {})

    class Kinematics:
        @staticmethod
        def get_state(q):
            return SimpleNamespace(ee_position=_ArrayTensor(q.value[:, :3]))

        @staticmethod
        def get_robot_as_spheres(q, filter_valid=True):
            assert filter_valid is True
            return [
                [
                    SimpleNamespace(
                        pose=[row[0], row[1], row[2], 1, 0, 0, 0], radius=0.03
                    )
                ]
                for row in q.value
            ]

    controller = SimpleNamespace(
        name="split_aloha",
        lr_name="right",
        robot_base_path="/World/task_0/robot/base",
        motion_gen=SimpleNamespace(kinematics=Kinematics()),
    )
    monkeypatch.setattr(trajectory_module, "get_prim_at_path", lambda path: path)
    monkeypatch.setattr(
        trajectory_module, "get_relative_transform", lambda source, target: np.eye(4)
    )
    cmd_plan = SimpleNamespace(
        position=_ArrayTensor(np.arange(60, dtype=float).reshape(10, 6) / 100.0)
    )
    visualizer.record_plan(controller, cmd_plan, "close_gripper")

    plan = stage.GetPrimAtPath(
        "/World/task_0/__debug_curobo_trajectory__/split_aloha/right/plan_000"
    )
    assert plan.IsValid()
    custom = plan.GetCustomData()
    assert custom["arm"] == "right"
    assert custom["command"] == "close_gripper"
    assert custom["trajectory_length"] == 10
    assert stage.GetPrimAtPath(f"{plan.GetPath()}/ee_path").IsA(UsdGeom.PointInstancer)
    assert stage.GetPrimAtPath(f"{plan.GetPath()}/robot_spheres").IsA(
        UsdGeom.PointInstancer
    )
    output = visualizer.export(tmp_path)
    assert output is not None and "plan_count = 1" in output.read_text(encoding="utf-8")

    snapshot = visualizer.clone_for_save()
    visualizer.clear()
    snapshot_output = snapshot.export(tmp_path / "snapshot")
    assert snapshot_output is not None
    assert "plan_count = 1" in snapshot_output.read_text(encoding="utf-8")


def test_marker_switches_hide_robot_spheres_and_space_ee_path(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/task_0")
    visualizer = CuroboTrajectoryVisualizer(
        stage,
        "/World/task_0",
        {
            "show_ee_path": True,
            "show_robot_spheres": False,
            "ee_radius_m": 0.02,
            "ee_min_center_spacing_m": 0.06,
        },
    )

    class Kinematics:
        @staticmethod
        def get_state(q):
            return SimpleNamespace(ee_position=_ArrayTensor(q.value[:, :3]))

        @staticmethod
        def get_robot_as_spheres(q, filter_valid=True):
            raise AssertionError("hidden robot spheres must skip CuRobo sphere FK")

    controller = SimpleNamespace(
        name="split_aloha",
        lr_name="right",
        robot_base_path="/World/task_0/robot/base",
        motion_gen=SimpleNamespace(kinematics=Kinematics()),
    )
    monkeypatch.setattr(trajectory_module, "get_prim_at_path", lambda path: path)
    monkeypatch.setattr(
        trajectory_module, "get_relative_transform", lambda source, target: np.eye(4)
    )
    positions = np.zeros((11, 6), dtype=float)
    positions[:, 0] = np.linspace(0.0, 0.2, num=11)
    visualizer.record_plan(
        controller, SimpleNamespace(position=_ArrayTensor(positions)), "reach"
    )

    plan_path = (
        "/World/task_0/__debug_curobo_trajectory__/split_aloha/right/plan_000"
    )
    ee_path = UsdGeom.PointInstancer.Get(stage, f"{plan_path}/ee_path")
    assert ee_path.GetPrim().IsValid()
    assert not stage.GetPrimAtPath(f"{plan_path}/robot_spheres").IsValid()
    ee_positions = np.asarray(ee_path.GetPositionsAttr().Get(), dtype=float)
    ee_scales = np.asarray(ee_path.GetScalesAttr().Get(), dtype=float)
    np.testing.assert_allclose(ee_positions[:, 0], [0.0, 0.06, 0.12, 0.2])
    np.testing.assert_allclose(ee_scales, 0.02)


def test_marker_switches_can_hide_ee_path_and_keep_robot_spheres(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/task_0")
    visualizer = CuroboTrajectoryVisualizer(
        stage,
        "/World/task_0",
        {"show_ee_path": False, "show_robot_spheres": True},
    )

    class Kinematics:
        @staticmethod
        def get_state(q):
            raise AssertionError("hidden EE path must skip EE FK")

        @staticmethod
        def get_robot_as_spheres(q, filter_valid=True):
            return [
                [SimpleNamespace(pose=[0, 0, 0, 1, 0, 0, 0], radius=0.03)]
                for _ in q.value
            ]

    controller = SimpleNamespace(
        name="split_aloha",
        lr_name="left",
        robot_base_path="/World/task_0/robot/base",
        motion_gen=SimpleNamespace(kinematics=Kinematics()),
    )
    monkeypatch.setattr(trajectory_module, "get_prim_at_path", lambda path: path)
    monkeypatch.setattr(
        trajectory_module, "get_relative_transform", lambda source, target: np.eye(4)
    )
    visualizer.record_plan(
        controller,
        SimpleNamespace(position=_ArrayTensor(np.zeros((3, 6), dtype=float))),
        "reach",
    )

    plan_path = "/World/task_0/__debug_curobo_trajectory__/split_aloha/left/plan_000"
    assert not stage.GetPrimAtPath(f"{plan_path}/ee_path").IsValid()
    assert stage.GetPrimAtPath(f"{plan_path}/robot_spheres").IsA(
        UsdGeom.PointInstancer
    )
