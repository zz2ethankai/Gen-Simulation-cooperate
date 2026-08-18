"""Pure geometry contracts for the Agent physics renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Gf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import visual_physics
from agent.visual_physics import (
    _finish_replicator,
    apply_xform,
    capture_rgb,
    load_items,
    normalized_collider_kind,
    object_physics_mode,
    partition_objects_by_physics,
    robot_pose_from_regions,
    robot_visual_items,
    static_convex_collision_method,
    support_body_box_geometry,
)
from workflows.simbox.core.robots.profile import PlacementFamily


def test_support_body_box_is_clipped_below_the_explicit_support_plane():
    center, size = support_body_box_geometry(
        (0.92, 0.05, 0.0),
        (2.28, 0.61, 0.91),
        support_surface_z=0.882,
        top_clearance=0.018,
    )

    assert center == pytest.approx((1.6, 0.33, 0.432))
    assert size == pytest.approx((1.36, 0.56, 0.864))
    assert center[2] + 0.5 * size[2] == pytest.approx(0.864)


def test_apply_xform_reuses_authored_op_precision_and_resets_order():
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    xform = UsdGeom.Xformable(prim)
    scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(0.5, 0.5, 0.5))

    apply_xform(
        prim,
        translation=(1.0, 2.0, 3.0),
        euler_deg=(10.0, 20.0, 30.0),
        scale=(1.0, 1.0, 1.0),
    )

    ordered_ops = xform.GetOrderedXformOps()
    assert [op.GetName() for op in ordered_ops] == [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
    ]
    assert ordered_ops[2].GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    assert tuple(ordered_ops[2].Get()) == pytest.approx((1.0, 1.0, 1.0))


def test_support_body_box_rejects_a_clearance_that_removes_the_body():
    with pytest.raises(ValueError, match="empty after clipping"):
        support_body_box_geometry(
            (0.0, 0.0, 0.5),
            (1.0, 1.0, 0.9),
            support_surface_z=0.5,
            top_clearance=0.01,
        )


@pytest.mark.parametrize(
    ("declared", "normalized"),
    [
        ("support_body_bbox", "supportbodybbox"),
        ("supportBodyBBox", "supportbodybbox"),
        ("convex_decomposition", "convexdecomposition"),
        ("static_mesh", "staticmesh"),
    ],
)
def test_collider_names_share_one_normalized_dialect(declared, normalized):
    assert normalized_collider_kind(declared, "coacd") == normalized


@pytest.mark.parametrize(
    ("collider_kind", "has_sidecar", "method"),
    [
        ("coacd", True, "coacd"),
        ("coacd", False, "missing_coacd"),
        ("convexdecomposition", True, "coacd"),
        ("convexdecomposition", False, "physx_convex_decomposition"),
    ],
)
def test_static_convex_collision_method_distinguishes_strict_coacd_from_physx(
    collider_kind, has_sidecar, method
):
    assert (
        static_convex_collision_method(
            collider_kind, has_coacd_sidecar=has_sidecar
        )
        == method
    )


@pytest.mark.parametrize(
    ("cfg", "mode"),
    [
        (
            {
                "physics": {"rigid_body": False},
                "static": False,
                "rigidbody": True,
                "target_class": "RigidObject",
            },
            "static",
        ),
        (
            {
                "physics": {"rigid_body": True},
                "static": True,
                "rigidbody": False,
                "target_class": "GeometryObject",
            },
            "dynamic",
        ),
        ({"static": True, "rigidbody": True}, "static"),
        ({"static": False, "rigidbody": False}, "dynamic"),
        ({"rigidbody": False, "target_class": "RigidObject"}, "static"),
        ({"target_class": "GeometryObject"}, "static"),
        ({"target_class": "PlaneObject"}, "static"),
        ({"target_class": "RigidObject"}, "dynamic"),
        ({"target_class": "CustomObject"}, "dynamic"),
    ],
)
def test_object_physics_mode_uses_declared_priority(cfg, mode):
    assert object_physics_mode(cfg) == mode


def test_object_physics_mode_rejects_non_boolean_declarations():
    with pytest.raises(ValueError, match="physics.rigid_body must be a boolean"):
        object_physics_mode({"physics": {"rigid_body": "false"}})


def test_partition_objects_by_physics_preserves_each_group_order():
    objects = [
        {"name": "static_a", "physics": {"rigid_body": False}},
        {"name": "dynamic_a", "target_class": "RigidObject"},
        {"name": "static_b", "target_class": "GeometryObject"},
        {"name": "dynamic_b", "rigidbody": True},
    ]

    static_objects, dynamic_objects = partition_objects_by_physics(objects)

    assert [cfg["name"] for cfg in static_objects] == ["static_a", "static_b"]
    assert [cfg["name"] for cfg in dynamic_objects] == ["dynamic_a", "dynamic_b"]


@pytest.mark.parametrize("target_field", ["B", "target"])
def test_robot_pose_uses_its_region_target_fixture_frame(target_field):
    scene_cfg = SimpleNamespace(
        arena={
            "fixtures": [
                {"name": "sink_counter", "translation": [1.6, 0.33, 0.45]},
                {"name": "storage_counter", "translation": [3.4, 0.53, 0.45]},
            ]
        },
        task={
            "regions": [
                {
                    "object": "franka",
                    target_field: "storage_counter",
                    "random_config": {
                        "pos_range": [
                            [0.175, 0.0125, 0.0],
                            [0.175, 0.0125, 0.0],
                        ],
                        "yaw_rotation": [0.0, 0.0],
                    },
                }
            ]
        },
    )

    pose = robot_pose_from_regions(
        scene_cfg,
        {"name": "franka", "euler": [0.0, 0.0, -180.0]},
    )

    assert pose == pytest.approx((3.575, 0.5425, -180.0))


def test_robot_visual_uses_canonical_asset_and_mount_support(
    monkeypatch,
    tmp_path,
):
    profile = SimpleNamespace(
        target_class="FR3",
        placement=SimpleNamespace(family=PlacementFamily.SUPPORT_MOUNTED),
    )
    asset_path = tmp_path / "robot.usd"
    monkeypatch.setattr(
        visual_physics,
        "load_robot_profile_for_task",
        lambda robot, task_path: profile,
    )
    monkeypatch.setattr(
        visual_physics,
        "resolve_robot_asset_path",
        lambda loaded_profile: asset_path,
    )
    scene_cfg = SimpleNamespace(
        task_path=tmp_path / "task.yaml",
        arena={
            "fixtures": [
                {
                    "name": "storage_counter",
                    "translation": [3.4, 0.53, 0.45],
                    "support_surface_z": 0.841,
                }
            ]
        },
        task={
            "regions": [
                {
                    "object": "franka",
                    "target": "storage_counter",
                    "random_config": {
                        "pos_range": [
                            [0.175, 0.0125, 0.003],
                            [0.175, 0.0125, 0.003],
                        ],
                        "yaw_rotation": [0.0, 0.0],
                    },
                }
            ]
        },
    )

    items, support_heights = robot_visual_items(
        scene_cfg,
        [
            {
                "name": "franka",
                "robot_config_file": "robots/fr3.yaml",
                "euler": [0.0, 0.0, -180.0],
            }
        ],
        {"min_x": 0.0, "max_x": 4.0, "min_y": 0.0, "max_y": 3.0, "floor_z": 0.0},
    )

    assert items[0]["path"] == str(asset_path)
    assert items[0]["target_class"] == "FR3"
    assert items[0]["translation"] == pytest.approx([3.575, 0.5425, 0.0])
    assert support_heights == pytest.approx({"franka": 0.844})


def test_support_mounted_robot_rejects_missing_support_height(
    monkeypatch,
    tmp_path,
):
    profile = SimpleNamespace(
        target_class="FR3",
        placement=SimpleNamespace(family=PlacementFamily.SUPPORT_MOUNTED),
    )
    monkeypatch.setattr(
        visual_physics,
        "load_robot_profile_for_task",
        lambda robot, task_path: profile,
    )
    monkeypatch.setattr(
        visual_physics,
        "resolve_robot_asset_path",
        lambda profile: tmp_path / "robot.usd",
    )
    scene_cfg = SimpleNamespace(
        task_path=tmp_path / "task.yaml",
        arena={"fixtures": [{"name": "storage_counter", "translation": [0.0, 0.0, 0.0]}]},
        task={
            "regions": [
                {
                    "object": "franka",
                    "target": "storage_counter",
                    "random_config": {"pos_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="support_surface_z"):
        robot_visual_items(
            scene_cfg,
            [{"name": "franka", "robot_config_file": "robots/fr3.yaml"}],
            {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0, "floor_z": 0.0},
        )


def test_robot_support_height_is_forwarded_to_reference_loader(
    monkeypatch,
    tmp_path,
):
    observed = {}

    def fake_load_reference(
        stage,
        root_path,
        cfg,
        asset_root,
        *,
        physics,
        support_z,
    ):
        observed["support_z"] = support_z
        return "/World/Robots/franka"

    monkeypatch.setattr(visual_physics, "load_reference", fake_load_reference)

    loaded = load_items(
        object(),
        "/World/Robots",
        [{"name": "franka", "path": str(tmp_path / "robot.usd")}],
        tmp_path,
        support_heights={"franka": 0.841},
    )

    assert loaded == ["/World/Robots/franka"]
    assert observed["support_z"] == pytest.approx(0.841)


@pytest.mark.parametrize("step_fails", [False, True])
def test_capture_rgb_releases_writer_and_render_product(
    monkeypatch,
    tmp_path,
    step_fails,
):
    events = []
    expected = tmp_path / "rgb_0000.png"

    class Orchestrator:
        @staticmethod
        def step(**_kwargs):
            events.append("step")
            if step_fails:
                raise RuntimeError("capture failed")
            expected.write_bytes(b"png")

        @staticmethod
        def wait_until_complete():
            events.append("wait")

    class Writer:
        @staticmethod
        def attach(_render_products):
            events.append("attach")

        @staticmethod
        def detach():
            events.append("detach")

    class RenderProduct:
        @staticmethod
        def destroy():
            events.append("destroy")

    monkeypatch.setattr(visual_physics, "SETTINGS", SimpleNamespace(rt_subframes=1))
    monkeypatch.setattr(visual_physics, "validate_png_not_blank", lambda _path: None)
    rep = SimpleNamespace(orchestrator=Orchestrator())

    if step_fails:
        with pytest.raises(RuntimeError, match="capture failed"):
            capture_rgb(rep, Writer(), RenderProduct(), tmp_path)
        assert events == ["attach", "step", "detach", "destroy"]
    else:
        assert capture_rgb(rep, Writer(), RenderProduct(), tmp_path) == expected
        assert events == ["attach", "step", "wait", "detach", "destroy"]


def test_finish_replicator_stops_drains_and_terminates_io_workers(monkeypatch):
    events = []
    orchestrator = SimpleNamespace(
        stop=lambda: events.append("stop"),
        wait_until_complete=lambda: events.append("wait"),
    )
    data_queue = SimpleNamespace(q=object(), destroy=lambda: events.append("destroy"))
    rep = SimpleNamespace(
        orchestrator=orchestrator,
        backends=SimpleNamespace(
            io_queue=SimpleNamespace(data_queue=data_queue),
        ),
    )
    monkeypatch.setattr(
        visual_physics.atexit,
        "unregister",
        lambda callback: events.append(("unregister", callback)),
    )

    _finish_replicator(rep)

    assert events == ["stop", "wait", ("unregister", data_queue.destroy), "destroy"]


def test_finish_replicator_skips_an_uninitialized_io_queue(monkeypatch):
    events = []
    data_queue = SimpleNamespace(q=None, destroy=lambda: events.append("destroy"))
    rep = SimpleNamespace(
        orchestrator=SimpleNamespace(
            stop=lambda: events.append("stop"),
            wait_until_complete=lambda: events.append("wait"),
        ),
        backends=SimpleNamespace(
            io_queue=SimpleNamespace(data_queue=data_queue),
        ),
    )
    monkeypatch.setattr(
        visual_physics.atexit,
        "unregister",
        lambda _callback: events.append("unregister"),
    )

    _finish_replicator(rep)

    assert events == ["stop", "wait"]


def test_run_render_records_failure_before_fast_shutdown(
    monkeypatch,
    tmp_path,
):
    observed = {}

    class FakeSimulationApp:
        def __init__(self, launch_config, **kwargs):
            observed["launch_config"] = launch_config
            observed["closed"] = False

        def close(self, *, wait_for_replicator):
            observed["closed"] = True
            observed["wait_for_replicator"] = wait_for_replicator

    monkeypatch.setitem(
        sys.modules,
        "isaacsim",
        SimpleNamespace(SimulationApp=FakeSimulationApp),
    )
    monkeypatch.setenv("INTERNDATA_ISAAC_ACTIVE_GPU", "2")

    def fail_render(*_args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(visual_physics, "_render_main", fail_render)
    result = visual_physics.run_render(
        SimpleNamespace(task=tmp_path / "task.yaml", output_dir=tmp_path / "view")
    )

    assert result == 1
    assert observed["launch_config"]["active_gpu"] == 2
    assert observed["launch_config"]["physics_gpu"] == 0
    assert observed["launch_config"]["multi_gpu"] is False
    assert observed["launch_config"]["max_gpu_count"] == 1
    assert observed["launch_config"]["fast_shutdown"] is True
    assert observed["closed"] is True
    assert observed["wait_for_replicator"] is False
    status = json.loads((tmp_path / "view" / "render_status.json").read_text())
    assert status == {
        "return_code": 1,
        "error": {"type": "RuntimeError", "message": "render failed"},
    }
