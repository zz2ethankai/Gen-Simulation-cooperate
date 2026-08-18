"""Offline Physics-schema discovery and object-state tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.collision_scene_manager import (  # noqa: E402
    CollisionObjectState,
    CollisionSceneError,
    CollisionSceneManager,
    validate_exact_exclusions,
)


def _cube(stage, path, *, enabled=True):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(enabled)
    return prim


def _task(stage):
    stage.DefinePrim("/World/task_0/sink_table_named_asset", "Xform")
    _cube(stage, "/World/task_0/sink_table_named_asset/collider")
    UsdGeom.Mesh.Define(stage, "/World/task_0/sink_table_named_asset/visual").CreatePointsAttr([])
    _cube(stage, "/World/task_0/sink_table_named_asset/disabled", enabled=False)

    stage.DefinePrim("/World/task_0/movable_b", "Xform")
    rigid = stage.GetPrimAtPath("/World/task_0/movable_b")
    UsdPhysics.RigidBodyAPI.Apply(rigid).CreateKinematicEnabledAttr(False)
    _cube(stage, "/World/task_0/movable_b/collider")
    return types.SimpleNamespace(
        fixtures={
            "sink_table_named_asset": types.SimpleNamespace(
                prim_path="/World/task_0/sink_table_named_asset"
            )
        },
        objects={"movable_b": types.SimpleNamespace(prim_path="/World/task_0/movable_b")},
        distractors={},
    )


def test_discovery_uses_enabled_collision_api_not_names_or_visual_meshes():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    assert manager.collision_prim_paths == [
        "/World/task_0/sink_table_named_asset/collider",
        "/World/task_0/movable_b/collider",
    ]
    assert manager.records["sink_table_named_asset"].mobility == "static"
    assert manager.records["movable_b"].mobility == "dynamic"


def test_affine_rotation_extraction_ignores_asset_scale():
    manager = CollisionSceneManager.__new__(CollisionSceneManager)
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = np.diag([0.001, 0.002, 0.003]) @ rotation
    extracted = manager._rotation_from_affine(transform)
    assert np.allclose(extracted, rotation, atol=1e-6)


def test_exact_exclusions_require_absolute_path_and_reason():
    with pytest.raises(ValueError, match="complete absolute"):
        validate_exact_exclusions([{"prim_path": "sink/collider", "reason": "known duplicate"}])
    with pytest.raises(ValueError, match="non-empty reason"):
        validate_exact_exclusions([{"prim_path": "/World/sink/collider", "reason": ""}])
    assert validate_exact_exclusions(
        [{"prim_path": "/World/sink/collider", "reason": "duplicate proxy"}]
    ) == {"/World/sink/collider": "duplicate proxy"}


def test_exact_exclusion_must_match_collider_inside_registered_entity():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    _cube(stage, "/World/unrelated/collider")
    with pytest.raises(CollisionSceneError, match="do not name enabled Stage colliders"):
        CollisionSceneManager(
            stage,
            task,
            {
                "strict": True,
                "exact_exclusions": [
                    {"prim_path": "/World/unrelated/collider", "reason": "not in task entities"}
                ],
            },
        )


def test_strict_mode_rejects_collision_api_on_unsupported_prim():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/task_0/object", "Xform")
    UsdPhysics.CollisionAPI.Apply(root)
    task = types.SimpleNamespace(
        fixtures={}, objects={"object": types.SimpleNamespace(prim_path=str(root.GetPath()))}, distractors={}
    )
    with pytest.raises(CollisionSceneError, match="unsupported enabled CollisionAPI"):
        CollisionSceneManager(stage, task, {"strict": True})


def test_non_geometry_collision_api_is_audited_when_supported_descendant_exists():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/task_0/object", "Xform")
    UsdPhysics.CollisionAPI.Apply(root)
    _cube(stage, "/World/task_0/object/collider")
    task = types.SimpleNamespace(
        fixtures={}, objects={"object": types.SimpleNamespace(prim_path=str(root.GetPath()))}, distractors={}
    )
    manager = CollisionSceneManager(stage, task, {"strict": True})
    assert manager.collision_prim_paths == ["/World/task_0/object/collider"]
    assert manager.schema_exclusions == {
        "/World/task_0/object": "non_geometry_collision_api_with_supported_descendant_colliders"
    }


def test_empty_enabled_geometry_collider_is_excluded_from_world():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    empty = UsdGeom.Mesh.Define(
        stage, "/World/task_0/sink_table_named_asset/empty_collider"
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(empty)

    manager = CollisionSceneManager(stage, task, {"strict": True})

    path = str(empty.GetPath())
    assert path not in manager.collision_prim_paths
    assert manager.schema_exclusions[path] == "empty_enabled_geometry_collider"


def test_guide_purpose_collision_geometry_is_discovered():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    guide = _cube(stage, "/World/task_0/sink_table_named_asset/guide_collider")
    UsdGeom.Imageable(guide).CreatePurposeAttr(UsdGeom.Tokens.guide)

    manager = CollisionSceneManager(stage, task, {"strict": True})

    assert str(guide.GetPath()) in manager.collision_prim_paths


def test_explicit_visual_only_entity_is_skipped_but_missing_claimed_collider_is_strict():
    stage = Usd.Stage.CreateInMemory()
    visual_root = stage.DefinePrim("/World/task_0/rug", "Xform")
    _cube(stage, "/World/task_0/solid/collider")
    stage.DefinePrim("/World/task_0/solid", "Xform")
    task = types.SimpleNamespace(
        fixtures={
            "rug": types.SimpleNamespace(
                prim_path=str(visual_root.GetPath()),
                cfg={"physics": {"collision_enabled": False}, "collider": "none"},
            ),
            "solid": types.SimpleNamespace(prim_path="/World/task_0/solid"),
        },
        objects={},
        distractors={},
    )
    manager = CollisionSceneManager(stage, task, {"strict": True})
    assert "rug" not in manager.records
    assert manager.schema_exclusions[str(visual_root.GetPath())].startswith(
        "config_declared_visual_only"
    )

    broken = stage.DefinePrim("/World/task_0/broken", "Xform")
    task.fixtures["broken"] = types.SimpleNamespace(
        prim_path=str(broken.GetPath()),
        cfg={"physics": {"collision_enabled": True}},
    )
    with pytest.raises(CollisionSceneError, match="no supported enabled collider"):
        CollisionSceneManager(stage, task, {"strict": True})


def test_world_colliders_and_consolidated_attach_prim_are_separate_contracts():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/task_0/a", "Xform")
    _cube(stage, "/World/task_0/a/combined")
    _cube(stage, "/World/task_0/a/detail")
    entity = types.SimpleNamespace(
        prim_path="/World/task_0/a",
        attach_collision_prim_paths=["/World/task_0/a/combined"],
    )
    task = types.SimpleNamespace(
        fixtures={},
        objects={"a": entity},
        distractors={},
        cfg={
            "skills": [
                {"robot": [{"left": [{"name": "Pick", "objects": ["a"]}], "right": []}]}
            ]
        },
    )
    manager = CollisionSceneManager(stage, task, {"strict": True})
    assert manager.records["a"].collision_prim_paths == [
        "/World/task_0/a/combined",
        "/World/task_0/a/detail",
    ]
    assert manager.attach_prim_paths["a"] == ["/World/task_0/a/combined"]

    entity.attach_collision_prim_paths = ["/World/task_0/a/visual"]
    with pytest.raises(CollisionSceneError, match="not an enabled collider"):
        CollisionSceneManager(stage, task, {"strict": True})


def test_refresh_after_task_reset_re_discovers_reloaded_attach_path():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/task_0/glass", "Xform")
    _cube(stage, "/World/task_0/glass/Scan_009")
    entity = types.SimpleNamespace(
        prim_path="/World/task_0/glass",
        attach_collision_prim_paths=["/World/task_0/glass/Scan_009"],
    )
    task = types.SimpleNamespace(
        fixtures={},
        objects={"glass": entity},
        distractors={},
        cfg={
            "skills": [
                {"robot": [{"left": [{"name": "Pick", "objects": ["glass"]}], "right": []}]}
            ]
        },
    )
    manager = CollisionSceneManager(stage, task, {"strict": True})

    stage.RemovePrim("/World/task_0/glass")
    stage.DefinePrim("/World/task_0/glass", "Xform")
    _cube(stage, "/World/task_0/glass/Scan_013")
    entity.attach_collision_prim_paths = ["/World/task_0/glass/Scan_013"]

    manager.refresh_after_task_reset()

    assert manager.attach_prim_paths["glass"] == ["/World/task_0/glass/Scan_013"]
    assert manager.collision_prim_paths == ["/World/task_0/glass/Scan_013"]


def test_attached_pose_tracking_uses_the_rigidbody_that_carries_attach_collider():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/task_0/bottle", "Xform")
    body = stage.DefinePrim("/World/task_0/bottle/Aligned/base_link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(False)
    collider = _cube(stage, "/World/task_0/bottle/Aligned/base_link/collider")
    entity = types.SimpleNamespace(
        prim_path="/World/task_0/bottle",
        attach_collision_prim_paths=[str(collider.GetPath())],
    )
    task = types.SimpleNamespace(
        fixtures={},
        objects={"bottle": entity},
        distractors={},
        cfg={
            "skills": [
                {
                    "robot": [
                        {"left": [], "right": [{"name": "Pick", "objects": ["bottle"]}]}
                    ]
                }
            ]
        },
    )

    manager = CollisionSceneManager(stage, task, {"strict": True})

    assert manager.records["bottle"].root_prim_path == "/World/task_0/bottle"
    assert (
        manager.records["bottle"].tracking_prim_path
        == "/World/task_0/bottle/Aligned/base_link"
    )


def test_state_machine_rejects_illegal_and_concurrent_transitions():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_TRANSIT, "robot", "right"
    )
    with pytest.raises(CollisionSceneError, match="UNSUPPORTED_CONCURRENT_MANIPULATION"):
        manager._transition(  # pylint: disable=protected-access
            "sink_table_named_asset", CollisionObjectState.ACTIVE_TARGET_TRANSIT, "robot", "left"
        )
    with pytest.raises(CollisionSceneError, match="illegal collision state transition"):
        manager._transition(  # pylint: disable=protected-access
            "movable_b", CollisionObjectState.PLACED_WORLD
        )


def test_invariants_allow_only_explicit_terminal_support_disable():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    key = ("robot", "right")
    manager.controllers[key] = types.SimpleNamespace(
        has_attached_collision_spheres=lambda: True
    )
    manager.controller_enabled[key] = {
        path: True for path in manager.collision_prim_paths
    }
    manager._temporary_disabled[key] = set()  # pylint: disable=protected-access
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_TRANSIT, *key
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_APPROACH, *key
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ATTACHED, *key
    )
    target_path = manager.records["movable_b"].collision_prim_paths[0]
    support_path = manager.records["sink_table_named_asset"].collision_prim_paths[0]
    manager.controller_enabled[key][target_path] = False
    manager.controller_enabled[key][support_path] = False
    manager._temporary_disabled[key].add(support_path)  # pylint: disable=protected-access

    manager.assert_invariants()

    manager._temporary_disabled[key].clear()  # pylint: disable=protected-access
    with pytest.raises(CollisionSceneError, match="world object is disabled"):
        manager.assert_invariants()


def test_placed_object_can_enter_and_restore_terminal_retreat_identity():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_TRANSIT, "robot", "right"
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_APPROACH, "robot", "right"
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ATTACHED, "robot", "right"
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.PLACED_WORLD
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_APPROACH, "robot", "right"
    )
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.PLACED_WORLD
    )
    assert manager.records["movable_b"].state == CollisionObjectState.PLACED_WORLD


def test_active_physics_object_state_blocks_legacy_activation():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    controller = types.SimpleNamespace(name="robot", lr_name="right")
    manager.controllers[("robot", "right")] = controller
    manager.controller_enabled[("robot", "right")] = {
        path: True for path in manager.collision_prim_paths
    }
    manager._temporary_disabled[("robot", "right")] = set()  # pylint: disable=protected-access
    manager._transition(  # pylint: disable=protected-access
        "movable_b", CollisionObjectState.ACTIVE_TARGET_TRANSIT, "robot", "right"
    )

    with pytest.raises(CollisionSceneError, match="cannot activate legacy"):
        manager.prepare_controller_for_legacy(controller)


def test_attached_owner_lookup_and_carry_validation_are_exact(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    record = manager.records["movable_b"]
    record.state = CollisionObjectState.ATTACHED
    record.owner_robot = "robot"
    record.owner_arm = "right"
    monkeypatch.setattr(manager, "assert_invariants", lambda: None)

    assert manager.get_attached_entity("robot", "right") == "movable_b"
    assert manager.get_attached_entity("robot", "left") is None
    assert manager.assert_attached_owner("movable_b", "robot", "right") is record
    with pytest.raises(CollisionSceneError, match="owner mismatch"):
        manager.assert_attached_owner("movable_b", "robot", "left")


def test_world_collision_diagnostic_isolates_entity_and_restores_enabled_state():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    key = ("robot", "left")
    enabled = set(manager.collision_prim_paths)
    collision_path = manager.records["movable_b"].collision_prim_paths[0]

    def enable_obstacle(path, value):
        if value:
            enabled.add(path)
        else:
            enabled.discard(path)

    controller = types.SimpleNamespace(
        name=key[0],
        lr_name=key[1],
        world_cfg=types.SimpleNamespace(
            get_obstacle=lambda _path: object(),
        ),
        planner=types.SimpleNamespace(
            scene_collision_checker=types.SimpleNamespace(
                enable_obstacle=enable_obstacle,
            ),
        ),
        check_current_start_state=lambda: (
            collision_path not in enabled,
            None if collision_path not in enabled else "Start state is colliding with world",
        ),
    )
    manager.controllers[key] = controller
    manager.controller_enabled[key] = {
        path: True for path in manager.collision_prim_paths
    }
    manager._temporary_disabled[key] = set()  # pylint: disable=protected-access

    result = manager.diagnose_controller_world_collision(controller)

    assert result["baseline_without_world_valid"] is True
    assert [item["entity"] for item in result["colliding_entities"]] == [
        "movable_b"
    ]
    assert enabled == set(manager.collision_prim_paths)
    assert all(manager.controller_enabled[key].values())


def test_physics_sync_and_export_skip_legacy_controllers(monkeypatch, tmp_path):
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    physics = types.SimpleNamespace(
        name="robot",
        lr_name="left",
        collision_world_mode="physics_schema",
        reference_prim_path="/World/task_0",
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
        planner=types.SimpleNamespace(
            scene_collision_checker=types.SimpleNamespace(
                update_obstacle_pose=lambda *_args: None
            )
        ),
    )
    legacy = types.SimpleNamespace(
        name="robot",
        lr_name="right",
        collision_world_mode="legacy_stage_scan",
        reference_prim_path="/World/task_0",
        tensor_args=object(),
    )
    manager.controllers[("robot", "left")] = physics
    manager.controllers[("robot", "right")] = legacy

    class Helper:
        @staticmethod
        def get_pose(path, inverse=False):
            if inverse:
                return np.eye(4)
            assert path.endswith("/collider")
            return np.eye(4)

    manager._usd_parser = Helper()  # pylint: disable=protected-access
    fake_curobo = types.ModuleType("curobo")
    fake_types = types.ModuleType("curobo.types")
    fake_types.Pose = types.SimpleNamespace(
        from_matrix=lambda value: np.asarray(value)
    )
    fake_curobo.types = fake_types
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo.types", fake_types)

    root = stage.GetPrimAtPath("/World/task_0/movable_b")
    UsdGeom.Xformable(root).AddTranslateOp().Set((0.1, 0.0, 0.0))
    manager.sync_dynamic_poses(step_id=1, interval_steps=1, force=True)

    audited = []
    monkeypatch.setattr(manager, "audit_controller", lambda controller: audited.append(controller))
    manager.export(tmp_path)
    assert audited == [physics]


def test_contact_force_reduction_preserves_filter_identity_across_sensors():
    class View:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def get_contact_force_matrix(self):
            return self.values

    maxima = CollisionSceneManager._filter_force_maxima(  # pylint: disable=protected-access
        [
            View([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]]),
            View([[[0.0, 0.0, 1.0], [0.0, 6.0, 8.0]]]),
        ],
        filter_count=2,
    )
    np.testing.assert_allclose(maxima, [5.0, 10.0])


def test_robot_contact_can_exclude_only_the_expected_active_object():
    class View:
        def get_contact_force_matrix(self):
            return np.asarray([[[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]])

    manager = CollisionSceneManager.__new__(CollisionSceneManager)
    manager.records = {
        "apple": types.SimpleNamespace(collision_prim_paths=["/World/apple/collider"]),
        "table": types.SimpleNamespace(collision_prim_paths=["/World/table/collider"]),
    }
    manager._robot_environment_contact_views = {("robot", "left"): [View()]}
    manager._robot_contact_debug_reported = set()

    assert manager.get_unexpected_robot_contact_force("robot", "left") == 12.0
    assert (
        manager.get_unexpected_robot_contact_force(
            "robot", "left", allowed_entity="apple"
        )
        == 12.0
    )

    class TargetOnlyView:
        def get_contact_force_matrix(self):
            return np.asarray([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])

    manager._robot_environment_contact_views = {("robot", "left"): [TargetOnlyView()]}
    assert (
        manager.get_unexpected_robot_contact_force(
            "robot", "left", allowed_entity="apple"
        )
        == 0.0
    )


def test_pending_detach_window_is_explicit():
    manager = CollisionSceneManager.__new__(CollisionSceneManager)
    manager._pending_detach = {"apple"}

    assert manager.is_pending_detach("apple")
    assert not manager.is_pending_detach("table")
    assert not manager.is_pending_detach(None)


def test_native_usd_parser_builds_exact_name_bbox_world(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})

    class FakeCuboid:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.pose = kwargs["pose"]
            self.dims = kwargs["dims"]

    class FakeWorldConfig:
        def __init__(self, cuboid):
            self.cuboid = cuboid

        def get_collision_check_world(self):
            return self

    fake_curobo = types.ModuleType("curobo")
    fake_src = types.ModuleType("curobo._src")
    fake_geom = types.ModuleType("curobo._src.geom")
    fake_geom_types = types.ModuleType("curobo._src.geom.types")
    fake_geom_types.Cuboid = FakeCuboid
    fake_geom_types.SceneCfg = FakeWorldConfig
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo._src", fake_src)
    monkeypatch.setitem(sys.modules, "curobo._src.geom", fake_geom)
    monkeypatch.setitem(sys.modules, "curobo._src.geom.types", fake_geom_types)

    manager._usd_parser = types.SimpleNamespace()  # pylint: disable=protected-access
    result = manager.build_world_config("/World/task_0")

    assert [obstacle.name for obstacle in result.cuboid] == manager.collision_prim_paths
    assert all(len(obstacle.pose) == 7 for obstacle in result.cuboid)
    assert all(min(obstacle.dims) > 0.0 for obstacle in result.cuboid)


def test_native_bbox_proxy_uses_sibling_reference_and_applies_scale_once(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    reference = UsdGeom.Xform.Define(stage, "/World/robot_base")
    reference.AddTranslateOp().Set((2.5, 1.0, 0.7))
    wall = UsdGeom.Xform.Define(stage, "/World/wall")
    wall.AddTranslateOp().Set((2.0, 0.0, 1.4))
    wall.AddRotateXYZOp().Set((90.0, 0.0, 0.0))
    collider = UsdGeom.Cube.Define(stage, "/World/wall/collision")
    collider.CreateSizeAttr().Set(1.0)
    collider_xform = UsdGeom.Xformable(collider)
    collider_xform.AddScaleOp().Set((4.0, 2.8, 0.02))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    class FakeCuboid:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.pose = kwargs["pose"]
            self.dims = kwargs["dims"]

    fake_curobo = types.ModuleType("curobo")
    fake_src = types.ModuleType("curobo._src")
    fake_geom = types.ModuleType("curobo._src.geom")
    fake_geom_types = types.ModuleType("curobo._src.geom.types")
    fake_geom_types.Cuboid = FakeCuboid
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo._src", fake_src)
    monkeypatch.setitem(sys.modules, "curobo._src.geom", fake_geom)
    monkeypatch.setitem(sys.modules, "curobo._src.geom.types", fake_geom_types)

    manager = CollisionSceneManager.__new__(CollisionSceneManager)
    manager.stage = stage
    proxy = manager._bbox_collision_proxy(  # pylint: disable=protected-access
        "/World/wall/collision", "/World/robot_base"
    )

    np.testing.assert_allclose(proxy.pose[:3], [-0.5, -1.0, 0.7], atol=1e-6)
    np.testing.assert_allclose(proxy.dims, [4.0, 2.8, 0.02], atol=1e-6)


def test_native_usd_parser_computes_relative_dynamic_pose(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    reference_from_world = np.eye(4)
    reference_from_world[0, 3] = -1.0
    world_from_prim = np.eye(4)
    world_from_prim[1, 3] = 2.0

    class Helper:
        @staticmethod
        def get_pose(path, inverse=False):
            if path == "/World/robot_base" and inverse:
                return reference_from_world
            if path.endswith("/collider") and not inverse:
                return world_from_prim
            raise AssertionError((path, inverse))

    converted = []

    class FakePose:
        @staticmethod
        def from_matrix(value):
            converted.append(np.asarray(value))
            return "relative-pose"

    fake_curobo = types.ModuleType("curobo")
    fake_types = types.ModuleType("curobo.types")
    fake_types.Pose = FakePose
    fake_curobo.types = fake_types
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo.types", fake_types)

    manager._usd_parser = Helper()  # pylint: disable=protected-access
    controller = types.SimpleNamespace(
        reference_prim_path="/World/robot_base",
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
    )
    result = manager._controller_obstacle_pose(  # pylint: disable=protected-access
        controller, "/World/task_0/movable_b/collider"
    )

    assert result == "relative-pose"
    np.testing.assert_allclose(converted, [reference_from_world @ world_from_prim])


def test_dynamic_pose_sync_does_not_rebuild_collision_geometry(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    manager = CollisionSceneManager(stage, task, {"strict": True})

    pose_calls = []

    class Helper:
        def get_pose(self, path, inverse=False):
            pose_calls.append((path, inverse))
            if inverse:
                return np.eye(4)
            pose = np.eye(4)
            pose[:3, 3] = [1.0, 2.0, 3.0]
            return pose

        def get_obstacles_from_stage(self, *_args, **_kwargs):
            raise AssertionError("dynamic pose sync must not rebuild geometry")

    updates = []
    scene_collision_checker = types.SimpleNamespace(
        update_obstacle_pose=lambda path, pose: updates.append((path, pose))
    )
    controller = types.SimpleNamespace(
        reference_prim_path="/World/task_0",
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
        planner=types.SimpleNamespace(scene_collision_checker=scene_collision_checker),
    )
    manager.controllers[("robot", "right")] = controller
    manager._usd_parser = Helper()  # pylint: disable=protected-access

    fake_curobo = types.ModuleType("curobo")
    fake_types = types.ModuleType("curobo.types")
    class FakePose:
        @staticmethod
        def from_matrix(value):
            return np.asarray(value)

    fake_types.Pose = FakePose
    fake_curobo.types = fake_types
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo.types", fake_types)

    root = stage.GetPrimAtPath("/World/task_0/movable_b")
    UsdGeom.Xformable(root).AddTranslateOp().Set((0.1, 0.0, 0.0))
    changed = manager.sync_dynamic_poses(step_id=5, interval_steps=5)

    assert changed == ["movable_b"]
    assert pose_calls == [
        ("/World/task_0", True),
        ("/World/task_0/movable_b/collider", False),
    ]
    assert len(updates) == 1
    assert updates[0][0] == "/World/task_0/movable_b/collider"
    np.testing.assert_allclose(updates[0][1], np.array([[1.0, 0.0, 0.0, 1.0],
                                                         [0.0, 1.0, 0.0, 2.0],
                                                         [0.0, 0.0, 1.0, 3.0],
                                                         [0.0, 0.0, 0.0, 1.0]]))


def test_dynamic_pose_sync_skips_placement_contact_objects():
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    manager.records["movable_b"].state = CollisionObjectState.PLACEMENT_CONTACT
    root = stage.GetPrimAtPath("/World/task_0/movable_b")
    UsdGeom.Xformable(root).AddTranslateOp().Set((0.25, 0.0, 0.0))

    assert manager.sync_dynamic_poses(step_id=1, interval_steps=1) == []
