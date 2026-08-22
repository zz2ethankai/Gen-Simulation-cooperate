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
    PlannerScenePort,
)
from core.planning.domain_types import PlannerKind, PlannerRuntimeProfile  # noqa: E402
from core.planning.planner_runtime import PlannerRuntime  # noqa: E402
from core.planning.native_scene_adapter import NativeSceneAdapter  # noqa: E402
import core.planning.native_bridge as native_bridge  # noqa: E402


def _cube(stage, path, *, enabled=True):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(enabled)
    return prim


def _task(stage, *, cfg=None):
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
        cfg=cfg or {},
    )


def _scene_port(
    *,
    arm="left",
    checker=None,
    obstacle_names=None,
    check_current_start_state=None,
    has_attached_collision_spheres=None,
):
    """Build the narrow scene dependency used by the manager tests."""

    if checker is None:
        checker = types.SimpleNamespace(
            get_obstacle=lambda _path: object(),
            enable_obstacle=lambda *_args: None,
            update_obstacle_pose=lambda *_args: None,
        )
    if obstacle_names is None:
        obstacle_names = [
            "/World/task_0/sink_table_named_asset/collider",
            "/World/task_0/movable_b/collider",
        ]
    checker.get_obstacle_names = lambda: list(obstacle_names)
    kinematics = types.SimpleNamespace(
        config=types.SimpleNamespace(
            kinematics_config=types.SimpleNamespace(
                get_number_of_spheres=lambda _link_name: 8
            )
        )
    )
    planner = types.SimpleNamespace(
        scene_collision_checker=checker,
        kinematics=kinematics,
        update_world=lambda world: setattr(planner, "world", world),
    )
    planner_runtime = PlannerRuntime(planner=planner, scene_revision=0)
    native_scene_adapter = NativeSceneAdapter(planner, strict=True)
    planner_runtime.register_scene_adapter(native_scene_adapter)
    runtime = types.SimpleNamespace(
        native_planner=planner,
        batch_planner=None,
        planner_runtime=planner_runtime,
        world=None,
        scene_revision=0,
        native_scene_adapter=native_scene_adapter,
    )
    def adopt_scene_revision(revision):
        runtime.scene_revision = planner_runtime.adopt_scene_revision(revision)
        return runtime.scene_revision

    runtime.adopt_scene_revision = adopt_scene_revision
    def update_world(world):
        runtime.world = world
        runtime.scene_revision = planner_runtime.update_world(
            world, revision=runtime.scene_revision + 1
        )
        return runtime.scene_revision

    runtime.update_world = update_world
    port = PlannerScenePort(
        name="robot",
        lr_name=arm,
        reference_prim_path="/World/task_0",
        robot_ee_path="/World/task_0",
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
        robot=types.SimpleNamespace(),
        runtime=runtime,
        native_scene_adapter=native_scene_adapter,
        adopt_scene_revision=runtime.adopt_scene_revision,
        check_current_start_state=check_current_start_state,
        attach_collision_object=lambda _paths: None,
        detach_attachment=lambda: None,
        has_attached_collision_spheres=(
            has_attached_collision_spheres
            if has_attached_collision_spheres is not None
            else (lambda: False)
        ),
    )
    return port, planner, checker


def _formal_port_from_planner_runtime(runtime, *, arm):
    """Wrap a PlannerRuntime fake in the production MotionPlannerRuntime port."""

    planner = runtime.ensure_planner()
    adapter = NativeSceneAdapter(planner, strict=True)
    runtime.register_scene_adapter(adapter)
    wrapper = types.SimpleNamespace(
        native_planner=planner,
        batch_planner=None,
        planner_runtime=runtime,
        world=runtime.world,
        scene_revision=runtime.scene_revision,
        native_scene_adapter=adapter,
    )
    def adopt_scene_revision(revision):
        wrapper.scene_revision = runtime.adopt_scene_revision(revision)
        return wrapper.scene_revision

    def update_world(world):
        wrapper.world = world
        wrapper.scene_revision = runtime.update_world(
            world, revision=runtime.scene_revision + 1
        )
        return wrapper.scene_revision

    wrapper.adopt_scene_revision = adopt_scene_revision
    wrapper.update_world = update_world
    return PlannerScenePort(
        name="robot",
        lr_name=arm,
        reference_prim_path="/World/task_0",
        robot_ee_path="/World/task_0",
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
        robot=types.SimpleNamespace(),
        runtime=wrapper,
        native_scene_adapter=adapter,
        adopt_scene_revision=wrapper.adopt_scene_revision,
        attach_collision_object=lambda _paths: None,
        detach_attachment=lambda: None,
        has_attached_collision_spheres=lambda: False,
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


def test_planning_exclusions_resolve_exact_task_entity_names():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage, cfg={"planning": {"planning_exclusions": ["sink_table_named_asset"]}})
    manager = CollisionSceneManager(stage, task, {"strict": True})
    assert manager._planning_exclusion_names == ("sink_table_named_asset",)


def test_planning_exclusion_does_not_use_substring_or_prim_path():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    with pytest.raises(CollisionSceneError, match="does not name.*collision entity"):
        CollisionSceneManager(stage, task, {"strict": True, "planning_exclusions": ["sink"]})


def test_planning_exclusion_disables_all_colliders_for_exact_record():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage, cfg={"planning": {"planning_exclusions": ["sink_table_named_asset"]}})
    second = _cube(stage, "/World/task_0/sink_table_named_asset/collider_two")
    assert second.IsValid()
    manager = CollisionSceneManager(stage, task, {"strict": True})
    key = ("robot", "right")
    updates = []
    checker = types.SimpleNamespace(
        get_obstacle=lambda _path: object(),
        enable_obstacle=lambda path, enabled: updates.append((path, enabled)),
    )
    controller, _planner, _checker = _scene_port(
        arm=key[1], checker=checker, obstacle_names=manager.collision_prim_paths
    )
    manager.bind_scene_port(controller)

    manager.apply_controller_planning_exclusions(controller)

    excluded = manager.records["sink_table_named_asset"].collision_prim_paths
    assert set(updates) == {(path, False) for path in excluded}
    assert all(not manager.controller_enabled[key][path] for path in excluded)


def test_floor_collision_api_must_be_native_addressable_before_episode_reset():
    """Schema discovery cannot outrun the native world's exact name set."""

    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    stage.DefinePrim("/World/task_0/floor", "Xform")
    _cube(stage, "/World/task_0/floor/collision_volume")
    task.fixtures["floor"] = types.SimpleNamespace(prim_path="/World/task_0/floor")

    manager = CollisionSceneManager(stage, task, {"strict": True})
    floor_path = "/World/task_0/floor/collision_volume"
    assert floor_path in manager.collision_prim_paths

    missing_checker = types.SimpleNamespace(
        get_obstacle_names=lambda: [
            path for path in manager.collision_prim_paths if path != floor_path
        ],
        enable_obstacle=lambda *_args: None,
    )
    missing_port, _planner, _checker = _scene_port(
        checker=missing_checker,
        obstacle_names=[
            path for path in manager.collision_prim_paths if path != floor_path
        ],
    )
    with pytest.raises(CollisionSceneError, match="missing=.*collision_volume"):
        manager.bind_scene_port(missing_port)

    manager = CollisionSceneManager(stage, task, {"strict": True})
    updates = []
    complete_checker = types.SimpleNamespace(
        get_obstacle_names=lambda: list(manager.collision_prim_paths),
        enable_obstacle=lambda path, enabled: updates.append((path, enabled)),
        update_obstacle_pose=lambda *_args: None,
    )
    port, _planner, _checker = _scene_port(
        checker=complete_checker,
        obstacle_names=manager.collision_prim_paths,
    )
    manager.bind_scene_port(port)
    manager.reset_episode()

    assert (floor_path, True) in updates


def test_lazy_batch_scene_sync_replays_masks_and_cached_dynamic_poses():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    paths = [
        "/World/task_0/sink_table_named_asset/collider",
        "/World/task_0/movable_b/collider",
    ]

    class Checker:
        def __init__(self):
            self.enabled = []
            self.poses = []

        def get_obstacle_names(self):
            return list(paths)

        def check_obstacle_exists(self, name):
            return str(name) in paths

        def enable_obstacle(self, name, enabled):
            self.enabled.append((str(name), bool(enabled)))

        def update_obstacle_pose(self, name, pose):
            self.poses.append((str(name), pose))

    class Planner:
        def __init__(self):
            self.scene_collision_checker = Checker()
            self.worlds = []
            self.kinematics = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    kinematics_config=types.SimpleNamespace(
                        get_number_of_spheres=lambda _link: 8
                    )
                )
            )

        def update_world(self, world):
            self.worlds.append(world)

    def factory(profile=None, kind=None):
        del profile, kind
        return Planner()

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=factory,
            batch_planner_factory=factory,
        )
    )
    runtime.update_world({"complete": paths}, revision=7)
    port = _formal_port_from_planner_runtime(runtime, arm="right")
    manager = CollisionSceneManager(stage, task, {"strict": True})
    manager.bind_scene_port(port)
    key = ("robot", "right")
    manager.controller_enabled[key][paths[0]] = False
    manager._native_pose_cache[key][paths[1]] = "cached-pose"  # pylint: disable=protected-access

    batch = runtime.ensure_batch_planner()

    assert batch.worlds == [{"complete": paths}]
    assert (paths[0], False) in batch.scene_collision_checker.enabled
    assert (paths[1], "cached-pose") in batch.scene_collision_checker.poses
    assert manager._native_scene_adapters[key][1].world_revision == 7  # pylint: disable=protected-access


def test_lazy_batch_scene_sync_rejects_missing_exact_collider():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    paths = [
        "/World/task_0/sink_table_named_asset/collider",
        "/World/task_0/movable_b/collider",
    ]

    class Checker:
        def __init__(self, names):
            self.names = set(names)

        def get_obstacle_names(self):
            return list(self.names)

        def check_obstacle_exists(self, name):
            return str(name) in self.names

        def enable_obstacle(self, *_args):
            return None

    class Planner:
        def __init__(self, names):
            self.scene_collision_checker = Checker(names)
            self.worlds = []
            self.kinematics = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    kinematics_config=types.SimpleNamespace(
                        get_number_of_spheres=lambda _link: 8
                    )
                )
            )

        def update_world(self, world):
            self.worlds.append(world)

    def factory(profile=None, kind=None):
        del profile
        return Planner(paths if kind != PlannerKind.BATCH else paths[:1])

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=factory,
            batch_planner_factory=factory,
        )
    )
    runtime.update_world({"complete": paths}, revision=4)
    port = _formal_port_from_planner_runtime(runtime, arm="left")
    manager = CollisionSceneManager(stage, task, {"strict": True})
    manager.bind_scene_port(port)

    with pytest.raises(CollisionSceneError, match="missing exact colliders"):
        runtime.ensure_batch_planner()


def test_scene_adapter_unbind_removes_runtime_fanout_and_listener():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage)
    paths = [
        "/World/task_0/sink_table_named_asset/collider",
        "/World/task_0/movable_b/collider",
    ]

    class Checker:
        def get_obstacle_names(self):
            return list(paths)

        def check_obstacle_exists(self, name):
            return str(name) in paths

    class Planner:
        def __init__(self):
            self.scene_collision_checker = Checker()
            self.kinematics = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    kinematics_config=types.SimpleNamespace(
                        get_number_of_spheres=lambda _link: 8
                    )
                )
            )

        def update_world(self, world):
            self.world = world

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=lambda **_kwargs: Planner(),
            batch_planner_factory=lambda **_kwargs: Planner(),
        )
    )
    runtime.update_world({"complete": paths}, revision=3)
    port = _formal_port_from_planner_runtime(runtime, arm="right")
    manager = CollisionSceneManager(stage, task, {"strict": True})
    manager.bind_scene_port(port)
    assert len(runtime._scene_adapters) == 1  # pylint: disable=protected-access
    assert len(runtime._planner_listeners) == 1  # pylint: disable=protected-access

    manager.unbind_scene_port(port)

    # The single adapter belongs to the runtime; manager unbind removes only
    # its batch/materialization registrations.
    assert len(runtime._scene_adapters) == 1  # pylint: disable=protected-access
    assert runtime._planner_listeners == []  # pylint: disable=protected-access
    assert ("robot", "right") not in manager._native_scene_adapters  # pylint: disable=protected-access


def test_duplicate_task_entity_name_is_rejected_before_exclusion_resolution():
    stage = Usd.Stage.CreateInMemory()
    task = _task(stage, cfg={"planning": {"planning_exclusions": ["sink_table_named_asset"]}})
    task.objects["sink_table_named_asset"] = types.SimpleNamespace(
        prim_path="/World/task_0/movable_b"
    )
    with pytest.raises(CollisionSceneError, match="multiple records"):
        CollisionSceneManager(stage, task, {"strict": True})


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
    controller, _planner, _checker = _scene_port(
        arm=key[1], has_attached_collision_spheres=lambda: True
    )
    manager.bind_scene_port(controller)
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

    checker = types.SimpleNamespace(
        get_obstacle=lambda _path: object(),
        enable_obstacle=enable_obstacle,
    )
    controller, _planner, _checker = _scene_port(
        arm=key[1],
        checker=checker,
        check_current_start_state=lambda: (
            collision_path not in enabled,
            None if collision_path not in enabled else "Start state is colliding with world",
        ),
    )
    manager.bind_scene_port(controller)

    result = manager.diagnose_controller_world_collision(controller)

    assert result["baseline_without_world_valid"] is True
    assert [item["entity"] for item in result["colliding_entities"]] == [
        "movable_b"
    ]
    assert enabled == set(manager.collision_prim_paths)
    assert all(manager.controller_enabled[key].values())


def test_physics_sync_and_export_use_bound_scene_ports(monkeypatch, tmp_path):
    stage = Usd.Stage.CreateInMemory()
    manager = CollisionSceneManager(stage, _task(stage), {"strict": True})
    checker = types.SimpleNamespace(
        get_obstacle=lambda _path: object(),
        update_obstacle_pose=lambda *_args: None,
    )
    physics, _planner, _checker = _scene_port(arm="left", checker=checker)
    manager.bind_scene_port(physics)

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

    monkeypatch.setattr(native_bridge, "Cuboid", FakeCuboid)
    monkeypatch.setattr(native_bridge, "SceneCfg", FakeWorldConfig)

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

    monkeypatch.setattr(native_bridge, "Cuboid", FakeCuboid)

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
    controller, _planner, _checker = _scene_port()
    controller = PlannerScenePort(
        name=controller.name,
        lr_name=controller.lr_name,
        reference_prim_path="/World/robot_base",
        robot_ee_path=controller.robot_ee_path,
        tensor_args=controller.tensor_args,
        robot=controller.robot,
        runtime=controller.runtime,
        native_scene_adapter=controller.native_scene_adapter,
        adopt_scene_revision=controller.adopt_scene_revision,
    )
    result = manager._port_obstacle_pose(  # pylint: disable=protected-access
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
        get_obstacle=lambda _path: object(),
        update_obstacle_pose=lambda path, pose: updates.append((path, pose))
    )
    controller, _planner, _checker = _scene_port(
        arm="right", checker=scene_collision_checker
    )
    manager.bind_scene_port(controller)
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
