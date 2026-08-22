"""Host-side contracts for the typed Placement planning port."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.curobo.placement_planning import PlacementPlanningPort  # noqa: E402
from core.planning.collision_scene_manager import PlannerScenePort  # noqa: E402
from core.planning.domain_types import CollisionPolicy  # noqa: E402


class _Runtime:
    scene_revision = 12


class _Manager:
    def __init__(self):
        self.calls = []
        self.records = {}

    def refresh_controller_reference_world(self, port, *, force=False):
        self.calls.append(("refresh", port.name, force))

    def sync_dynamic_poses(self, step_id, *, interval_steps, force=False):
        self.calls.append(("sync", step_id, interval_steps, force))
        return []

    def assert_attached_owner(self, entity, robot, arm):
        self.calls.append(("attached", entity, robot, arm))

    def begin_placement_descent(self, entity, support, robot, arm):
        self.calls.append(("descent", entity, support, robot, arm))

    def restore_world(self, entity):
        self.calls.append(("restore", entity))

    def has_native_obstacle(self, _port, path):
        return path == "/World/target/mesh"


def _scene_port():
    return PlannerScenePort(
        name="robot",
        lr_name="right",
        reference_prim_path="/World/robot/base",
        robot_ee_path="/World/robot/ee",
        tensor_args=SimpleNamespace(),
        robot=SimpleNamespace(),
        runtime=_Runtime(),
    )


def _port(manager, requests=None):
    requests = [] if requests is None else requests

    def single(_position, _orientation, **kwargs):
        requests.append(("single", kwargs))
        return "single-result"

    def batch(_positions, _orientations, **kwargs):
        requests.append(("batch", kwargs))
        return "batch-result"

    def from_path(_position, _orientation, _path, **kwargs):
        requests.append(("path", kwargs))
        return "path-result"

    return PlacementPlanningPort(
        scene_port=_scene_port(),
        collision_scene_manager=manager,
        execution_ee_pose=lambda: ("position", "orientation"),
        execution_forward_kinematic=lambda joints, **_kwargs: (joints, "orientation"),
        sync_native_batch_attachment=lambda **_kwargs: True,
        arm_base_transform=lambda: "base",
        plan_pose=single,
        plan_pose_batch=batch,
        plan_pose_result=single,
        plan_pose_from_path=from_path,
        measure_cartesian_path=lambda *_args: (1.0, 0.0),
        robot_file="panda_right.yml",
        batch_capability=True,
    )


def test_placement_port_uses_formal_scene_port_and_typed_transitions():
    manager = _Manager()
    planning = _port(manager)

    assert planning.prepare_world("object", "support") == 12
    assert manager.calls[:3] == [
        ("refresh", "robot", True),
        ("sync", 0, 1, True),
        ("attached", "object", "robot", "right"),
    ]

    assert (
        planning.transition_target(
            "object", "support", collision_policy=CollisionPolicy.PLACEMENT_DESCENT
        )
        == 12
    )
    assert manager.calls[-1] == (
        "descent",
        "object",
        "support",
        "robot",
        "right",
    )
    planning.restore_world("object")
    assert manager.calls[-1] == ("restore", "object")


def test_placement_port_stamps_target_support_and_policy_on_queries():
    manager = _Manager()
    requests = []
    planning = _port(manager, requests)
    planning.prepare_world("object", "support")

    planning.plan_pose_batch(
        [[0.0, 0.0, 0.1]],
        [[1.0, 0.0, 0.0, 0.0]],
        collision_policy=CollisionPolicy.ATTACHED_CARRY,
    )
    assert requests[-1][1]["request_metadata"] == {
        "phase_id": "place_preplace_batch",
        "collision_policy": CollisionPolicy.ATTACHED_CARRY,
        "active_target": "object",
        "support": "support",
    }

    planning.plan_pose_result(
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.0],
        collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
    )
    assert requests[-1][1]["request_metadata"]["phase_id"] == "place_terminal"
    assert requests[-1][1]["request_metadata"]["support"] == "support"

    planning.plan_pose_from_path(
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.0],
        "preplace-path",
        collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
    )
    assert requests[-1][1]["request_metadata"]["collision_policy"] is CollisionPolicy.PLACEMENT_DESCENT
    assert planning.has_native_obstacle("/World/target/mesh")


def test_place_and_dexplace_do_not_reach_through_controller_planner_fields():
    for filename in ("place.py", "dexplace.py"):
        path = ROOT / "workflows" / "simbox" / "core" / "skills" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        for forbidden in (
            "controller.collision_scene_manager",
            "controller.collision_world_mode",
            "controller.plan_pose",
            "controller.plan_pose_batch",
            "controller.plan_pose_result",
            "controller.plan_pose_from_path",
            "controller.sync_native_batch_attachment",
        ):
            assert forbidden not in source
