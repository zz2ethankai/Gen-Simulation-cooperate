"""Contract tests for the typed CuRobo-v2 collision-policy boundary."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.runtime import MotionPlannerRuntime  # noqa: E402
from core.planning.domain_types import (  # noqa: E402
    CollisionOptions,
    CollisionPolicy,
    PosePlanRequest,
    PlanningProfile,
)
from core.planning.native_planner_adapter import (  # noqa: E402
    NativeCollisionPolicyError,
    NativePlannerAdapter,
    map_collision_policy,
)
from core.planning.planner_runtime import PlannerCallError, PlannerRuntime  # noqa: E402


def _request(policy, *, options=None, target="/World/object", support="/World/support"):
    return PosePlanRequest(
        goal="goal",
        start_state="state",
        collision_policy=policy,
        collision_options=options,
        active_target=target,
        support=support,
    )


@pytest.mark.parametrize(
    ("policy", "expected_disable", "requires_spheres"),
    [
        (CollisionPolicy.WORLD_TRANSIT, (), False),
        (CollisionPolicy.TARGET_APPROACH, ("/World/object/collider",), False),
        (CollisionPolicy.ATTACHED_CARRY, ("/World/object/collider",), True),
        (
            CollisionPolicy.PLACEMENT_DESCENT,
            ("/World/object/collider", "/World/support/collider"),
            True,
        ),
        (CollisionPolicy.RETREAT, ("/World/object/collider",), False),
    ],
)
def test_native_collision_policy_mapping_is_explicit_and_deterministic(
    policy, expected_disable, requires_spheres
):
    options = CollisionOptions(
        policy=policy,
        target_obstacles=("/World/object/collider",),
        support_obstacles=("/World/support/collider",),
        attached_obstacles=("/World/object/collider",),
        allow_target_contact=policy is CollisionPolicy.TARGET_APPROACH,
        allow_object_support_contact=policy is CollisionPolicy.PLACEMENT_DESCENT,
    )
    mapped = map_collision_policy(_request(policy, options=options))

    assert mapped.policy is policy
    assert mapped.disable_obstacles == expected_disable
    assert mapped.require_attached_spheres is requires_spheres
    assert mapped.native_expressible is True
    assert mapped.enable_obstacles == ()


def test_passthrough_is_not_silently_sent_to_native_planner():
    mapped = map_collision_policy(
        _request(
            CollisionPolicy.PASSTHROUGH,
            options=CollisionOptions(
                policy=CollisionPolicy.PASSTHROUGH,
                target_obstacles=("/World/object/collider",),
            ),
        )
    )
    assert mapped.native_expressible is False
    assert "execution-only" in (mapped.unsupported_reason or "")

    with pytest.raises(NativeCollisionPolicyError):
        NativePlannerAdapter(types.SimpleNamespace()).plan_pose(
            _request(
                CollisionPolicy.PASSTHROUGH,
                options=CollisionOptions(policy=CollisionPolicy.PASSTHROUGH),
            )
        )


class _SceneChecker:
    def __init__(self, names):
        self.names = set(names)
        self.events = []

    def get_obstacle_names(self):
        return tuple(sorted(self.names))

    def check_obstacle_exists(self, name):
        return name in self.names

    def enable_obstacle(self, name, enabled):
        self.events.append((str(name), bool(enabled)))


class _AttachedPlanner:
    def __init__(self):
        self.scene_collision_checker = _SceneChecker(
            {"/World/object/collider", "/World/support/collider"}
        )
        self.attachment_manager = types.SimpleNamespace(
            has_attached_collision_spheres=lambda link_name="attached_object": True
        )
        self.calls = []

    def plan_pose(self, goal, current_state, **kwargs):
        self.calls.append((goal, current_state, kwargs))
        return types.SimpleNamespace(success=True, path=[goal])


def test_attached_carry_temporarily_disables_support_and_preserves_native_profile():
    planner = _AttachedPlanner()
    runtime = PlannerRuntime(planner=planner)
    request = _request(
        CollisionPolicy.ATTACHED_CARRY,
        options=CollisionOptions(
            policy=CollisionPolicy.ATTACHED_CARRY,
            target_obstacles=("/World/object/collider",),
            support_obstacles=("/World/support/collider",),
            attached_obstacles=("/World/object/collider",),
            allow_object_support_contact=True,
            allow_target_robot_contact=True,
        ),
    )
    request.kwargs = types.MappingProxyType(
        {"max_attempts": 4, "enable_graph_attempt": 1}
    )

    result = runtime.plan_pose(request)

    assert result.success is True
    assert planner.calls[0][2] == {"max_attempts": 4, "enable_graph_attempt": 1}
    assert planner.scene_collision_checker.events == [
        ("/World/object/collider", False),
        ("/World/support/collider", False),
        ("/World/support/collider", True),
    ]


def test_request_builder_resolves_logical_entities_to_exact_native_paths():
    runtime = object.__new__(MotionPlannerRuntime)
    runtime.planner_runtime = types.SimpleNamespace(scene_revision=11)
    runtime.robot_port = types.SimpleNamespace(
        collision_scene_manager=types.SimpleNamespace(
            records={
                "object": types.SimpleNamespace(
                    collision_prim_paths=["/World/object/collider"]
                ),
                "support": {"collision_prim_paths": ["/World/support/collider"]},
            }
        )
    )
    runtime.attachment_runtime = types.SimpleNamespace(
        attached_obstacle_names=("/World/object/collider",)
    )

    common = runtime._request_common_kwargs(
        {
            "collision_policy": CollisionPolicy.ATTACHED_CARRY,
            "collision_options": CollisionOptions(
                policy=CollisionPolicy.ATTACHED_CARRY,
                allow_object_support_contact=True,
            ),
            "active_target": "object",
            "support": "support",
            "profile": PlanningProfile.ATTACHED_CARRY,
        },
        default_profile=PlanningProfile.TRANSIT,
    )

    assert common["collision_options"].target_obstacles == (
        "/World/object/collider",
    )
    assert common["collision_options"].support_obstacles == (
        "/World/support/collider",
    )
    assert common["collision_options"].attached_obstacles == (
        "/World/object/collider",
    )
