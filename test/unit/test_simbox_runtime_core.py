"""Pure-Python contract tests for the SimBox planner/runtime boundary."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.attachment_runtime import AttachmentRuntime  # noqa: E402
from core.planning.domain_types import (  # noqa: E402
    AttachmentSpec,
    BatchPlanResult,
    BatchPosePlanRequest,
    CollisionOptions,
    CommandStatus,
    CspacePlanRequest,
    GripperCommand,
    HoldCommand,
    JointCommand,
    JointTrajectory,
    PlanResult,
    PlannerKind,
    PlannerRuntimeProfile,
    PosePlanRequest,
    CollisionPolicy,
    PlannerStatus,
    PlanningProfile,
    PoseCommand,
    SceneCommand,
)
import core.planning.domain_types as domain_types  # noqa: E402
from core.planning.planner_runtime import (  # noqa: E402
    PlannerDestroyedError,
    PlannerRuntime,
    StaleSceneError,
)
from core.planning.native_scene_adapter import (  # noqa: E402
    NativeSceneAdapter,
    NativeSceneAdapterError,
)
from core.planning.scene_runtime import SceneRuntime  # noqa: E402
from core.runtime import ArmSpec, JointOrderError, RobotRuntime  # noqa: E402
from core.controllers.controller_component import (  # noqa: E402
    ComponentPort,
    ComponentState,
    MutableExecutionState,
    PlanningConfig,
)
from core.controllers.controller_execution import ControllerExecution  # noqa: E402
from core.controllers.phase_executor import PhaseExecutor  # noqa: E402
from core.controllers.runtime import MotionPlannerRuntime  # noqa: E402
from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402


class _Planner:
    def __init__(self, *, fail_attach: bool = False, batch: bool = False):
        self.worlds = []
        self.poses = []
        self.calls = []
        self.destroyed = 0
        self.fail_attach = fail_attach
        self.batch = batch
        self.scene_collision_checker = self

    def update_world(self, world):
        self.worlds.append(world)

    def update_obstacle_pose(self, name, pose):
        self.poses.append((name, pose))

    def plan_pose(self, goal, state=None, **kwargs):
        self.calls.append(("pose", goal, state, kwargs))
        if self.batch:
            return type(
                "Result",
                (),
                {
                    "success": [True, False],
                    "interpolated_trajectory": type(
                        "Trajectory",
                        (),
                        {
                            "position": [
                                [[goal, "q0"]],
                                [[goal, "q1"]],
                            ],
                            "joint_names": ["joint_0", "joint_1"],
                        },
                    )(),
                },
            )()
        return type(
            "Result",
            (),
            {"success": True, "path": [goal], "metrics": {"position_error": 0.01}},
        )()

    def plan_cspace(self, goal, state=None, **kwargs):
        self.calls.append(("cspace", goal, state, kwargs))
        return type("Result", (), {"success": True, "path": [goal]})()

    def destroy(self):
        self.destroyed += 1


class _AttachmentManager:
    def __init__(self, fail_on=None):
        self.events = []
        self.fail_on = fail_on

    def attach(self, state, meshes, **kwargs):
        self.events.append(("attach", state, meshes, kwargs))
        if self.fail_on == "attach":
            raise RuntimeError("attach failed")

    def detach(self):
        self.events.append(("detach",))
        if self.fail_on == "detach":
            raise RuntimeError("detach failed")


def test_planner_runtime_keeps_batch_lazy_and_updates_late_batch_world_once():
    made = []

    def factory(profile=None, kind=None):
        made.append(kind)
        return _Planner(batch=kind == PlannerKind.BATCH)

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(planner_factory=factory, batch_planner_factory=factory)
    )
    runtime.update_world({"obstacle": 1})
    assert made == [PlannerKind.SINGLE]
    assert runtime.planner.worlds == [{"obstacle": 1}]

    result = runtime.plan_pose(PosePlanRequest("goal", "state"))
    assert result.success
    assert isinstance(result.trajectory, JointTrajectory)
    assert result.trajectory == ["goal"]
    assert result.trajectory.joint_names == ()
    assert not hasattr(result, "raw")
    assert result.source == "native"
    typed_pose = runtime.plan_pose(
        PosePlanRequest(goal="typed", start_state="state", request_id="req-1")
    )
    assert isinstance(typed_pose, PlanResult)
    assert typed_pose.request_id == "req-1"
    assert typed_pose.metrics["position_error"] == 0.01
    typed_batch = runtime.plan_pose_batch(
        BatchPosePlanRequest(goals=["g0", "g1"], start_state="state", batch_size=2)
    )
    assert isinstance(typed_batch, BatchPlanResult)
    assert typed_batch.success_mask == (True, False)
    assert typed_batch.trajectories[0].joint_names == ("joint_0", "joint_1")
    assert typed_batch.trajectories[1] is None
    assert not hasattr(typed_batch, "raw")
    assert made == [PlannerKind.SINGLE, PlannerKind.BATCH]
    assert runtime.batch_planner.worlds == [{"obstacle": 1}]

    runtime.destroy()
    assert runtime.status == PlannerStatus.DESTROYED
    assert runtime.planner is None
    with pytest.raises(PlannerDestroyedError):
        runtime.plan_pose(PosePlanRequest("goal"))


def test_late_batch_materialization_carries_scene_revision_to_listeners():
    made = []

    def factory(profile=None, kind=None):
        planner = _Planner(batch=kind == PlannerKind.BATCH)
        made.append((kind, planner))
        return planner

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(planner_factory=factory, batch_planner_factory=factory)
    )
    runtime.update_world({"complete": "scene"}, revision=11)
    runtime.update_obstacle_pose("/World/a/collider", "pose", revision=12)
    events = []
    runtime.register_planner_listener(
        lambda planner, kind, world, revision: events.append(
            (planner, kind, world, revision)
        ),
        replay=False,
    )

    batch = runtime.ensure_batch_planner()

    assert batch.worlds == [{"complete": "scene"}]
    assert events == [(batch, PlannerKind.BATCH, {"complete": "scene"}, 12)]
    assert batch.poses == [("/World/a/collider", "pose")]
    assert runtime.batch_planner is batch
    assert runtime._batch_world_synced is True  # pylint: disable=protected-access


def test_batch_materialization_listener_failure_is_atomic_and_retries_audit():
    made = []

    def factory(profile=None, kind=None):
        del profile
        planner = _Planner(batch=kind == PlannerKind.BATCH)
        made.append(planner)
        return planner

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=factory,
            batch_planner_factory=factory,
        )
    )
    runtime.update_world({"complete": "scene"}, revision=5)
    failures = []

    def reject_batch(planner, kind, world, revision):
        del planner, kind, world, revision
        failures.append(True)
        raise RuntimeError("strict batch audit failed")

    runtime.register_planner_listener(reject_batch, replay=False)
    with pytest.raises(RuntimeError, match="strict batch audit failed"):
        runtime.ensure_batch_planner()
    assert runtime.batch_planner is None
    assert runtime._batch_world_synced is False  # pylint: disable=protected-access
    assert made[-1].destroyed == 1
    assert failures == [True]

    runtime.unregister_planner_listener(reject_batch)
    batch = runtime.ensure_batch_planner()
    assert runtime.batch_planner is batch
    assert batch.worlds == [{"complete": "scene"}]


def test_native_scene_adapter_uses_public_v2_presence_shapes():
    class Checker:
        def get_obstacle_names(self):
            return ("/World/a/collider", "/World/b/collider")

        def check_obstacle_exists(self, name):
            return name in self.get_obstacle_names()

    adapter = NativeSceneAdapter(Checker(), strict=True)

    assert adapter.get_obstacle_names() == (
        "/World/a/collider",
        "/World/b/collider",
    )
    assert adapter.check_obstacle_exists("/World/a/collider") is True
    assert adapter.check_obstacle_exists("/World/missing/collider") is False
    adapter.require_obstacles(
        ["/World/a/collider", "/World/b/collider"], exact=True
    )
    assert not hasattr(adapter, "get_obstacle")
    with pytest.raises(NativeSceneAdapterError, match="missing"):
        adapter.require_obstacles(["/World/missing/collider"])


def test_planner_runtime_rejects_untyped_and_legacy_plan_inputs():
    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))
    with pytest.raises(TypeError, match="PosePlanRequest"):
        runtime.plan_pose("goal")
    with pytest.raises(TypeError, match="PosePlanRequest"):
        runtime.plan_pose({"goal": "goal"})
    with pytest.raises(AttributeError):
        runtime.plan(PosePlanRequest("goal"))


def test_scene_runtime_increments_only_on_change_and_fans_dynamic_poses():
    first = _Planner()
    second = _Planner()
    scene = SceneRuntime({"a": 1})
    scene.subscribe(first)
    scene.subscribe(second)
    assert first.worlds == [{"a": 1}]
    assert scene.update_world({"a": 1}).changed is False
    assert scene.revision == 0
    scene.update_world({"a": 2})
    scene.update_poses({"moving": (1, 2, 3)})
    assert scene.revision == 2
    assert first.worlds[-1] == {"a": 2}
    assert second.poses[-1] == ("moving", (1, 2, 3))


def test_planner_runtime_scene_binding_propagates_world_revision_to_late_batch():
    made = []

    def factory(profile=None, kind=None):
        del profile
        planner = _Planner(batch=kind == PlannerKind.BATCH)
        made.append(planner)
        return planner

    scene = SceneRuntime({"world": 0})
    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=factory,
            batch_planner_factory=factory,
        ),
        scene=scene,
    )
    scene.update_world({"world": 1})
    assert runtime.world_revision == scene.revision == 1

    batch = runtime.ensure_batch_planner()
    assert batch.worlds == [{"world": 1}]
    assert runtime.world_revision == 1


def test_planner_request_rejects_stale_scene_unless_policy_allows_it():
    planner = _Planner()
    runtime = PlannerRuntime(planner=planner, scene_revision=3)
    with pytest.raises(StaleSceneError):
        runtime.plan_pose(PosePlanRequest("goal", world_revision=2))
    result = runtime.plan_pose(
        PosePlanRequest(
            "goal",
            world_revision=2,
            collision_options=CollisionOptions(
                policy=CollisionPolicy.WORLD_TRANSIT,
                allow_stale_scene=True,
            ),
        )
    )
    assert result.success


def test_attachment_failure_restores_previous_manager_state():
    manager = _AttachmentManager()
    runtime = AttachmentRuntime(manager)
    runtime.attach(AttachmentSpec("old", state="s0", meshes=["m0"]))
    manager.fail_on = "attach"
    with pytest.raises(RuntimeError, match="attach failed"):
        runtime.attach(AttachmentSpec("new", state="s1", meshes=["m1"]))
    assert runtime.attached_names == ("old",)
    # The failed new attach is followed by detach + reattach of old.
    assert manager.events[-2][0] == "detach"
    assert manager.events[-1][0] == "attach"


def test_controller_attachment_batch_sync_audits_target_and_rolls_back():
    from core.controllers.controller_attachment import ControllerAttachment

    class BatchAttachmentManager:
        def __init__(self):
            self.events = []
            self.fail_on = None

        def attach(self, state, meshes, **kwargs):
            self.events.append(("attach", state, meshes, kwargs))
            if self.fail_on == "attach":
                raise RuntimeError("batch attach failed")

        def detach(self):
            self.events.append(("detach",))

    class BatchPlanner:
        def __init__(self, manager):
            self.attachment_manager = manager

    batch_manager = BatchAttachmentManager()
    batch = BatchPlanner(batch_manager)
    batch_attachment_runtime = AttachmentRuntime(batch_manager, strict=False)
    attachment_runtime = AttachmentRuntime(_AttachmentManager(), strict=False)
    attachment_runtime.attach(
        AttachmentSpec(
            name="/World/object/collider",
            state="single-state",
            meshes=["single-mesh"],
            disable_obstacle_names=("/World/object/collider",),
        )
    )
    runtime = types.SimpleNamespace(
        ensure_batch_planner=lambda: batch,
        attachment_runtime=attachment_runtime,
        batch_attachment_runtime=batch_attachment_runtime,
    )
    checker = types.SimpleNamespace(
        get_obstacle_names=lambda: ["/World/object/collider"],
        check_obstacle_exists=lambda name: str(name) == "/World/object/collider",
    )
    adapter = NativeSceneAdapter(types.SimpleNamespace(scene_collision_checker=checker))
    component = object.__new__(ControllerAttachment)
    component.runtime = runtime
    component.batch_capability = True
    component._require_batch_scene_adapter = lambda: adapter
    component._attach_batch_runtime_spec = (
        lambda paths, **_kwargs: batch_attachment_runtime.attach(
            AttachmentSpec(
                name="|".join(paths),
                state="batch-state",
                meshes=["batch-mesh"],
                disable_obstacle_names=tuple(paths),
            )
        )
    )
    batch_manager.fail_on = "attach"

    with pytest.raises(RuntimeError, match="batch attach failed"):
        component.sync_native_batch_attachment()
    assert batch_attachment_runtime.attached_names == ()
    assert batch_attachment_runtime.attached_obstacle_names == ()
    assert [event[0] for event in batch_manager.events] == ["attach", "detach"]

    missing_checker = types.SimpleNamespace(
        get_obstacle_names=lambda: [],
        check_obstacle_exists=lambda _name: False,
    )
    component._require_batch_scene_adapter = lambda: NativeSceneAdapter(
        types.SimpleNamespace(scene_collision_checker=missing_checker)
    )
    with pytest.raises(NativeSceneAdapterError, match="missing"):
        component.sync_native_batch_attachment()


def test_controller_attachment_batch_sync_requires_registered_target_adapter():
    """Batch attachment must not synthesize an unregistered native adapter."""

    from core.controllers.controller_attachment import ControllerAttachment

    batch = types.SimpleNamespace(
        attachment_manager=types.SimpleNamespace(detach=lambda: None)
    )
    component = object.__new__(ControllerAttachment)
    component.runtime = types.SimpleNamespace(ensure_batch_planner=lambda: batch)
    component.batch_capability = True

    with pytest.raises(RuntimeError, match="strict target-batch adapter"):
        component.sync_native_batch_attachment()


def test_attachment_names_and_reset_are_owned_by_attachment_runtime():
    """Runtime reset uses formal attachment ports rather than façade state lists."""

    runtime = object.__new__(MotionPlannerRuntime)
    single_manager = _AttachmentManager()
    batch_manager = _AttachmentManager()
    runtime.attachment_runtime = AttachmentRuntime(single_manager, strict=False)
    runtime.batch_attachment_runtime = AttachmentRuntime(batch_manager, strict=False)
    spec = AttachmentSpec(
        name="object",
        state="state",
        meshes=["mesh"],
        disable_obstacle_names=("/World/object/collider",),
    )

    runtime.attach_object(spec)
    runtime.batch_attachment_runtime.attach(spec)
    assert runtime.attachment_runtime.attached_names == ("object",)
    assert runtime.batch_attachment_runtime.attached_names == ("object",)
    assert not hasattr(runtime, "_native_attached_obstacle_names")
    assert not hasattr(runtime, "_native_batch_attached_obstacle_names")

    runtime.reset_attachments()
    assert runtime.attachment_runtime.attached_names == ()
    assert runtime.batch_attachment_runtime.attached_names == ()
    assert runtime.attachment_runtime.attached_obstacle_names == ()
    assert runtime.batch_attachment_runtime.attached_obstacle_names == ()


def test_host_component_wiring_shares_state_across_typed_plan_execution_and_reset():
    """A typed phase crosses phase/planning/execution ports without state copies."""

    state = MutableExecutionState()
    phase_executor = PhaseExecutor()
    planning_config = PlanningConfig()
    begin_calls = []
    plan_calls = []
    ee_reset_calls = []

    class _PhaseComponent(ComponentState):
        def begin(self, command):
            if command is self._active_phase_command:
                return False
            self._active_phase_command = command
            self._phase_plan_started = False
            self._phase_plan_finished = False
            self._phase_plan_failed = False
            self._phase_tracking_failed = False
            self._phase_dwell_count = 0
            self._last_command_name = command.phase.value
            begin_calls.append(command)
            return True

    class _PlanningComponent(ComponentState):
        def plan(self):
            plan_calls.append(True)
            return PlanResult(
                success=True,
                trajectory=JointTrajectory(
                    positions=[[0.1], [0.2]],
                    joint_names=("joint_0",),
                ),
            )

        def install(self, path, **_kwargs):
            self.phase_executor.install(path)
            self._phase_plan_started = True
            return path

    phase = _PhaseComponent(
        ComponentPort(
            {
                "execution_state": state,
                "planning_config": planning_config,
            }
        )
    )
    planning = _PlanningComponent(
        ComponentPort(
            {
                "execution_state": state,
                "planning_config": planning_config,
                "phase_executor": phase_executor,
            }
        )
    )

    class _Robot:
        dof_names = ("joint_0", "gripper_0")

        @staticmethod
        def get_joints_state():
            return types.SimpleNamespace(positions=np.asarray([0.0, 0.0]))

    robot = _Robot()
    runtime = types.SimpleNamespace(plan_cspace=lambda *_args, **_kwargs: planning.plan())
    execution = ControllerExecution(
        ComponentPort(
            {
                "execution_state": state,
                "planning_config": planning_config,
                "phase_executor": phase_executor,
                "runtime": runtime,
                "tensor_args": types.SimpleNamespace(
                    to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
                ),
                "raw_js_names": ["joint_0"],
                "name": "robot",
                "lr_name": "left",
                "robot": robot,
                "arm_indices": np.asarray([0]),
                "gripper_indices": np.asarray([1]),
                "ds_ratio": 1,
                "_begin_phase_command": phase.begin,
                "_install_command_plan": planning.install,
                "_result_success": lambda result: result.success,
                "_result_path": lambda result: result.trajectory,
                "_command_path": lambda path: path,
                "_log_plan_result": lambda *_args, **_kwargs: None,
                "collision_scene_manager": types.SimpleNamespace(
                    begin_target_transit=lambda *_args: None
                ),
            }
        )
    )
    execution.get_gripper_action = lambda: np.asarray([0.0])

    target_position = np.asarray([0.1, 0.2, 0.3])
    target_orientation = np.asarray([1.0, 0.0, 0.0, 0.0])

    def ee_reset_callback():
        ee_reset_calls.append(True)
        return target_position.copy(), target_orientation.copy()

    execution.get_ee_pose = ee_reset_callback

    def injected_plan_and_forward(position, orientation, **_kwargs):
        if not phase_executor.active:
            planning.install(
                planning.plan().trajectory,
                target_position=position,
                target_orientation=orientation,
            )
        return execution._forward_installed_joint_path()

    execution.ee_forward = injected_plan_and_forward
    command = MotionPhaseCommand(
        phase=MotionPhase.TRANSIT_PREGRASP,
        target_position=target_position,
        target_orientation=target_orientation,
        phase_id="pick.pregrasp",
    )

    first_action = execution.forward_phase_command(command)
    first_status = execution.execution_status(command)
    np.testing.assert_allclose(first_action["arm_action"], [0.1])
    assert first_status.status is CommandStatus.ACTIVE
    assert first_status.plan_active is True

    second_action = execution.forward_phase_command(command)
    second_status = execution.execution_status(command)
    np.testing.assert_allclose(second_action["arm_action"], [0.2])
    assert second_status.status is CommandStatus.COMPLETED
    assert second_status.complete is True
    assert begin_calls == [command]
    assert plan_calls == [True]
    assert ee_reset_calls
    assert execution.execution_state is state
    assert phase.execution_state is state
    assert planning.execution_state is state
    assert state.active_phase_command is command
    assert state.phase_plan_started is True
    assert state.phase_plan_finished is True

    state.reset()
    assert execution._active_phase_command is None
    assert phase._active_phase_command is None
    assert planning._phase_plan_finished is False


def test_robot_runtime_reorders_named_joint_state_and_rejects_missing_joint():
    runtime = RobotRuntime(
        ["unrelated", "joint_2", "gripper", "joint_1"],
        {
            "left": ArmSpec("left", ["joint_1", "joint_2"], gripper_names=["gripper"])
        },
    )
    assert runtime.arm_indices("left") == (3, 1)
    assert runtime.gripper_indices("left") == (2,)
    assert runtime.reorder(
        [10, 20, 30, 40],
        source_names=["unrelated", "joint_2", "gripper", "joint_1"],
        target_names=["joint_1", "joint_2"],
    ) == [40, 20]
    with pytest.raises(JointOrderError, match="missing"):
        RobotRuntime(["joint_1"], {"left": ArmSpec("left", ["joint_2"])})


def test_public_domain_types_are_finite_and_commands_are_discriminated():
    assert not hasattr(domain_types, "_LegacyPlannerRequest")
    assert not hasattr(domain_types, "_LegacyPlannerResult")
    assert not hasattr(domain_types, "PlannerProfile")
    assert {item.name for item in PlanningProfile} == {
        "TRANSIT",
        "TERMINAL_LINEAR",
        "ATTACHED_CARRY",
        "CSPACE",
        "DYNAMIC_REPLAN",
    }
    assert {item.name for item in CollisionPolicy} == {
        "WORLD_TRANSIT",
        "TARGET_APPROACH",
        "ATTACHED_CARRY",
        "PLACEMENT_DESCENT",
        "RETREAT",
        "PASSTHROUGH",
    }
    assert {item.name for item in CommandStatus} == {
        "IDLE",
        "ACTIVE",
        "COMPLETED",
        "PLAN_FAILED",
        "TRACKING_FAILED",
        "SCENE_FAILED",
        "CANCELLED",
    }

    common = dict(
        phase_id="pick.pregrasp",
        completion_policy="pose_tolerance",
        replan_policy="dynamic_scene",
        collision_policy=CollisionPolicy.TARGET_APPROACH,
        active_target="apple",
        support="table",
        profile=PlanningProfile.TERMINAL_LINEAR,
        preplanned_trajectory="cached-path",
    )
    commands = [
        PoseCommand(target_position=[1, 2, 3], target_orientation=[1, 0, 0, 0], **common),
        JointCommand(target_positions=[0.1], **common),
        GripperCommand(gripper_action="close", **common),
        SceneCommand(operation="update_world", **common),
        HoldCommand(**common),
    ]
    assert [command.kind.value for command in commands] == [
        "pose",
        "joint",
        "gripper",
        "scene",
        "hold",
    ]
    assert all(command.phase_id == "pick.pregrasp" for command in commands)

    assert isinstance(PosePlanRequest(goal="g"), PosePlanRequest)
    assert isinstance(BatchPosePlanRequest(goals=["g"]), BatchPosePlanRequest)
    assert CspacePlanRequest(goal_positions=[0]).profile == PlanningProfile.CSPACE
    assert isinstance(PlanResult(success=True), PlanResult)
    normalized = BatchPlanResult(success=[True], trajectories=["path"])
    assert isinstance(normalized, BatchPlanResult)
    assert normalized.success_mask == (True,)
    assert isinstance(normalized.trajectories[0], JointTrajectory)
    assert not hasattr(normalized, "raw")


def test_controller_runtime_propagates_phase_metadata_into_native_requests(monkeypatch):
    class _Planner:
        def __init__(self):
            self.pose_request = None
            self.cspace_request = None
            self.joint_names = ["joint_0"]
            self.scene_revision = 0

        def ensure_planner(self):
            return self

        def plan_pose(self, request):
            self.pose_request = request
            return request

        def plan_cspace(self, request):
            self.cspace_request = request
            return request

    runtime = object.__new__(MotionPlannerRuntime)
    runtime.max_plan_attempts = 3
    runtime.batch_max_attempts = 2
    runtime.single_graph_attempt = 1
    runtime.batch_graph_attempt = 1
    runtime.robot_port = type(
        "RobotPort",
        (),
        {
            "robot": object(),
            "planner_names": ["joint_0"],
            "tensor_args": type("TensorArgs", (), {"to_device": staticmethod(lambda value: value)})(),
        },
    )()
    runtime._goal_tool_pose = lambda position, orientation, batch_size=1: (
        position,
        orientation,
        batch_size,
    )
    runtime.arm_joint_state = lambda state, repeat=1: state
    runtime.planner_runtime = _Planner()

    class _JointState:
        @classmethod
        def from_position(cls, positions, joint_names):
            return (positions, tuple(joint_names))

    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)

    command = MotionPhaseCommand(
        phase=MotionPhase.TERMINAL_GRASP_APPROACH,
        active_target="apple",
        support="table",
        completion_policy="pose_tolerance",
        replan_policy="dynamic_scene",
        phase_id="pick.approach",
    )
    pose_result = runtime.plan_pose(
        [0.1, 0.2, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        start_state="pose-state",
        request_metadata=command.planning_request_metadata,
    )
    pose_request = runtime.planner_runtime.pose_request
    assert pose_result is pose_request
    assert pose_request.phase_id == "pick.approach"
    assert pose_request.active_target == "apple"
    assert pose_request.support == "table"
    assert pose_request.collision_policy is CollisionPolicy.TARGET_APPROACH
    assert pose_request.profile is PlanningProfile.TERMINAL_LINEAR
    assert pose_request.completion_policy == "pose_tolerance"
    assert pose_request.replan_policy == "dynamic_scene"
    assert pose_request.world_revision == runtime.planner_runtime.scene_revision == 0
    assert pose_request.metadata["world_revision"] == 0

    joint_command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        joint_target=[0.2],
        phase_id="home.cspace",
        completion_policy="joint_tolerance",
        replan_policy="forbidden",
    )
    cspace_result = runtime.plan_cspace(
        [0.2],
        start_state="joint-state",
        request_metadata=joint_command.planning_request_metadata,
    )
    cspace_request = runtime.planner_runtime.cspace_request
    assert cspace_result is cspace_request
    assert cspace_request.phase_id == "home.cspace"
    assert cspace_request.profile is PlanningProfile.CSPACE
    assert cspace_request.collision_policy is CollisionPolicy.WORLD_TRANSIT
    assert cspace_request.completion_policy == "joint_tolerance"
    assert cspace_request.replan_policy == "forbidden"


def test_motion_runtime_maps_single_and_batch_kwargs_to_curobo_v2(monkeypatch):
    """Exact V2 fakes reject the removed ``num_ik_seeds`` call parameter."""

    class _ExactSingle:
        def __init__(self):
            self.pose_calls = []
            self.cspace_calls = []
            self.joint_names = ["joint_0"]

        def plan_pose(
            self,
            goal,
            current_state,
            use_implicit_goal=True,
            max_attempts=5,
            enable_graph_attempt=1,
        ):
            self.pose_calls.append(
                (goal, current_state, use_implicit_goal, max_attempts, enable_graph_attempt)
            )
            return types.SimpleNamespace(success=True, path=[[0.0]])

        def plan_cspace(
            self,
            goal,
            current_state,
            max_attempts=5,
            enable_graph_attempt=1,
        ):
            self.cspace_calls.append(
                (goal, current_state, max_attempts, enable_graph_attempt)
            )
            return types.SimpleNamespace(success=True, path=[[0.0]])

    class _ExactBatch:
        def __init__(self):
            self.pose_calls = []

        def plan_pose(
            self,
            goal,
            current_state,
            use_implicit_goal=True,
            max_attempts=1,
            success_ratio=1.0,
            enable_graph_attempt=0,
        ):
            self.pose_calls.append(
                (
                    goal,
                    current_state,
                    use_implicit_goal,
                    max_attempts,
                    success_ratio,
                    enable_graph_attempt,
                )
            )
            # Return two native batch candidates with the public trajectory
            # shape [T, D].  ``goal`` is a request payload tuple in this fake,
            # not a joint trajectory, so embedding it here would exercise an
            # invalid path shape rather than the kwargs contract under test.
            return types.SimpleNamespace(
                success=[True, True],
                path=[[[0.0]], [[0.1]]],
            )

    single = _ExactSingle()
    batch = _ExactBatch()
    planner_runtime = PlannerRuntime(planner=single, batch_planner=batch)
    runtime = object.__new__(MotionPlannerRuntime)
    runtime.max_plan_attempts = 4
    runtime.batch_max_attempts = 2
    runtime.single_graph_attempt = 1
    runtime.batch_graph_attempt = 0
    runtime.robot_port = types.SimpleNamespace(
        robot=types.SimpleNamespace(get_joints_state=lambda: "sim-state"),
        tensor_args=types.SimpleNamespace(to_device=lambda value: value),
    )
    runtime._goal_tool_pose = lambda position, orientation, batch_size=1: (
        position,
        orientation,
        batch_size,
    )
    runtime.arm_joint_state = lambda state, repeat=1: (state, repeat)
    runtime.planner_runtime = planner_runtime

    class _JointState:
        @classmethod
        def from_position(cls, positions, joint_names):
            return types.SimpleNamespace(position=positions, joint_names=joint_names)

    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)

    runtime.plan_pose([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0], start_state="single")
    runtime.plan_pose_batch(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        start_state="batch",
        batch_size=2,
    )
    runtime.plan_cspace([0.2], start_state="cspace")

    assert single.pose_calls[0][2:] == (True, 4, 1)
    assert batch.pose_calls[0][2:] == (True, 2, 1.0, 0)
    assert single.cspace_calls[0][2:] == (4, 1)


def test_motion_runtime_normalizes_native_trajectory_endpoint_before_joint_state(monkeypatch):
    """List/NumPy/tensor paths cross the DeviceCfg before CuRobo construction."""

    class _JointState:
        def __init__(self, position, joint_names):
            self.position = position
            self.joint_names = list(joint_names)

        @classmethod
        def from_position(cls, position, joint_names):
            # A raw list reaches this constructor in CuRobo as ``position * 0``
            # and fails.  The fake deliberately asserts the runtime boundary.
            assert isinstance(position, torch.Tensor)
            return cls(position, joint_names)

        def reorder(self, joint_names):
            names = list(joint_names)
            indices = [self.joint_names.index(name) for name in names]
            return type(self)(self.position[..., indices], names)

    class _Planner:
        joint_names = ["joint_a", "joint_b"]

    class _Path:
        def __init__(self, position):
            self.position = position
            self.joint_names = ["joint_a", "joint_b"]

    class _TensorArgs:
        def __init__(self):
            self.calls = []

        def to_device(self, value):
            self.calls.append(value)
            return torch.as_tensor(value, dtype=torch.float32)

    tensor_args = _TensorArgs()
    runtime = object.__new__(MotionPlannerRuntime)
    runtime.robot_port = types.SimpleNamespace(
        tensor_args=tensor_args,
        robot=types.SimpleNamespace(get_joints_state=lambda: "sim-state"),
    )
    runtime.planner_runtime = types.SimpleNamespace(
        ensure_planner=lambda: _Planner(),
    )
    runtime.arm_joint_state = lambda _state: _JointState(
        torch.tensor([8.0, 9.0]), ["joint_a", "joint_b"]
    )

    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)

    paths = [
        _Path([[0.0, 1.0], [2.0, 3.0]]),
        _Path(np.asarray([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64)),
        _Path(torch.tensor([[8.0, 9.0], [10.0, 11.0]], dtype=torch.float64)),
    ]
    for path, expected in zip(paths, ([2.0, 3.0], [6.0, 7.0], [10.0, 11.0])):
        state = runtime.joint_state_from_path_endpoint(path)
        assert state.joint_names == ["joint_a", "joint_b"]
        assert state.position.dtype == torch.float32
        torch.testing.assert_close(state.position, torch.tensor(expected))

    wrapped = _Path([[[[12.0, 13.0], [14.0, 15.0]]]])
    wrapped_state = runtime.joint_state_from_path_endpoint(wrapped)
    torch.testing.assert_close(wrapped_state.position, torch.tensor([14.0, 15.0]))

    with pytest.raises(ValueError, match="non-singleton leading dimensions"):
        runtime.joint_state_from_path_endpoint(
            _Path(np.zeros((2, 1, 2, 2), dtype=np.float32))
        )
    with pytest.raises(ValueError, match="at most leading singleton"):
        runtime.joint_state_from_path_endpoint(
            _Path(np.zeros((1, 1, 1, 2, 2), dtype=np.float32))
        )

    batch_state = runtime.batch_start_state_from_paths([paths[0], None, paths[1]])
    assert batch_state.joint_names == ["joint_a", "joint_b"]
    torch.testing.assert_close(
        batch_state.position,
        torch.tensor([[2.0, 3.0], [8.0, 9.0], [6.0, 7.0]]),
    )
    assert len(tensor_args.calls) >= 5


def test_batch_result_selects_candidate_and_seed_before_endpoint_normalization():
    """Batch [B,S,T,D] results become one selected [T,D] path per item."""

    class _NativeTrajectory:
        joint_names = ["joint_a", "joint_b"]
        position = np.asarray(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ],
                [
                    [[8.0, 9.0], [10.0, 11.0]],
                    [[12.0, 13.0], [14.0, 15.0]],
                ],
            ],
            dtype=np.float32,
        )

    raw = types.SimpleNamespace(
        success=[[False, True], [True, False]],
        interpolated_trajectory=_NativeTrajectory(),
    )
    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))

    result = runtime._wrap_result(raw, revision=0, batch=True)

    assert result.success_mask == (True, True)
    assert result.trajectories[0].positions == [[4.0, 5.0], [6.0, 7.0]]
    assert result.trajectories[1].positions == [[8.0, 9.0], [10.0, 11.0]]


def test_native_result_trims_interpolation_tail_for_single_and_batch_paths():
    """Typed paths must drop CuRobo's fixed padded interpolation horizon."""

    class _NativeTrajectory:
        joint_names = ["joint_a"]

        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float32)

    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))
    single = runtime._wrap_result(
        types.SimpleNamespace(
            success=True,
            interpolated_trajectory=_NativeTrajectory(
                [[[[0.0], [1.0], [2.0], [3.0], [4.0]]]]
            ),
            interpolated_last_tstep=3,
        ),
        revision=0,
    )
    assert single.trajectory.positions == [[0.0], [1.0], [2.0]]

    single_batch_axis = runtime._wrap_result(
        types.SimpleNamespace(
            success=[True],
            interpolated_trajectory=_NativeTrajectory(
                [[[5.0], [6.0], [7.0], [8.0], [9.0]]]
            ),
            interpolated_last_tstep=[3],
        ),
        revision=0,
    )
    assert single_batch_axis.trajectory.positions == [[5.0], [6.0], [7.0]]

    batch = runtime._wrap_result(
        types.SimpleNamespace(
            success=[[False, True], [True, False]],
            interpolated_trajectory=_NativeTrajectory(
                [
                    [
                        [[0.0], [1.0], [2.0], [3.0], [4.0]],
                        [[10.0], [11.0], [12.0], [13.0], [14.0]],
                    ],
                    [
                        [[20.0], [21.0], [22.0], [23.0], [24.0]],
                        [[30.0], [31.0], [32.0], [33.0], [34.0]],
                    ],
                ]
            ),
            interpolated_last_tstep=[[2, 4], [3, 1]],
        ),
        revision=0,
        batch=True,
    )
    assert batch.trajectories[0].positions == [[10.0], [11.0], [12.0], [13.0]]
    assert batch.trajectories[1].positions == [[20.0], [21.0], [22.0]]
