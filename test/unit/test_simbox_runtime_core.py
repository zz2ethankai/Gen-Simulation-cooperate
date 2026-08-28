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
    JointTrajectory,
    PlanResult,
    PlannerKind,
    PlannerRuntimeProfile,
    PosePlanRequest,
    CollisionPolicy,
    PlannerStatus,
    PlanningProfile,
)
import core.planning.domain_types as domain_types  # noqa: E402
from core.planning.planner_runtime import (  # noqa: E402
    PlannerDestroyedError,
    PlannerRuntime,
    StaleSceneError,
)
from core.runtime import ArmSpec, JointOrderError, RobotRuntime  # noqa: E402
from core.controllers.curobo.components import MutableExecutionState  # noqa: E402
from core.controllers.curobo.runtime import MotionPlannerRuntime  # noqa: E402
from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402
from core.utils.constants import CUROBO_BATCH_SIZE  # noqa: E402


def _bind_motion_runtime(runtime, planner, batch_planner=None):
    """Bind the direct MotionPlannerRuntime owner used by object-new tests."""

    runtime._planner = planner
    runtime._batch_planner = batch_planner
    runtime._planner_factory = None
    runtime._batch_planner_factory = None
    runtime.profile = PlannerRuntimeProfile(
        name="test.motion",
        max_batch_size=CUROBO_BATCH_SIZE,
        batch_enabled=batch_planner is not None,
    )
    runtime.name = "test.motion"
    runtime._world = None
    runtime._world_set = False
    runtime._obstacle_poses = {}
    runtime._scene_revision = 0
    runtime._status = PlannerStatus.READY
    runtime._last_error = None
    runtime._planning_count = 0
    runtime._world_update_count = 0
    runtime._destroyed = False
    runtime._warmup_done = False
    runtime._warmup_kinds = set()
    runtime._obstacle_enabled = {}


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
                    "interpolated_last_tstep": [1, 1],
                },
            )()
        return type(
            "Result",
            (),
            {
                "success": True,
                "interpolated_trajectory": type(
                    "Trajectory",
                    (),
                    {"position": [[goal]], "joint_names": ["joint_0"]},
                )(),
                "interpolated_last_tstep": 1,
                "metrics": {"position_error": 0.01},
            },
        )()

    def plan_cspace(self, goal, state=None, **kwargs):
        self.calls.append(("cspace", goal, state, kwargs))
        return type(
            "Result",
            (),
            {
                "success": True,
                "interpolated_trajectory": type(
                    "Trajectory",
                    (),
                    {"position": [[goal]], "joint_names": ["joint_0"]},
                )(),
                "interpolated_last_tstep": 1,
            },
        )()

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
    assert result.trajectory.positions == [["goal"]]
    assert result.trajectory.joint_names == ("joint_0",)
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
    assert typed_batch.success == (True, False)
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


def test_planner_runtime_rejects_untyped_and_legacy_plan_inputs():
    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))
    with pytest.raises(TypeError, match="PosePlanRequest"):
        runtime.plan_pose("goal")
    with pytest.raises(TypeError, match="PosePlanRequest"):
        runtime.plan_pose({"goal": "goal"})
    with pytest.raises(AttributeError):
        runtime.plan(PosePlanRequest("goal"))


def test_planner_runtime_propagates_world_revision_to_late_batch():
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
        ),
        world={"world": 0},
    )
    runtime.update_world({"world": 1}, revision=1)
    assert runtime.world_revision == 1

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


def test_attachment_runtime_drives_one_manager_and_scene_owner():
    manager = _AttachmentManager()
    enabled = []
    scene = types.SimpleNamespace(
        set_obstacle_enabled=lambda name, value: enabled.append((name, value))
    )
    runtime = AttachmentRuntime(manager, scene=scene)
    spec = AttachmentSpec(
        "object", state="s0", meshes=["m0"],
        disable_obstacle_names=("/World/object/collider",),
    )
    runtime.attach(spec)
    assert runtime.attached_names == ("object",)
    assert enabled == [("/World/object/collider", False)]
    runtime.detach()
    assert runtime.attached_names == ()
    assert enabled[-1] == ("/World/object/collider", True)


def test_attachment_names_and_reset_are_owned_by_attachment_runtime():
    """Runtime reset uses formal attachment ports rather than façade state lists."""

    runtime = object.__new__(MotionPlannerRuntime)
    single_manager = _AttachmentManager()
    batch_manager = _AttachmentManager()
    scene = types.SimpleNamespace(set_obstacle_enabled=lambda *_args: None)
    runtime.attachment_runtime = AttachmentRuntime(single_manager, scene=scene)
    runtime.batch_attachment_runtime = AttachmentRuntime(batch_manager, scene=scene)
    spec = AttachmentSpec(
        name="object",
        state="state",
        meshes=["mesh"],
        disable_obstacle_names=("/World/object/collider",),
    )

    runtime.attach_object(spec)
    assert runtime.attachment_runtime.attached_names == ("object",)
    assert runtime.batch_attachment_runtime.attached_names == ("object",)
    assert not hasattr(runtime, "_native_attached_obstacle_names")
    assert not hasattr(runtime, "_native_batch_attached_obstacle_names")

    runtime.reset_attachments()
    assert runtime.attachment_runtime.attached_names == ()
    assert runtime.batch_attachment_runtime.attached_names == ()
    assert runtime.attachment_runtime.attached_obstacle_names == ()
    assert runtime.batch_attachment_runtime.attached_obstacle_names == ()


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


def test_public_domain_types_are_finite_and_requests_are_typed():
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

    assert isinstance(PosePlanRequest(goal="g", start_state="state"), PosePlanRequest)
    assert isinstance(BatchPosePlanRequest(goals=["g"], start_state="state", batch_size=1), BatchPosePlanRequest)
    assert CspacePlanRequest(goal_positions=[0], start_state="state").profile == PlanningProfile.CSPACE
    assert isinstance(PlanResult(success=True), PlanResult)
    normalized = BatchPlanResult(
        success=(True,),
        trajectories=(JointTrajectory(positions=[[0.0]], joint_names=("joint_0",)),),
    )
    assert isinstance(normalized, BatchPlanResult)
    assert normalized.success == (True,)
    assert isinstance(normalized.trajectories[0], JointTrajectory)
    assert not hasattr(normalized, "raw")


def test_controller_runtime_builds_one_canonical_typed_request_metadata():
    runtime = object.__new__(MotionPlannerRuntime)
    runtime._scene_revision = 0
    runtime.attachment_runtime = None
    runtime.robot_port = types.SimpleNamespace(collision_scene_manager=None)

    command = MotionPhaseCommand(
        phase=MotionPhase.TERMINAL_GRASP_APPROACH,
        active_object="apple",
        support_object="table",
        completion_policy="pose_tolerance",
        replan_policy="dynamic_scene",
        phase_id="pick.approach",
    )
    pose_request = runtime._request_common(
        phase_id=command.phase_id,
        default_profile=PlanningProfile.TRANSIT,
        collision_policy=command.collision_policy,
        active_target=command.active_object,
        support=command.support_object,
        collision_options=command.collision_options,
        profile=command.profile,
        completion_policy=command.completion_policy,
        replan_policy=command.replan_policy,
        metadata=command.metadata,
        attachment_runtime=None,
    )
    assert pose_request["phase_id"] == "pick.approach"
    assert pose_request["active_target"] == "apple"
    assert pose_request["support"] == "table"
    assert pose_request["collision_policy"] is CollisionPolicy.TARGET_APPROACH
    assert pose_request["profile"] is PlanningProfile.TERMINAL_LINEAR
    assert pose_request["completion_policy"] == "pose_tolerance"
    assert pose_request["replan_policy"] == "dynamic_scene"
    assert pose_request["world_revision"] == 0

    joint_command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        joint_target=[0.2],
        phase_id="home.cspace",
        completion_policy="joint_tolerance",
        replan_policy="forbidden",
    )
    cspace_request = runtime._request_common(
        phase_id=joint_command.phase_id,
        default_profile=PlanningProfile.CSPACE,
        collision_policy=joint_command.collision_policy,
        active_target=joint_command.active_object,
        support=joint_command.support_object,
        collision_options=joint_command.collision_options,
        profile=joint_command.profile,
        completion_policy=joint_command.completion_policy,
        replan_policy=joint_command.replan_policy,
        metadata=joint_command.metadata,
        attachment_runtime=None,
    )
    assert cspace_request["phase_id"] == "home.cspace"
    assert cspace_request["profile"] is PlanningProfile.CSPACE
    assert cspace_request["collision_policy"] is CollisionPolicy.WORLD_TRANSIT
    assert cspace_request["completion_policy"] == "joint_tolerance"
    assert cspace_request["replan_policy"] == "forbidden"


def test_motion_runtime_batches_single_cspace_goal_and_live_start(monkeypatch):
    class _JointState:
        def __init__(self, positions, joint_names):
            self.position = np.asarray(positions)
            self.joint_names = tuple(joint_names)

        @classmethod
        def from_position(cls, positions, joint_names):
            return cls(positions, joint_names)

        def unsqueeze(self, dim):
            return type(self)(np.expand_dims(self.position, dim), self.joint_names)

    class _Planner:
        scene_revision = 0
        joint_names = ["joint_0"]

        def __init__(self):
            self.request = None

        def ensure_planner(self):
            return self

        def plan_cspace(self, goal, current_state, **kwargs):
            self.request = (goal, current_state, kwargs)
            return types.SimpleNamespace(success=True, path=[[0.2]])

    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)

    planner = _Planner()
    runtime = object.__new__(MotionPlannerRuntime)
    runtime.max_plan_attempts = 4
    runtime.single_graph_attempt = 4
    runtime.robot_port = types.SimpleNamespace(
        robot=types.SimpleNamespace(get_joints_state=lambda: "sim-state"),
        tensor_args=types.SimpleNamespace(
            to_device=lambda value: np.asarray(value, dtype=float)
        ),
    )
    runtime.arm_joint_state = lambda _state: _JointState([0.1], ["joint_0"])
    _bind_motion_runtime(runtime, planner)

    request = runtime.plan_cspace([0.2])

    assert planner.request[0].position.shape == (1, 1)
    assert planner.request[1].position.shape == (1, 1)


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
    runtime = object.__new__(MotionPlannerRuntime)
    runtime.max_plan_attempts = 4
    runtime.single_graph_attempt = 1
    runtime._timing_scope = None
    runtime.attachment_runtime = None
    runtime.batch_attachment_runtime = None
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
    _bind_motion_runtime(runtime, single, batch)

    class _JointState:
        def __init__(self, positions, joint_names):
            self.position = positions
            self.joint_names = tuple(joint_names)

        @classmethod
        def from_position(cls, positions, joint_names):
            return cls(positions, joint_names)

        def unsqueeze(self, dim):
            return type(self)(np.expand_dims(self.position, dim), self.joint_names)

    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)

    runtime.plan_pose(
        [0.1, 0.2, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        start_state=_JointState([[0.0]], ["joint_0"]),
    )
    runtime.plan_pose_batch(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        start_state=_JointState([[0.0], [0.0]], ["joint_0"]),
    )
    runtime.plan_cspace(
        [0.2], start_state=_JointState([[0.0]], ["joint_0"])
    )

    assert single.pose_calls[0][2:] == (True, 4, 1)
    assert batch.pose_calls[0][2:] == (True, 4, 0.5, 1)
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
    runtime._planner = _Planner()
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
        interpolated_last_tstep=[2, 2],
    )
    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))

    result = runtime._normalize(
        raw, BatchPosePlanRequest(goals=["g0", "g1"], start_state="state", batch_size=2), True
    )

    assert result.success == (True, True)
    assert result.trajectories[0].positions == [[4.0, 5.0], [6.0, 7.0]]
    assert result.trajectories[1].positions == [[8.0, 9.0], [10.0, 11.0]]


def test_native_result_trims_interpolation_tail_for_single_and_batch_paths():
    """Typed paths must drop CuRobo's fixed padded interpolation horizon."""

    class _NativeTrajectory:
        joint_names = ["joint_a"]

        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float32)

    runtime = PlannerRuntime(planner=_Planner(), batch_planner=_Planner(batch=True))
    single = runtime._normalize(
        types.SimpleNamespace(
            success=True,
            interpolated_trajectory=_NativeTrajectory(
                [[[[0.0], [1.0], [2.0], [3.0], [4.0]]]]
            ),
            interpolated_last_tstep=3,
        ),
        PosePlanRequest(goal="goal", start_state="state"), False,
    )
    assert single.trajectory.positions == [[0.0], [1.0], [2.0]]

    single_batch_axis = runtime._normalize(
        types.SimpleNamespace(
            success=[True],
            interpolated_trajectory=_NativeTrajectory(
                [[[5.0], [6.0], [7.0], [8.0], [9.0]]]
            ),
            interpolated_last_tstep=[3],
        ),
        PosePlanRequest(goal="goal", start_state="state"), False,
    )
    assert single_batch_axis.trajectory.positions == [[5.0], [6.0], [7.0]]

    batch = runtime._normalize(
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
        BatchPosePlanRequest(goals=["g0", "g1"], start_state="state", batch_size=2), True,
    )
    assert batch.trajectories[0].positions == [[10.0], [11.0], [12.0], [13.0]]
    assert batch.trajectories[1].positions == [[20.0], [21.0], [22.0]]
