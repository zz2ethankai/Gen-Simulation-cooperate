"""Typed joint phases must plan through native c-space before execution."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.controllers.controller_component import ComponentPort
from core.controllers.controller_execution import ControllerExecution
from core.controllers.phase_executor import PhaseExecutor
from core.planning.domain_types import (
    CollisionPolicy,
    JointTrajectory,
    PlanResult,
    PlanningProfile,
)
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.planning.planner_runtime import PlannerRuntime
from core.utils.plan_utils import sort_by_difference_js


def test_joint_phase_executes_native_cspace_path_and_propagates_metadata():
    calls = []
    executor = ControllerExecution(ComponentPort({}))
    executor.phase_executor = PhaseExecutor()
    executor.collision_scene_manager = None
    executor.runtime = SimpleNamespace()
    executor.name = "robot"
    executor.lr_name = "left"
    executor.robot = SimpleNamespace(
        get_joints_state=lambda: SimpleNamespace(positions=np.asarray([0.0, 0.0]))
    )
    executor.arm_indices = np.asarray([0, 1])
    executor.gripper_indices = np.asarray([], dtype=int)
    executor.raw_js_names = ["joint_0", "joint_1"]
    tensor_calls = []
    executor.tensor_args = SimpleNamespace(
        to_device=lambda value: (
            tensor_calls.append(value)
            or torch.as_tensor(value, dtype=torch.float32)
        )
    )
    executor.ds_ratio = 1
    executor._step_idx = 0
    executor._last_arm_action = None
    executor._last_commanded_arm_position = None
    executor._phase_plan_started = False
    executor._phase_plan_finished = False
    executor._phase_plan_failed = False
    executor.num_plan_failed = 0
    executor.num_last_cmd = 0
    executor.get_gripper_action = lambda: np.asarray([])
    executor._apply_gripper_action = lambda _action: None
    executor._make_action = lambda arm, gripper: {
        "arm_action": np.asarray(arm),
        "gripper_action": np.asarray(gripper),
    }
    executor.runtime.plan_cspace = lambda target, **kwargs: (
        calls.append((np.asarray(target), kwargs))
        or PlanResult(
            success=True,
            trajectory=JointTrajectory(
                positions=[[0.2, 0.3], [0.4, 0.5]],
                joint_names=("joint_0", "joint_1"),
            ),
        )
    )
    executor._result_success = lambda result: bool(result.success)
    executor._result_path = lambda result: result.trajectory
    executor._command_path = lambda path: path
    executor._log_plan_result = lambda *args, **kwargs: None

    def install(path, **kwargs):
        del kwargs
        executor.phase_executor.install(path)
        executor._phase_plan_started = True

    executor._install_command_plan = install

    command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        joint_target=[0.2, 0.3],
        phase_id="arm.home",
        completion_policy="joint_tolerance",
        replan_policy="dynamic_scene",
    )
    executor._active_phase_command = command

    action = executor._forward_joint_target(command, first_step=True)

    assert calls[0][0].tolist() == [0.2, 0.3]
    request_metadata = calls[0][1]["request_metadata"]
    assert request_metadata["phase_id"] == "arm.home"
    assert request_metadata["profile"] is PlanningProfile.CSPACE
    assert request_metadata["collision_policy"] is CollisionPolicy.WORLD_TRANSIT
    assert request_metadata["completion_policy"] == "joint_tolerance"
    assert request_metadata["replan_policy"] == "dynamic_scene"
    np.testing.assert_allclose(action["arm_action"], [0.2, 0.3])
    assert action["arm_action"].dtype == np.dtype(float)
    assert executor.phase_executor.index == 1
    assert executor.phase_executor.current is not None
    second_action = executor._forward_installed_joint_path()
    np.testing.assert_allclose(second_action["arm_action"], [0.4, 0.5])
    assert executor.phase_executor.current is None
    assert tensor_calls
    executor._phase_plan_finished = True
    assert executor.is_phase_command_complete(command)


def test_plan_utils_sorts_named_list_trajectories_at_tensor_boundary():
    paths = [
        JointTrajectory(
            positions=[[0.0, 0.0], [1.0, 0.0]],
            joint_names=("joint_0", "joint_1"),
        ),
        JointTrajectory(
            positions=[[0.0, 0.0], [0.1, 0.0]],
            joint_names=("joint_0", "joint_1"),
        ),
    ]

    result = sort_by_difference_js(paths)

    assert result.tolist() == [1, 0]


def test_state_planning_native_fallback_returns_public_named_trajectory(monkeypatch):
    class _JointState:
        def __init__(self, position, joint_names):
            self.position = position
            self.joint_names = tuple(joint_names)

        @classmethod
        def from_position(cls, position, joint_names):
            return cls(position, joint_names)

        def reorder(self, joint_names):
            indices = [self.joint_names.index(name) for name in joint_names]
            return type(self)(self.position[..., indices], joint_names)

    fake_curobo = types.ModuleType("curobo")
    fake_curobo.__path__ = []
    fake_curobo_types = types.ModuleType("curobo.types")
    fake_curobo_types.GoalToolPose = object
    fake_curobo_types.JointState = _JointState
    monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
    monkeypatch.setitem(sys.modules, "curobo.types", fake_curobo_types)
    from core.controllers.controller_state_planning import ControllerStatePlanning

    class _NativeTrajectory:
        def __init__(self, positions, joint_names):
            self.position = torch.as_tensor(positions, dtype=torch.float32)
            self.joint_names = tuple(joint_names)

    full_native = _NativeTrajectory([[0.4], [0.5]], ("arm_0",))
    native_planner = SimpleNamespace(
        joint_names=("planner_joint",),
        kinematics=SimpleNamespace(get_full_js=lambda _state: full_native),
    )
    component = ControllerStatePlanning(
        ComponentPort(
            {
                "raw_js_names": ["arm_0"],
                "tensor_args": SimpleNamespace(
                    to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
                ),
                "runtime": SimpleNamespace(native_planner=native_planner),
            }
        )
    )

    result = component._command_path(
        _NativeTrajectory([[0.1], [0.2]], ("planner_joint",))
    )

    assert isinstance(result, JointTrajectory)
    assert result.joint_names == ("arm_0",)
    np.testing.assert_allclose(result.positions, [[0.4], [0.5]])
    assert not isinstance(result, _NativeTrajectory)


def test_phase_executor_rejects_untyped_paths():
    with pytest.raises(TypeError, match="requires JointTrajectory"):
        PhaseExecutor().install([[0.0]])


def test_native_singleton_batch_trajectory_executes_all_waypoints():
    class _NativeTrajectory:
        position = np.asarray(
            [[[[0.1], [0.2], [0.3]]]],
            dtype=np.float32,
        )
        joint_names = ("joint_0",)

    runtime = PlannerRuntime(planner=SimpleNamespace())
    result = runtime._wrap_result(
        SimpleNamespace(success=True, interpolated_trajectory=_NativeTrajectory()),
        revision=0,
    )
    trajectory = result.trajectory

    assert isinstance(trajectory, JointTrajectory)
    np.testing.assert_allclose(trajectory.positions, [[0.1], [0.2], [0.3]])
    with pytest.raises(ValueError, match="non-singleton leading"):
        JointTrajectory.from_native(
            SimpleNamespace(
                position=np.zeros((2, 1, 3, 1), dtype=np.float32),
                joint_names=("joint_0",),
            )
        )

    execution = ControllerExecution(ComponentPort({}))
    execution.phase_executor = PhaseExecutor()
    execution.phase_executor.install(trajectory)
    execution.tensor_args = SimpleNamespace(
        to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
    )
    execution.raw_js_names = ["joint_0"]
    execution.robot = SimpleNamespace(
        get_joints_state=lambda: SimpleNamespace(positions=np.asarray([0.0]))
    )
    execution.arm_indices = np.asarray([0])
    execution.gripper_indices = np.asarray([], dtype=int)
    execution.ds_ratio = 1
    execution._step_idx = 0
    execution._last_arm_action = None
    execution._last_commanded_arm_position = None
    execution._phase_plan_finished = False
    execution.num_last_cmd = 0
    execution.get_gripper_action = lambda: np.asarray([])
    execution._make_action = lambda arm, gripper: {
        "arm_action": np.asarray(arm),
        "gripper_action": np.asarray(gripper),
    }

    actions = [execution._forward_installed_joint_path() for _ in range(3)]

    np.testing.assert_allclose([action["arm_action"][0] for action in actions], [0.1, 0.2, 0.3])
    assert execution.phase_executor.current is None
    assert execution._phase_plan_finished is True
