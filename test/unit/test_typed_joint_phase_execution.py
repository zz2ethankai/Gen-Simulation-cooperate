"""Typed joint phases must plan through native c-space before execution."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.controllers.curobo.components import MutableExecutionState
from core.execution.curobo_execution import ControllerExecution
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.planning.domain_types import (
    CollisionPolicy,
    JointTrajectory,
    PlanResult,
    PosePlanRequest,
    PlanningProfile,
)
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.planning.planner_runtime import PlannerRuntime
from core.utils.plan_utils import sort_by_difference_js


def test_joint_phase_executes_native_cspace_path_and_propagates_command_fields():
    calls = []
    executor = ControllerExecution()
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
    executor.state.step_idx = 0
    executor.state.last_arm_action = None
    executor.state.last_commanded_arm_position = None
    executor.state.phase_plan_finished = False
    executor.state.phase_plan_failed = False
    executor.state.num_plan_failed = 0
    executor.state.num_last_cmd = 0
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
    executor._command_path = lambda path: path
    def install(path, **kwargs):
        del kwargs
        executor.phase_executor.install(path)

    executor._install_command_plan = install

    command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        joint_target=[0.2, 0.3],
        phase_id="arm.home",
        completion_policy="joint_tolerance",
        replan_policy="dynamic_scene",
    )
    executor.state.active_phase_command = command

    action = executor._forward_joint_target(command, first_step=True)

    assert calls[0][0].tolist() == [0.2, 0.3]
    assert calls[0][1]["phase_id"] == "arm.home"
    assert calls[0][1]["profile"] is PlanningProfile.CSPACE
    assert calls[0][1]["collision_policy"] is CollisionPolicy.WORLD_TRANSIT
    assert calls[0][1]["completion_policy"] == "joint_tolerance"
    assert calls[0][1]["replan_policy"] == "dynamic_scene"
    np.testing.assert_allclose(action["arm_action"], [0.2, 0.3])
    assert action["arm_action"].dtype == np.dtype(float)
    assert executor.phase_executor.index == 1
    assert executor.phase_executor.current is not None
    second_action = executor._forward_installed_joint_path()
    np.testing.assert_allclose(second_action["arm_action"], [0.4, 0.5])
    assert executor.phase_executor.current is None
    assert tensor_calls
    executor.state.phase_plan_finished = True
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
    result = runtime._normalize(
        SimpleNamespace(
            success=True,
            interpolated_trajectory=_NativeTrajectory(),
            interpolated_last_tstep=3,
        ),
        PosePlanRequest(goal="goal", start_state="state"), False,
    )
    trajectory = result.trajectory

    assert isinstance(trajectory, JointTrajectory)
    np.testing.assert_allclose(trajectory.positions, [[0.1], [0.2], [0.3]])
    with pytest.raises(ValueError, match="trajectory must be"):
        JointTrajectory(
            np.zeros((2, 1, 3, 1), dtype=np.float32),
            joint_names=("joint_0",),
        )

    execution = ControllerExecution()
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
    execution.state.step_idx = 0
    execution.state.last_arm_action = None
    execution.state.last_commanded_arm_position = None
    execution.state.phase_plan_finished = False
    execution.state.num_last_cmd = 0
    execution.get_gripper_action = lambda: np.asarray([])
    execution._make_action = lambda arm, gripper: {
        "arm_action": np.asarray(arm),
        "gripper_action": np.asarray(gripper),
    }

    actions = [execution._forward_installed_joint_path() for _ in range(3)]

    np.testing.assert_allclose([action["arm_action"][0] for action in actions], [0.1, 0.2, 0.3])
    assert execution.phase_executor.current is None
    assert execution.state.phase_plan_finished is True
