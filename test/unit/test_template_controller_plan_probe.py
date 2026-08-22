"""Regression tests for the composed planning-query component."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.curobo.planning_queries import (  # noqa: E402
    ControllerPlanningQueries,
)
from core.controllers.curobo.components import ComponentPort  # noqa: E402
from core.controllers.curobo.phase_execution import PhaseExecutor  # noqa: E402
import core.controllers.curobo.planning_queries as queries_module  # noqa: E402
from core.planning.domain_types import JointTrajectory, PlanResult  # noqa: E402


class _NamedJointState:
    def __init__(self, position, velocity=None, acceleration=None, jerk=None, joint_names=None):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.joint_names = joint_names

    @classmethod
    def from_position(cls, position, joint_names):
        return cls(position, joint_names=list(joint_names))

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]

        def _reorder(value):
            return None if value is None else value[..., indices]

        return _NamedJointState(
            _reorder(self.position),
            velocity=_reorder(self.velocity),
            acceleration=_reorder(self.acceleration),
            jerk=_reorder(self.jerk),
            joint_names=list(joint_names),
        )

    def __getitem__(self, index):
        return _NamedJointState(
            self.position[index],
            velocity=None if self.velocity is None else self.velocity[index],
            acceleration=None if self.acceleration is None else self.acceleration[index],
            jerk=None if self.jerk is None else self.jerk[index],
            joint_names=list(self.joint_names),
        )


class _Plan:
    def __init__(self, endpoint):
        if isinstance(endpoint, _Plan):
            self.position = endpoint.position.clone()
            self.joint_names = list(endpoint.joint_names)
        else:
            self.position = torch.as_tensor(endpoint, dtype=torch.float32).reshape(1, -1)
            self.joint_names = ["arm_0", "arm_1"]

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]
        path = _Plan(self)
        path.position = self.position[..., indices]
        path.joint_names = list(joint_names)
        return path

    def __getitem__(self, index):
        return _NamedJointState(self.position[index], joint_names=list(self.joint_names))


class _Result(PlanResult):
    def __init__(self, success, endpoint):
        super().__init__(
            success=torch.tensor([[success]], dtype=torch.bool),
            trajectory=_Plan(endpoint),
        )


class _Controller:
    def __init__(self, success=True):
        self.arm_indices = np.array([1, 3])
        self.raw_js_names = ["arm_0", "arm_1"]
        self.batch_capability = False
        self.num_plan_failed = 4
        self.phase_executor = PhaseExecutor()
        self.runtime_plan = JointTrajectory(
            positions=[[0.0, 0.0]], joint_names=("arm_0", "arm_1")
        )
        self.phase_executor.install(self.runtime_plan)
        self.robot = SimpleNamespace(
            dof_names=["base", "arm_0", "gripper", "arm_1"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.array([9.0, 0.1, 0.02, 0.2]),
                velocities=np.zeros(4),
            ),
        )
        self.tensor_args = SimpleNamespace(to_device=lambda value: torch.as_tensor(value))
        self.success = success
        self.planning_start = None

    def _native_plan_pose(self, _ee_trans, _ee_ori, sim_js, _js_names):
        self.planning_start = np.asarray(sim_js.positions).copy()
        return _Result(self.success, [0.7, 0.8])

    @staticmethod
    def _result_path(result):
        return result.trajectory

    @staticmethod
    def _result_success(result):
        return bool(result.success)

    @staticmethod
    def _command_path(path):
        return _Plan(path.positions).reorder(["arm_0", "arm_1"])


def _single_plan_port(controller):
    return ComponentPort(
        {
            "arm_indices": controller.arm_indices,
            "robot": controller.robot,
            "_native_plan_pose": controller._native_plan_pose,
            "_result_success": controller._result_success,
            "_result_path": controller._result_path,
            "_command_path": controller._command_path,
            "tensor_args": controller.tensor_args,
        }
    )


def test_plan_probe_uses_requested_start_and_preserves_runtime_state():
    controller = _Controller(success=True)
    queries = ControllerPlanningQueries(_single_plan_port(controller))

    success, endpoint, result = queries.plan_pose_from_joint_positions(
        np.ones(3),
        np.ones(4),
        start_arm_positions=np.array([0.4, 0.5]),
    )

    assert success is True
    np.testing.assert_allclose(controller.planning_start, [9.0, 0.4, 0.02, 0.5])
    np.testing.assert_allclose(endpoint, [0.7, 0.8])
    assert result.is_success
    assert controller.num_plan_failed == 4
    assert controller.phase_executor.current is controller.runtime_plan


def test_failed_plan_probe_returns_no_endpoint_or_runtime_failure_side_effect():
    controller = _Controller(success=False)
    queries = ControllerPlanningQueries(_single_plan_port(controller))

    success, endpoint, result = queries.plan_pose_from_joint_positions(
        np.ones(3), np.ones(4)
    )

    assert success is False
    assert endpoint is None
    assert not result.is_success
    assert controller.num_plan_failed == 4
    assert controller.phase_executor.current is controller.runtime_plan


def test_plan_probe_uses_single_planner_when_batch_capability_is_enabled():
    controller = _Controller(success=True)
    controller.batch_capability = True
    queries = ControllerPlanningQueries(_single_plan_port(controller))

    success, endpoint, _result = queries.plan_pose_from_joint_positions(
        np.ones(3), np.ones(4)
    )

    assert success is True
    np.testing.assert_allclose(endpoint, [0.7, 0.8])


def test_terminal_probe_propagates_named_full_trajectory_endpoint(monkeypatch):
    class _NativePlanner:
        def __init__(self):
            self.start_state = None

    class _Controller:
        def __init__(self):
            self.cmd_js_names = [f"active_{index}" for index in range(7)]
            self._planner = _NativePlanner()
            self._result = PlanResult(success=True, trajectory="terminal-path")

        @staticmethod
        def _planner_state(state):
            return state

        def _plan_pose_from_state(
            self,
            _ee_trans,
            _ee_ori,
            start_state,
            *,
            context=None,
            request_metadata=None,
        ):
            del context, request_metadata
            self._planner.start_state = start_state
            return self._result

    full_names = [
        *(f"panda_joint{index}" for index in range(1, 8)),
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    path = _NamedJointState(
        torch.arange(9.0).reshape(1, 9),
        joint_names=full_names,
    )
    controller = _Controller()
    monkeypatch.setattr(queries_module, "JointState", _NamedJointState)

    result = ControllerPlanningQueries(
        ComponentPort(
            {
                "tensor_args": SimpleNamespace(
                    to_device=lambda value: value
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, dtype=torch.float32)
                ),
                "_planner_state": controller._planner_state,
                "_plan_pose_from_state": controller._plan_pose_from_state,
            }
        )
    ).plan_pose_from_path(
        np.zeros(3), np.ones(4), path
    )

    assert result.trajectory == "terminal-path"
    assert controller._planner.start_state.joint_names == full_names
    assert controller._planner.start_state.position.shape == (9,)
    assert torch.equal(controller._planner.start_state.position, torch.arange(9.0))


def test_batch_terminal_probe_preserves_and_reorders_named_full_endpoints(monkeypatch):
    class _NativeBatchPlanner:
        def __init__(self):
            self.start_state = None
            self.batch_size = 20

    class _Controller:
        def __init__(self):
            self.planner_names = names
            self._batch_planner = _NativeBatchPlanner()
            self.runtime = SimpleNamespace(
                batch_planner=self._batch_planner,
                ensure_batch_planner=lambda: self._batch_planner,
            )
            self.tensor_args = SimpleNamespace(to_device=lambda value: torch.as_tensor(value))
            self._result = PlanResult(success=torch.tensor([True, True]), trajectory="batch-path")

        def _planner_joint_names(self):
            return list(self.planner_names)

        def _planner_state(self, state):
            return state.reorder(self.planner_names)

        def _plan_batch_from_state(
            self, _positions, _orientations, start_state, *, batch_size=None, context=None
        ):
            del batch_size, context
            self._batch_planner.start_state = start_state
            return self._result

    names = [
        *(f"panda_joint{index}" for index in range(1, 8)),
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    reversed_names = list(reversed(names))
    first = _NamedJointState(torch.arange(9.0).reshape(1, 9), joint_names=names)
    second_values = torch.arange(9.0, 18.0).reshape(1, 9)
    second = _NamedJointState(second_values, joint_names=reversed_names)
    controller = _Controller()
    monkeypatch.setattr(queries_module, "JointState", _NamedJointState)

    result = ControllerPlanningQueries(
        ComponentPort(
            {
                "runtime": controller.runtime,
                "tensor_args": controller.tensor_args,
                "_planner_joint_names": controller._planner_joint_names,
                "_planner_state": controller._planner_state,
                "_plan_batch_from_state": controller._plan_batch_from_state,
            }
        )
    )._plan_pose_batch_from_paths(
        np.zeros((2, 3)), np.ones((2, 4)), [first, second]
    )

    assert result.trajectory == "batch-path"
    state = controller._batch_planner.start_state
    assert state.joint_names == names
    assert state.position.shape == (2, 9)
    assert torch.equal(state.position[0], torch.arange(9.0))
    assert torch.equal(state.position[1], torch.arange(9.0, 18.0).flip(0))


def test_batch_terminal_probe_rejects_unnamed_or_mismatched_endpoints(monkeypatch):
    class _NativeBatchPlanner:
        batch_size = 20

    class _Controller:
        planner_names = ["a", "b"]
        runtime = SimpleNamespace(
            batch_planner=_NativeBatchPlanner(),
        )

        def __init__(self):
            self.runtime.ensure_batch_planner = lambda: self.runtime.batch_planner
        tensor_args = SimpleNamespace(to_device=lambda value: torch.as_tensor(value))

        def _planner_joint_names(self):
            return list(self.planner_names)

        def _planner_state(self, state):
            return state.reorder(self.planner_names)

    controller = _Controller()
    monkeypatch.setattr(queries_module, "JointState", _NamedJointState)
    queries = ControllerPlanningQueries(
        ComponentPort(
            {
                "runtime": controller.runtime,
                "tensor_args": controller.tensor_args,
                "_planner_joint_names": controller._planner_joint_names,
                "_planner_state": controller._planner_state,
            }
        )
    )
    unnamed = _NamedJointState(torch.zeros(1, 2), joint_names=None)
    with pytest.raises(ValueError, match="explicit joint_names"):
        queries._plan_pose_batch_from_paths(
            np.zeros((1, 3)), np.ones((1, 4)), [unnamed]
        )

    named = _NamedJointState(torch.zeros(1, 2), joint_names=["a", "b"])
    wrong = _NamedJointState(torch.zeros(1, 3), joint_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="same named joint contract"):
        queries._plan_pose_batch_from_paths(
            np.zeros((2, 3)), np.ones((2, 4)), [named, wrong]
        )


def test_cartesian_measurement_reduces_full_path_by_explicit_active_names():
    class _Controller:
        raw_js_names = ["a", "b"]

        def __init__(self):
            self.fk_inputs = []

        def _forward_kinematic_batch(self, joint_positions, *, joint_names):
            del joint_names
            values = np.asarray(joint_positions.detach().cpu()).copy()
            self.fk_inputs.append(values)
            return np.concatenate([values, np.zeros((len(values), 1))], axis=1)

    path = _NamedJointState(
        torch.tensor([[0.0, 0.0, 0.9], [1.0, 0.2, 0.8]]),
        joint_names=["a", "b", "locked_finger"],
    )
    controller = _Controller()

    ratio, deviation = ControllerPlanningQueries(
        ComponentPort(
            {
                "raw_js_names": controller.raw_js_names,
                "tensor_args": SimpleNamespace(
                    to_device=lambda value: value
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, dtype=torch.float32)
                ),
                "_forward_kinematic_batch": controller._forward_kinematic_batch,
            }
        )
    ).measure_cartesian_path(path, np.zeros(3), np.array([1.0, 0.2, 0.0]))

    assert ratio >= 1.0
    assert deviation >= 0.0
    assert len(controller.fk_inputs) == 1
    assert controller.fk_inputs[0].shape == (2, 2)
    np.testing.assert_allclose(controller.fk_inputs[0][1], [1.0, 0.2])
