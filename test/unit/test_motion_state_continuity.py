"""Behavior checks for measured-state starts and named trajectory execution."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.controllers.curobo.phase_execution import PhaseExecutor
from core.execution.curobo_execution import ControllerExecution
from core.planning.domain_types import JointTrajectory, PlanResult


def test_cartesian_fk_path_reorders_named_positions_in_one_native_call(monkeypatch):
    pytest.importorskip("curobo")
    from core.controllers.curobo import runtime as runtime_module

    class JointState:
        @classmethod
        def from_position(cls, position, joint_names):
            return SimpleNamespace(position=position, joint_names=tuple(joint_names))

    class NativePlanner:
        tool_frames = ["ee"]
        joint_names = ["joint_0", "joint_1"]

        def __init__(self):
            self.seen = None

        def compute_kinematics(self, state):
            self.seen = state
            return SimpleNamespace(
                tool_poses=SimpleNamespace(
                    get_link_pose=lambda _name: SimpleNamespace(
                        position=state.position[..., :3]
                    )
                )
            )

    planner = NativePlanner()
    runtime = object.__new__(runtime_module.MotionPlannerRuntime)
    runtime._planner = planner
    runtime.tensor_args = SimpleNamespace(
        to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
    )
    import curobo.types
    monkeypatch.setattr(curobo.types, "JointState", JointState)

    # The helper uses the runtime's CuRobo import directly; patch the module
    # import site so this test exercises the real named Cartesian path logic.
    result = runtime._compute_cartesian_fk_batch(
        [[1.0, 2.0], [3.0, 4.0]], ["joint_1", "joint_0"]
    )

    np.testing.assert_allclose(planner.seen.position, [[2.0, 1.0], [4.0, 3.0]])
    np.testing.assert_allclose(result, [[2.0, 1.0], [4.0, 3.0]])


def _execution(state=None):
    measured = np.asarray([0.1, 0.2, 0.3, 0.01])
    execution = ControllerExecution(
        name="test_robot",
        lr_name="left",
        robot=SimpleNamespace(
            get_joints_state=lambda: SimpleNamespace(positions=measured)
        ),
        tensor_args=SimpleNamespace(
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
        ),
        raw_js_names=["joint_0", "joint_1", "joint_2"],
        arm_indices=[0, 1, 2],
        gripper_indices=[3],
        phase_executor=PhaseExecutor(),
        execution_state=state,
    )
    execution.state.ee_trans = torch.zeros(3)
    execution.state.ee_ori = torch.zeros(4)
    execution.state.last_arm_action = np.asarray([0.7, 0.8, 0.9])
    execution.get_ee_pose = lambda: (np.zeros(3), np.zeros(4))
    execution.get_gripper_action = lambda: np.asarray([0.01])
    return execution, measured


def test_replan_uses_measured_joint_state_and_discards_stale_path():
    execution, measured = _execution()
    stale = JointTrajectory(
        positions=[[1.1, 1.2, 1.3]],
        joint_names=("joint_0", "joint_1", "joint_2"),
    )
    execution.phase_executor.install(stale)
    seen = []

    def start_state(sim_state):
        seen.append(np.asarray(sim_state.positions).copy())
        return SimpleNamespace(unsqueeze=lambda _dim: sim_state)

    execution.runtime = SimpleNamespace(
        arm_joint_state=start_state,
        plan_pose=lambda *args, **kwargs: PlanResult(success=False),
    )

    action = execution.ee_forward(np.ones(3), np.ones(4))

    np.testing.assert_allclose(seen[0], measured)
    np.testing.assert_allclose(action["arm_action"], measured[:3])
    assert execution.phase_executor.current is None
    assert execution.state.num_plan_failed == 1


def test_named_trajectory_is_reordered_once_and_consumed_in_order():
    execution, _ = _execution()
    execution.raw_js_names = ("joint_1", "joint_0", "joint_2")
    execution.phase_executor.install(
        JointTrajectory(
            positions=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            joint_names=("joint_0", "joint_1", "joint_2"),
        )
    )
    execution.state.last_arm_action = None
    execution._make_action = lambda arm, gripper: {"arm_action": np.asarray(arm)}

    actions = [execution._forward_installed_joint_path() for _ in range(2)]

    np.testing.assert_allclose(actions[0]["arm_action"], [2.0, 1.0, 3.0])
    np.testing.assert_allclose(actions[1]["arm_action"], [5.0, 4.0, 6.0])
    assert execution.phase_executor.current is None
