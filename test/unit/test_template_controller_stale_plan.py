"""Regression coverage for replacing a stale typed execution path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.curobo.components import MutableExecutionState  # noqa: E402
from core.controllers.curobo.phase_execution import PhaseExecutor  # noqa: E402
from core.execution.curobo_execution import ControllerExecution  # noqa: E402
from core.planning.domain_types import JointTrajectory, PlanResult  # noqa: E402


def test_failed_typed_replan_holds_measured_joints_not_stale_path():
    state = MutableExecutionState()
    execution = ControllerExecution(
        name="test_robot",
        lr_name="left",
        robot=SimpleNamespace(
            get_joints_state=lambda: SimpleNamespace(
                positions=np.asarray([0.1, 0.2, 0.3, 0.01])
            )
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
    stale = JointTrajectory(
        positions=[[1.1, 1.2, 1.3]],
        joint_names=("joint_0", "joint_1", "joint_2"),
    )
    execution.phase_executor.install(stale)

    execution.runtime = SimpleNamespace(
        arm_joint_state=lambda sim_state: SimpleNamespace(
            unsqueeze=lambda _dim: sim_state
        ),
        plan_pose=lambda *args, **kwargs: PlanResult(
            success=False, error="test failure"
        ),
    )

    action = execution.ee_forward(np.ones(3), np.ones(4))

    np.testing.assert_allclose(action["arm_action"], [0.1, 0.2, 0.3])
    assert execution.phase_executor.current is None
    assert execution.state.num_plan_failed == 1
