"""Recovery-budget and same-phase replan tests for ExecutionSupervisor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.execution.execution_supervisor import ExecutionSupervisor  # noqa: E402
from core.execution.safety_monitor import (  # noqa: E402
    SafetyDecision,
    SafetyMeasurements,
    SafetyMonitor,
)
from core.controllers.curobo.phase_execution import ExecutionStatus  # noqa: E402
from core.controllers.curobo.components import MutableExecutionState  # noqa: E402
from core.controllers.curobo.skill_runtime import SkillRuntimePort  # noqa: E402
from core.planning.domain_types import CommandStatus  # noqa: E402
from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402


class FakeRuntimeOwner:
    def __init__(self):
        self.clear_count = 0
        self.hold_count = 0
        self.execution_state = MutableExecutionState()
        self.status = ExecutionStatus(
            status=CommandStatus.ACTIVE,
            phase="transit_pregrasp",
            plan_id="plan_000",
            replan_allowed=True,
        )

    def clear_plan_and_hold(self):
        self.clear_count += 1

    def hold_action(self):
        self.hold_count += 1
        return "hold"

    def execute(self, command):
        assert isinstance(command, MotionPhaseCommand)
        return "motion"

    def execution_status(self, _command=None):
        return self.status

    def port(self):
        return SkillRuntimePort(
            robot=SimpleNamespace(),
            runtime=None,
            execution_state=self.execution_state,
            arm_spec=None,
            arm_indices=[0],
            gripper_indices=[1],
            name="split_aloha",
            arm_name="right",
            ee_pose=lambda: (SimpleNamespace(), SimpleNamespace()),
            arm_base_pose=lambda: (SimpleNamespace(), SimpleNamespace()),
            compute_fk=lambda joints: joints,
            execution_status=self.execution_status,
            execute=self.execute,
            hold=self.hold_action,
            clear_plan_and_hold=self.clear_plan_and_hold,
        )


def _fixture(max_replans=2, hold_steps=2):
    owner = FakeRuntimeOwner()
    runtime = owner.port()
    skill = SimpleNamespace(skill_runtime=runtime)
    command = MotionPhaseCommand(
        phase=MotionPhase.TRANSIT_PREGRASP,
        plan_id="plan_000",
        replan_allowed=True,
        params={"preplanned_joint_path": object()},
    )
    supervisor = ExecutionSupervisor(
        SafetyMonitor(),
        {"max_replans_per_phase": max_replans, "hold_steps_before_replan": hold_steps},
    )
    return supervisor, runtime, skill, command, owner


def test_recovery_holds_then_replans_same_command_from_measured_state():
    supervisor, runtime, skill, command, owner = _fixture()
    decision = supervisor.evaluate(
        SafetyMeasurements(dynamic_obstacle_changed=True),
        step_id=1,
        robot="split_aloha",
        skill=skill,
        command=command,
        world_revision=3,
    )
    assert decision == SafetyDecision.HOLD_AND_REPLAN
    assert owner.clear_count == 1
    assert "preplanned_joint_path" not in command.params
    assert supervisor.forward_or_hold(runtime, command) == "hold"
    assert supervisor.forward_or_hold(runtime, command) == "hold"
    assert supervisor.forward_or_hold(runtime, command) == "motion"


def test_third_trigger_after_two_replans_aborts_and_clears_old_plan():
    supervisor, runtime, skill, command, owner = _fixture(max_replans=2, hold_steps=1)
    decisions = []
    for step in range(3):
        decisions.append(
            supervisor.evaluate(
                SafetyMeasurements(dynamic_obstacle_changed=True),
                step_id=step,
                robot="split_aloha",
                skill=skill,
                command=command,
                world_revision=step,
            )
        )
        supervisor.forward_or_hold(runtime, command)
    assert decisions == [
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.ABORT,
    ]
    assert owner.clear_count == 3
    assert supervisor.failure_reason == "dynamic_obstacle_changed"
    assert supervisor.monitor.events[-1].replan_index == 2


def test_phase_skill_restores_cached_path_before_carry_home_forward():
    supervisor, runtime, _, command, _owner = _fixture(hold_steps=1)
    command.phase = MotionPhase.CARRY_HOME
    restored = object()

    def replan_after_safety(current_command):
        current_command.params["preplanned_joint_path"] = restored
        return True

    skill = SimpleNamespace(
        skill_runtime=runtime,
        replan_after_safety=replan_after_safety,
    )
    decision = supervisor.evaluate(
        SafetyMeasurements(dynamic_obstacle_changed=True),
        step_id=1,
        robot="split_aloha",
        skill=skill,
        command=command,
        world_revision=3,
    )

    assert decision == SafetyDecision.HOLD_AND_REPLAN
    assert command.params["preplanned_joint_path"] is restored
    assert supervisor.forward_or_hold(runtime, command) == "hold"
    assert supervisor.forward_or_hold(runtime, command) == "motion"


def test_terminal_pick_budget_allows_fourth_candidate_then_records_exhaustion():
    supervisor, runtime, _unused_skill, command, _owner = _fixture(
        max_replans=2, hold_steps=1
    )
    command.phase = MotionPhase.TERMINAL_GRASP_APPROACH
    command.candidate_replan_limit = 3
    command.replan_policy = "terminal_candidate_fallback"
    command.params["candidate_index"] = 13
    candidates = [13, 10, 14, 176]
    attempted = []

    def replan_after_safety(current_command):
        replacement = candidates[len(attempted) + 1]
        attempted.append(replacement)
        current_command.params["candidate_index"] = replacement
        return True

    skill = SimpleNamespace(
        skill_runtime=runtime,
        replan_after_safety=replan_after_safety,
    )
    decisions = []
    for step in range(4):
        decisions.append(
            supervisor.evaluate(
                SafetyMeasurements(dynamic_obstacle_changed=True),
                step_id=step,
                robot="split_aloha",
                skill=skill,
                command=command,
                world_revision=step,
            )
        )
        if decisions[-1] == SafetyDecision.HOLD_AND_REPLAN:
            supervisor.forward_or_hold(runtime, command)

    assert decisions == [
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.ABORT,
    ]
    assert attempted == [10, 14, 176]
    assert command.params["candidate_index"] == 176
    assert command.params["candidate_replan_exhausted"] is True
    assert command.params["candidate_replan_exhausted_at"] == 3
    assert command.params["candidate_replan_exhausted_reason"] == "replan_limit"
    assert command.planning_request_metadata["candidate_replan_limit"] == 3
