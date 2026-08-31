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
        self.name = "split_aloha"
        self.arm_name = "right"

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
        return self


def _fixture(max_replans=2, hold_steps=2):
    owner = FakeRuntimeOwner()
    runtime = owner.port()
    skill = SimpleNamespace(skill_runtime=runtime)
    command = MotionPhaseCommand(
        phase=MotionPhase.TRANSIT_PREGRASP,
        plan_id="plan_000",
        replan_allowed=True,
        preplanned_joint_path=object(),
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
    assert command.preplanned_joint_path is None
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
        current_command.preplanned_joint_path = restored
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
    assert command.preplanned_joint_path is restored
    assert supervisor.forward_or_hold(runtime, command) == "hold"
    assert supervisor.forward_or_hold(runtime, command) == "motion"
