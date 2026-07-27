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


class FakeController:
    name = "split_aloha"
    lr_name = "right"

    def __init__(self):
        self.clear_count = 0
        self.hold_count = 0

    def clear_plan_and_hold(self):
        self.clear_count += 1

    def hold_action(self):
        self.hold_count += 1
        return "hold"


def _fixture(max_replans=2, hold_steps=2):
    controller = FakeController()
    skill = SimpleNamespace(controller=controller)
    command = SimpleNamespace(
        phase=SimpleNamespace(value="transit_pregrasp"),
        plan_id="plan_000",
        replan_allowed=True,
        params={"preplanned_joint_path": object()},
    )
    supervisor = ExecutionSupervisor(
        SafetyMonitor(),
        {"max_replans_per_phase": max_replans, "hold_steps_before_replan": hold_steps},
    )
    return supervisor, controller, skill, command


def test_recovery_holds_then_replans_same_command_from_measured_state():
    supervisor, controller, skill, command = _fixture()
    decision = supervisor.evaluate(
        SafetyMeasurements(dynamic_obstacle_changed=True),
        step_id=1,
        robot="split_aloha",
        skill=skill,
        command=command,
        world_revision=3,
    )
    assert decision == SafetyDecision.HOLD_AND_REPLAN
    assert controller.clear_count == 1
    assert "preplanned_joint_path" not in command.params
    assert supervisor.forward_or_hold(controller, lambda: "motion") == "hold"
    assert supervisor.forward_or_hold(controller, lambda: "motion") == "hold"
    assert supervisor.forward_or_hold(controller, lambda: "motion") == "motion"


def test_third_trigger_after_two_replans_aborts_and_clears_old_plan():
    supervisor, controller, skill, command = _fixture(max_replans=2, hold_steps=1)
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
        supervisor.forward_or_hold(controller, lambda: "motion")
    assert decisions == [
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.HOLD_AND_REPLAN,
        SafetyDecision.ABORT,
    ]
    assert controller.clear_count == 3
    assert supervisor.failure_reason == "dynamic_obstacle_changed"
    assert supervisor.monitor.events[-1].replan_index == 2
