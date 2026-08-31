"""ExecutionSupervisor consumes the typed runtime command status."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.execution.execution_supervisor import ExecutionSupervisor  # noqa: E402
from core.execution.safety_monitor import SafetyDecision, SafetyMeasurements, SafetyMonitor  # noqa: E402
from core.controllers.curobo.components import MutableExecutionState  # noqa: E402
from core.controllers.curobo.phase_execution import ExecutionStatus  # noqa: E402
from core.planning.domain_types import CommandStatus  # noqa: E402
from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402


class _RuntimeOwner:
    def __init__(self, status):
        self.status = status
        self.execution_state = MutableExecutionState()
        self.clear_count = 0
        self.executions = 0
        self.name = "robot"
        self.arm_name = "right"

        self.execution = SimpleNamespace(
            execution_status=lambda _command=None: self.status,
            forward_phase_command=self._execute,
            hold_action=lambda: "hold",
            clear_plan_and_hold=self._clear,
        )

    def _execute(self, _command):
        self.executions += 1
        return "runtime-motion"

    def _clear(self):
        self.clear_count += 1

    def port(self):
        return self


def _fixture(status):
    owner = _RuntimeOwner(status)
    runtime = owner.port()
    command = MotionPhaseCommand(
        phase=MotionPhase.TRANSIT_PREGRASP,
        plan_id="old-command-id",
        replan_allowed=True,
        params={},
    )
    skill = SimpleNamespace(skill_runtime=runtime)
    return ExecutionSupervisor(SafetyMonitor()), runtime, owner, skill, command


def test_safety_event_uses_command_status_phase_and_plan_id():
    status = ExecutionStatus(
        status=CommandStatus.ACTIVE,
        phase="status-phase",
        plan_id="status-plan",
        replan_allowed=False,
    )
    supervisor, runtime, owner, skill, command = _fixture(status)

    decision = supervisor.evaluate(
        SafetyMeasurements(dynamic_obstacle_changed=True),
        step_id=2,
        robot="robot",
        skill=skill,
        command=command,
        world_revision=4,
    )

    assert decision == SafetyDecision.ABORT
    event = supervisor.monitor.events[-1]
    assert event.phase == "status-phase"
    assert event.plan_id == "status-plan"
    assert owner.clear_count == 1
    assert runtime.execution.execution_status().status is CommandStatus.ACTIVE


def test_forward_uses_typed_runtime_when_not_holding():
    status = ExecutionStatus(status=CommandStatus.ACTIVE, phase="status-phase")
    supervisor, runtime, owner, _skill, command = _fixture(status)

    assert supervisor.forward_or_hold(runtime, command) == "runtime-motion"
    assert owner.executions == 1


def test_workflow_and_supervisor_do_not_read_trajectory_storage_fields():
    workflow_source = (ROOT / "workflows/simbox_dual_workflow.py").read_text(encoding="utf-8")
    supervisor_source = (
        ROOT / "workflows/simbox/core/execution/execution_supervisor.py"
    ).read_text(encoding="utf-8")
    for source in (workflow_source, supervisor_source):
        assert "cmd_plan" not in source
        assert "cmd_idx" not in source
        assert "activate_collision_world_mode" not in source
        assert "getattr(controller, \"command_status\")" not in source
        assert "getattr(controller, 'command_status')" not in source
        assert "getattr(runtime, \"execute\")" not in source
        assert "getattr(controller, \"execute\")" not in source
        assert "getattr(controller, \"forward\")" not in source
    assert "exact_exclusions" not in workflow_source
    assert "planning_exclusions" in workflow_source
    assert "_initialize_legacy_skills" not in workflow_source
    assert "skills[0]" not in workflow_source
    assert "lambda: controller.forward" not in workflow_source
