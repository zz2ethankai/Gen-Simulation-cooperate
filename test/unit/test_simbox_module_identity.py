"""Regression tests for the single ``core.*`` runtime module namespace."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dual_workflow_does_not_double_load_core_runtime_types():
    source = (ROOT / "workflows/simbox_dual_workflow.py").read_text(encoding="utf-8")

    # Loading the same source as both ``core.*`` and
    # ``workflows.simbox.core.*`` creates distinct dataclass/Enum identities.
    # MotionPhaseCommand isinstance checks and SafetyDecision comparisons then
    # silently fail even though repr/type names look identical.
    assert "from .simbox.core" not in source
    assert "from core.planning.motion_command import MotionPhase, MotionPhaseCommand" in source
    assert "from core.execution.safety_monitor import (" in source
    assert "from core.execution.execution_supervisor import ExecutionSupervisor" in source
    assert "if command is not controller._active_phase_command:" in source
    assert "arm_velocity = velocity[controller.arm_indices]" in source
    assert "arm_velocity_rad_s=float(" in source


def test_phase_identity_uses_object_reference_not_reusable_python_id():
    source = (
        ROOT / "workflows/simbox/core/controllers/controller_phases.py"
    ).read_text(encoding="utf-8")
    assert "self._active_phase_command = command" in source
    assert "command is self._active_phase_command" in source
    assert "_active_phase_command_token" not in source
