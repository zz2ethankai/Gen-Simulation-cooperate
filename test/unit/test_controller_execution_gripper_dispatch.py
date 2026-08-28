"""Tests for the finite typed gripper dispatch map."""

from __future__ import annotations

from pathlib import Path
import pytest

from core.controllers.curobo.components import MutableExecutionState
from core.execution.curobo_execution import ControllerExecution


ROOT = Path(__file__).resolve().parents[2]


def test_gripper_actions_use_typed_dispatch_and_reject_unknown_values():
    calls = []

    class RecordingExecution(ControllerExecution):
        def open_gripper(self):
            calls.append("open")

        def close_gripper(self):
            calls.append("close")

    execution = RecordingExecution(execution_state=MutableExecutionState())

    execution._apply_gripper_action("open_gripper")
    execution._apply_gripper_action("close_gripper")
    execution._apply_gripper_action(None)

    assert calls == ["open", "close"]
    with pytest.raises(ValueError, match="unsupported typed gripper action"):
        execution._apply_gripper_action("delete_everything")


def test_controller_has_no_string_reflection_for_gripper_actions():
    source = (
        ROOT / "workflows/simbox/core/execution/curobo_execution.py"
    ).read_text(encoding="utf-8")
    assert "hasattr(self, command.gripper_action)" not in source
    assert "getattr(self, command.gripper_action)" not in source
