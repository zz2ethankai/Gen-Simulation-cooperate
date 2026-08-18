from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402


def test_motion_phase_command_validates_pose_and_exposes_semantics():
    command = MotionPhaseCommand(
        phase=MotionPhase.TERMINAL_GRASP_APPROACH,
        target_position=np.zeros(3),
        target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="cup",
        allow_target_finger_contact=True,
        completion_tolerance={"position_m": 0.005},
    )
    assert command.is_terminal
    assert not command.is_bookkeeping
    assert command.translation_tolerance == 0.005
    assert command.planning_epsilon == 1e-4
    assert command.planning_epsilon < command.translation_tolerance


def test_motion_phase_command_keeps_target_robot_contact_explicit():
    command = MotionPhaseCommand(
        phase=MotionPhase.DETACH_AND_SETTLE,
        active_object="apple",
        allow_target_robot_contact=True,
    )
    assert command.allow_target_robot_contact is True


def test_motion_phase_command_rejects_partial_or_invalid_pose():
    with pytest.raises(ValueError, match="provided together"):
        MotionPhaseCommand(MotionPhase.TRANSIT_PREGRASP, target_position=np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        MotionPhaseCommand(
            MotionPhase.TRANSIT_PREGRASP,
            target_position=np.zeros(2),
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
