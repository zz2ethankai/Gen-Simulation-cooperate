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
from core.planning.domain_types import CollisionPolicy, PlanningProfile  # noqa: E402


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


def test_motion_phase_command_preserves_typed_planner_fields():
    command = MotionPhaseCommand(
        phase=MotionPhase.TERMINAL_PLACE_DESCENT,
        target_position=np.zeros(3),
        target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="apple",
        support_object="tray",
        completion_policy="contact_or_tolerance",
        replan_policy="dynamic_scene",
        collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
        profile=PlanningProfile.TERMINAL_LINEAR,
        phase_id="place.descent",
    )

    assert command.active_object == "apple"
    assert command.support_object == "tray"
    assert command.active_object == "apple"
    assert command.support_object == "tray"
    assert command.phase_id == "place.descent"
    assert command.completion_policy == "contact_or_tolerance"
    assert command.replan_policy == "dynamic_scene"
    assert command.collision_policy is CollisionPolicy.PLACEMENT_DESCENT
    assert command.profile is PlanningProfile.TERMINAL_LINEAR


def test_joint_target_is_planner_input_and_direct_payload_requires_hold():
    command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        joint_target=np.array([0.1, 0.2]),
    )
    assert command.target_joint_positions.tolist() == [0.1, 0.2]
    assert command.profile is PlanningProfile.CSPACE
    assert command.collision_policy is CollisionPolicy.WORLD_TRANSIT

    with pytest.raises(ValueError, match="direct_joint_action"):
        MotionPhaseCommand(
            phase=MotionPhase.CARRY_HOME,
            params={"direct_joint_action": np.array([0.1, 0.2])},
        )


def test_direct_joint_action_is_the_typed_execution_boundary():
    command = MotionPhaseCommand(
        phase=MotionPhase.CARRY_HOME,
        direct_joint_action=np.array([0.1, 0.2]),
        gripper_state=1.0,
    )

    assert command.is_direct
    assert command.collision_policy is CollisionPolicy.PASSTHROUGH
    np.testing.assert_allclose(command.direct_joint_action, [0.1, 0.2])
