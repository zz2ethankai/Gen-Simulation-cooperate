from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.execution.safety_monitor import (  # noqa: E402
    SafetyDecision,
    SafetyMeasurements,
    SafetyMonitor,
)


def _evaluate(monitor, measurements, step, replan_allowed=True):
    return monitor.evaluate(
        measurements,
        step_id=step,
        robot="split_aloha",
        arm="right",
        skill="pick",
        phase="transit_pregrasp",
        world_revision=4,
        plan_id="plan_000",
        replan_index=0,
        replan_allowed=replan_allowed,
    )


def test_soft_threshold_is_debounced_then_requests_replan():
    monitor = SafetyMonitor({"soft_trigger_consecutive_steps": 3})
    measurements = SafetyMeasurements(joint_error_rad=0.11)
    assert _evaluate(monitor, measurements, 0) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, measurements, 1) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, measurements, 2) == SafetyDecision.HOLD_AND_REPLAN
    assert monitor.events[-1].trigger == "joint_tracking_error"


def test_hard_threshold_and_illegal_state_abort_immediately():
    monitor = SafetyMonitor()
    assert _evaluate(monitor, SafetyMeasurements(joint_error_rad=0.26), 0) == SafetyDecision.ABORT
    assert _evaluate(monitor, SafetyMeasurements(illegal_object_state=True), 1) == SafetyDecision.ABORT


def test_dynamic_change_replans_without_debounce_and_nonreplannable_aborts():
    monitor = SafetyMonitor()
    measurements = SafetyMeasurements(dynamic_obstacle_changed=True)
    assert _evaluate(monitor, measurements, 0) == SafetyDecision.HOLD_AND_REPLAN
    assert _evaluate(monitor, measurements, 1, replan_allowed=False) == SafetyDecision.ABORT


def test_object_collision_and_attached_slip_are_hard_failures():
    monitor = SafetyMonitor()
    assert _evaluate(
        monitor, SafetyMeasurements(unexpected_object_contact_n=21.0), 0
    ) == SafetyDecision.ABORT
    assert monitor.events[-1].trigger == "unexpected_object_contact"
    assert _evaluate(
        monitor, SafetyMeasurements(attached_slip_translation_m=0.021), 1
    ) == SafetyDecision.ABORT
    assert monitor.events[-1].trigger == "attached_object_translation_slip"
    assert _evaluate(
        monitor, SafetyMeasurements(attached_slip_rotation_deg=10.1), 2
    ) == SafetyDecision.ABORT
    assert monitor.events[-1].trigger == "attached_object_rotation_slip"


def test_arm_velocity_spike_is_debounced_but_sustained_or_hard_aborts():
    monitor = SafetyMonitor(
        {
            "arm_velocity_soft_rad_s": 5.0,
            "arm_velocity_hard_rad_s": 8.0,
            "soft_trigger_consecutive_steps": 3,
        }
    )
    soft = SafetyMeasurements(arm_velocity_rad_s=5.5)
    assert _evaluate(monitor, soft, 0) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, SafetyMeasurements(arm_velocity_rad_s=0.5), 1) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, soft, 2) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, soft, 3) == SafetyDecision.CONTINUE
    assert _evaluate(monitor, soft, 4) == SafetyDecision.HOLD_AND_REPLAN
    assert monitor.events[-1].trigger == "arm_velocity"
    assert _evaluate(monitor, SafetyMeasurements(arm_velocity_rad_s=8.1), 5) == SafetyDecision.ABORT
    assert monitor.events[-1].trigger == "arm_velocity"


def test_explicit_abnormal_velocity_flag_remains_immediate_hard_failure():
    monitor = SafetyMonitor()
    assert _evaluate(
        monitor, SafetyMeasurements(abnormal_velocity=True), 0
    ) == SafetyDecision.ABORT
    assert monitor.events[-1].trigger == "abnormal_velocity"
