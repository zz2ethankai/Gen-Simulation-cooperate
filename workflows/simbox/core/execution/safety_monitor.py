"""Per-step execution tracking decisions for CuRobo manipulation phases."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SafetyDecision(str, Enum):
    CONTINUE = "continue"
    HOLD_AND_REPLAN = "hold_and_replan"
    ABORT = "abort"


@dataclass
class SafetyMeasurements:
    joint_error_rad: float = 0.0
    ee_position_error_m: float = 0.0
    ee_orientation_error_rad: float = 0.0
    base_translation_m: float = 0.0
    base_rotation_deg: float = 0.0
    unexpected_contact_n: float = 0.0
    unexpected_object_contact_n: float = 0.0
    allowed_object_support_contact_n: float = 0.0
    attached_slip_translation_m: float = 0.0
    attached_slip_rotation_deg: float = 0.0
    dynamic_obstacle_changed: bool = False
    nan_detected: bool = False
    joint_limit_violation: bool = False
    arm_velocity_rad_s: float = 0.0
    abnormal_velocity: bool = False
    illegal_object_state: bool = False
    attached_object_dropped: bool = False
    plan_failed: bool = False
    tracking_completion_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyEvent:
    step_id: int
    robot: str
    arm: str
    skill: str
    phase: str
    trigger: str
    severity: str
    measurements: dict[str, Any]
    thresholds: dict[str, Any]
    world_revision: int
    plan_id: str
    replan_index: int
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THRESHOLDS = {
    "joint_error_soft_rad": 0.10,
    "joint_error_hard_rad": 0.25,
    "ee_position_soft_m": 0.03,
    "ee_position_hard_m": 0.06,
    "ee_orientation_soft_rad": 0.15,
    "ee_orientation_hard_rad": 0.30,
    "base_translation_soft_m": 0.01,
    "base_translation_hard_m": 0.03,
    "base_rotation_soft_deg": 2.0,
    "base_rotation_hard_deg": 5.0,
    "unexpected_contact_soft_n": 5.0,
    "unexpected_contact_hard_n": 20.0,
    "arm_velocity_soft_rad_s": 50.0,
    "arm_velocity_hard_rad_s": 100.0,
    "soft_trigger_consecutive_steps": 3,
    "attached_object_slip_translation_m": 0.02,
    "attached_object_slip_rotation_deg": 10.0,
}


class SafetyMonitor:
    """Convert tracking/contact measurements into continue, hold or abort."""

    def __init__(self, config: Any | None = None):
        values = dict(DEFAULT_THRESHOLDS)
        if config is not None and hasattr(config, "get"):
            for key in values:
                values[key] = config.get(key, values[key])
        self.thresholds = values
        self.events: list[SafetyEvent] = []
        self._soft_counts: dict[tuple[str, str, str], int] = {}

    @staticmethod
    def _finite(measurements: SafetyMeasurements) -> bool:
        for value in measurements.to_dict().values():
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)):
                return False
        return True

    def _hard_trigger(self, m: SafetyMeasurements) -> str | None:
        boolean_triggers = (
            (m.nan_detected or not self._finite(m), "nan_or_non_finite"),
            (m.joint_limit_violation, "joint_limit_violation"),
            (m.abnormal_velocity, "abnormal_velocity"),
            (m.illegal_object_state, "illegal_object_state"),
            (m.attached_object_dropped, "attached_object_dropped"),
        )
        for active, name in boolean_triggers:
            if active:
                return name
        checks = (
            (m.joint_error_rad, "joint_error_hard_rad", "joint_tracking_error"),
            (m.ee_position_error_m, "ee_position_hard_m", "ee_position_tracking_error"),
            (m.ee_orientation_error_rad, "ee_orientation_hard_rad", "ee_orientation_tracking_error"),
            (m.base_translation_m, "base_translation_hard_m", "base_translation_drift"),
            (m.base_rotation_deg, "base_rotation_hard_deg", "base_rotation_drift"),
            (m.unexpected_contact_n, "unexpected_contact_hard_n", "unexpected_contact"),
            (m.arm_velocity_rad_s, "arm_velocity_hard_rad_s", "arm_velocity"),
            (
                m.unexpected_object_contact_n,
                "unexpected_contact_hard_n",
                "unexpected_object_contact",
            ),
            (
                m.attached_slip_translation_m,
                "attached_object_slip_translation_m",
                "attached_object_translation_slip",
            ),
            (
                m.attached_slip_rotation_deg,
                "attached_object_slip_rotation_deg",
                "attached_object_rotation_slip",
            ),
        )
        for value, threshold, name in checks:
            if float(value) > float(self.thresholds[threshold]):
                return name
        return None

    def _soft_triggers(self, m: SafetyMeasurements) -> list[str]:
        checks = (
            (m.joint_error_rad, "joint_error_soft_rad", "joint_tracking_error"),
            (m.ee_position_error_m, "ee_position_soft_m", "ee_position_tracking_error"),
            (m.ee_orientation_error_rad, "ee_orientation_soft_rad", "ee_orientation_tracking_error"),
            (m.base_translation_m, "base_translation_soft_m", "base_translation_drift"),
            (m.base_rotation_deg, "base_rotation_soft_deg", "base_rotation_drift"),
            (m.unexpected_contact_n, "unexpected_contact_soft_n", "unexpected_contact"),
            (m.arm_velocity_rad_s, "arm_velocity_soft_rad_s", "arm_velocity"),
            (
                m.unexpected_object_contact_n,
                "unexpected_contact_soft_n",
                "unexpected_object_contact",
            ),
        )
        result = [name for value, threshold, name in checks if float(value) > float(self.thresholds[threshold])]
        if m.dynamic_obstacle_changed:
            result.append("dynamic_obstacle_changed")
        if m.plan_failed:
            result.append("plan_failed")
        if m.tracking_completion_failed:
            result.append("tracking_completion_failed")
        return result

    def evaluate(
        self,
        measurements: SafetyMeasurements,
        *,
        step_id: int,
        robot: str,
        arm: str,
        skill: str,
        phase: str,
        world_revision: int,
        plan_id: str = "",
        replan_index: int = 0,
        replan_allowed: bool = True,
    ) -> SafetyDecision:
        hard = self._hard_trigger(measurements)
        if hard is not None:
            return self._record(
                SafetyDecision.ABORT,
                hard,
                "hard",
                measurements,
                step_id,
                robot,
                arm,
                skill,
                phase,
                world_revision,
                plan_id,
                replan_index,
            )

        active_soft = set(self._soft_triggers(measurements))
        prefix = (str(robot), str(arm))
        for key in list(self._soft_counts):
            if key[:2] == prefix and key[2] not in active_soft:
                self._soft_counts[key] = 0
        required = int(self.thresholds["soft_trigger_consecutive_steps"])
        for trigger in sorted(active_soft):
            key = (*prefix, trigger)
            self._soft_counts[key] = self._soft_counts.get(key, 0) + 1
            if trigger in {
                "dynamic_obstacle_changed",
                "plan_failed",
                "tracking_completion_failed",
            } or self._soft_counts[key] >= required:
                decision = SafetyDecision.HOLD_AND_REPLAN if replan_allowed else SafetyDecision.ABORT
                return self._record(
                    decision,
                    trigger,
                    "soft" if replan_allowed else "hard",
                    measurements,
                    step_id,
                    robot,
                    arm,
                    skill,
                    phase,
                    world_revision,
                    plan_id,
                    replan_index,
                )
        return SafetyDecision.CONTINUE

    def _record(
        self,
        decision: SafetyDecision,
        trigger: str,
        severity: str,
        measurements: SafetyMeasurements,
        step_id: int,
        robot: str,
        arm: str,
        skill: str,
        phase: str,
        world_revision: int,
        plan_id: str,
        replan_index: int,
    ) -> SafetyDecision:
        self.events.append(
            SafetyEvent(
                step_id=int(step_id),
                robot=str(robot),
                arm=str(arm),
                skill=str(skill),
                phase=str(phase),
                trigger=trigger,
                severity=severity,
                measurements=measurements.to_dict(),
                thresholds=dict(self.thresholds),
                world_revision=int(world_revision),
                plan_id=str(plan_id),
                replan_index=int(replan_index),
                decision=decision.value,
            )
        )
        return decision

    def reset(self) -> None:
        self.events.clear()
        self._soft_counts.clear()

    def export(self, episode_dir: str | Path) -> None:
        path = Path(episode_dir) / "safety_events.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def quaternion_angle(q0, q1) -> float:
    """Shortest angular distance for scalar-first quaternions."""

    first = np.asarray(q0, dtype=float)
    second = np.asarray(q1, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0.0:
        return float("inf")
    dot = float(np.clip(abs(np.dot(first, second)) / denominator, 0.0, 1.0))
    return float(2.0 * np.arccos(dot))
