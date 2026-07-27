"""Hold/replan/abort orchestration shared by SimBox execution loops."""

from __future__ import annotations

import logging
from typing import Any, Callable

from core.execution.safety_monitor import SafetyDecision, SafetyMeasurements, SafetyMonitor


LOGGER = logging.getLogger("de_logger")


class ExecutionSupervisor:
    """Own bounded recovery state so both workflow loops behave identically."""

    def __init__(self, monitor: SafetyMonitor, config: Any | None = None):
        self.monitor = monitor
        self.config = config or {}
        self.max_replans = int(self.config.get("max_replans_per_phase", 2))
        self.hold_steps = int(self.config.get("hold_steps_before_replan", 5))
        self.holds: dict[tuple[str, str], int] = {}
        self.replan_counts: dict[tuple[int, int], int] = {}
        self.failure_reason = ""

    @staticmethod
    def controller_key(controller) -> tuple[str, str]:
        return str(controller.name), str(controller.lr_name)

    def is_holding(self, controller) -> bool:
        return self.controller_key(controller) in self.holds

    def evaluate(
        self,
        measurements: SafetyMeasurements,
        *,
        step_id: int,
        robot: str,
        skill: Any,
        command: Any,
        world_revision: int,
    ) -> SafetyDecision:
        controller = skill.controller
        phase_key = (id(skill), id(command))
        replan_index = self.replan_counts.get(phase_key, 0)
        decision = self.monitor.evaluate(
            measurements,
            step_id=step_id,
            robot=robot,
            arm=controller.lr_name,
            skill=skill.__class__.__name__,
            phase=command.phase.value,
            world_revision=world_revision,
            plan_id=command.plan_id or f"{command.phase.value}_{id(command)}",
            replan_index=replan_index,
            replan_allowed=command.replan_allowed and replan_index < self.max_replans,
        )
        if decision == SafetyDecision.ABORT:
            controller.clear_plan_and_hold()
            self.failure_reason = self.monitor.events[-1].trigger
        elif decision == SafetyDecision.HOLD_AND_REPLAN:
            controller.clear_plan_and_hold()
            # Cached terminal paths start at the original pre-grasp endpoint;
            # recovery must plan from the measured hold state.
            command.params.pop("preplanned_joint_path", None)
            self.replan_counts[phase_key] = replan_index + 1
            self.holds[self.controller_key(controller)] = max(1, self.hold_steps)
        if decision != SafetyDecision.CONTINUE:
            LOGGER.warning(
                "[SafetyDebug] decision=%s robot=%s arm=%s phase=%s trigger=%s replan=%d/%d measurements=%s",
                decision.value,
                robot,
                controller.lr_name,
                command.phase.value,
                self.monitor.events[-1].trigger,
                self.replan_counts.get(phase_key, replan_index),
                self.max_replans,
                self.monitor.events[-1].measurements,
            )
        return decision

    def forward_or_hold(self, controller, forward: Callable[[], dict]):
        key = self.controller_key(controller)
        remaining = self.holds.get(key)
        if remaining is None:
            return forward()
        action = controller.hold_action()
        remaining -= 1
        if remaining <= 0:
            self.holds.pop(key, None)
        else:
            self.holds[key] = remaining
        return action

    def reset(self) -> None:
        self.holds.clear()
        self.replan_counts.clear()
        self.failure_reason = ""
