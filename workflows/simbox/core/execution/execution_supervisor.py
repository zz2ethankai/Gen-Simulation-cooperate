"""Hold/replan/abort orchestration shared by SimBox execution loops."""

from __future__ import annotations

import logging
from typing import Any

from core.execution.safety_monitor import SafetyDecision, SafetyMeasurements, SafetyMonitor
from core.planning.motion_command import MotionPhaseCommand


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
    def controller_key(runtime) -> tuple[str, str]:
        return str(runtime.name), str(runtime.arm_name)

    @staticmethod
    def _execution_status(runtime):
        """Read the detailed execution snapshot from the typed runtime."""

        return runtime.execution.execution_status()

    @staticmethod
    def _status_value(status, name, default=None):
        if isinstance(status, dict):
            return status.get(name, default)
        return getattr(status, name, default)

    @classmethod
    def _phase_name(cls, status, command) -> str:
        phase = cls._status_value(status, "phase", None)
        if phase is None:
            phase = command.phase
        return str(getattr(phase, "value", phase))

    def _replan_limit(self, command: MotionPhaseCommand) -> int:
        """Return the command-scoped safety budget without changing defaults."""

        if command.candidate_replan_limit is None:
            return self.max_replans
        return max(0, int(command.candidate_replan_limit))

    @classmethod
    def _hold(cls, runtime, reason: str):
        """Emit a typed measured hold; no legacy controller fallback."""

        del reason
        return runtime.execution.hold_action()

    def is_holding(self, runtime) -> bool:
        return self.controller_key(runtime) in self.holds

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
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("ExecutionSupervisor accepts MotionPhaseCommand only")
        runtime = skill.skill_runtime
        if runtime is None:
            raise RuntimeError("ExecutionSupervisor requires a bound typed runtime")
        phase_key = (id(skill), id(command))
        replan_index = self.replan_counts.get(phase_key, 0)
        replan_limit = self._replan_limit(command)
        execution_status = self._execution_status(runtime)
        phase_name = self._phase_name(execution_status, command)
        plan_id = self._status_value(execution_status, "plan_id", None)
        if not plan_id:
            plan_id = command.plan_id
        replan_allowed = self._status_value(
            execution_status,
            "replan_allowed",
            command.replan_allowed,
        )
        decision = self.monitor.evaluate(
            measurements,
            step_id=step_id,
            robot=robot,
            arm=runtime.arm_name,
            skill=skill.__class__.__name__,
            phase=phase_name,
            world_revision=world_revision,
            plan_id=plan_id or f"{phase_name}_{id(command)}",
            replan_index=replan_index,
            replan_allowed=bool(replan_allowed) and replan_index < replan_limit,
        )
        if decision == SafetyDecision.ABORT:
            runtime.execution.clear_plan_and_hold()
            if (
                command.candidate_replan_limit is not None
                and replan_index >= replan_limit
            ):
                command.params["candidate_replan_exhausted"] = True
                command.params["candidate_replan_exhausted_at"] = int(replan_index)
                command.params["candidate_replan_limit"] = int(replan_limit)
                command.params["candidate_replan_exhausted_reason"] = (
                    "replan_limit"
                )
            self.failure_reason = self.monitor.events[-1].trigger
        elif decision == SafetyDecision.HOLD_AND_REPLAN:
            runtime.execution.clear_plan_and_hold()
            # Cached terminal paths start at the original pre-grasp endpoint;
            # recovery must plan from the measured hold state.
            command.preplanned_joint_path = None
            # A skill may own a phase-specific cached path that cannot be
            # lazily rebuilt by TemplateController.forward().  CARRY_HOME is
            # one such phase: it must keep the attached object and the same
            # goal posture while replanning from the measured hold state.
            # Give the skill a chance to restore that path before the hold
            # window starts.  A failed recovery becomes a clean safety abort;
            # it must never turn into a traceback on the next forward call.
            try:
                replan_callback = skill.replan_after_safety
            except AttributeError:
                replan_callback = None
            recovered = True
            if callable(replan_callback):
                try:
                    recovered = bool(replan_callback(command))
                except Exception as exc:  # pragma: no cover - simulator-only guard
                    recovered = False
                    LOGGER.exception(
                        "[SafetyDebug] phase recovery failed robot=%s arm=%s phase=%s: %s",
                        robot,
                        runtime.arm_name,
                        phase_name,
                        exc,
                    )
            if not recovered:
                decision = SafetyDecision.ABORT
                self.holds.pop(self.controller_key(runtime), None)
                try:
                    failure_reason = skill.failure_reason
                except AttributeError:
                    failure_reason = ""
                self.failure_reason = failure_reason or f"{phase_name}_replan_failed"
            else:
                self.replan_counts[phase_key] = replan_index + 1
                self.holds[self.controller_key(runtime)] = max(1, self.hold_steps)
        if decision != SafetyDecision.CONTINUE:
            LOGGER.warning(
                "[SafetyDebug] decision=%s robot=%s arm=%s phase=%s trigger=%s replan=%d/%d measurements=%s",
                decision.value,
                robot,
                runtime.arm_name,
                phase_name,
                self.monitor.events[-1].trigger,
                self.replan_counts.get(phase_key, replan_index),
                replan_limit,
                self.monitor.events[-1].measurements,
            )
        return decision

    def forward_or_hold(self, runtime, command: MotionPhaseCommand):
        """Execute one typed phase command, or hold during recovery.

        The command is passed explicitly rather than recovered from controller
        trajectory storage.  This matters at phase boundaries: the next DAG
        node may be ready while the controller still reports the previous
        command as active.
        """

        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("ExecutionSupervisor accepts MotionPhaseCommand only")
        key = self.controller_key(runtime)
        remaining = self.holds.get(key)
        if remaining is None:
            return runtime.execution.forward_phase_command(command)
        action = self._hold(runtime, "execution_recovery_hold")
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
