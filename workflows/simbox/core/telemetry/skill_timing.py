"""Low-overhead, best-effort timing for completed SimBox skills.

The recorder is deliberately standalone: callers can add timing around a skill
without changing the skill's return values or exception behaviour.  A skill's
wall-clock duration is recorded separately from nested phase durations.  Phase
durations are inclusive, while their category contribution is exclusive of
child phases; this keeps nested layers from double-counting category totals.

Typical usage::

    timing = SkillTimingRecorder()
    with timing.skill("pick") as pick:
        with pick.phase("simulation", category="simulation"):
            step_simulation()
        with pick.curobo("plan"):
            make_plan()

    report = timing.summary()

Only normally completed skills contribute to ``by_skill``, ``by_phase`` and
``by_category``.  Exceptions are re-raised and become entries in
``failure_events``.  Set ``enabled=False`` when a caller wants a true no-op;
the disabled path does not call the clock.
"""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import math
import numbers
import os
import operator
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


SIMULATION_CATEGORY = "simulation"
EXECUTION_CATEGORY = "execution"
PLANNER_CATEGORY = "planner"
CUROBO_CATEGORY = "curobo"
OTHER_CATEGORY = "other"

_PLANNER_CATEGORIES = frozenset({PLANNER_CATEGORY, CUROBO_CATEGORY})
_CATEGORY_ALIASES = {
    "sim": SIMULATION_CATEGORY,
    "simulate": SIMULATION_CATEGORY,
    "simulation": SIMULATION_CATEGORY,
    "exec": EXECUTION_CATEGORY,
    "execute": EXECUTION_CATEGORY,
    "execution": EXECUTION_CATEGORY,
    "run": EXECUTION_CATEGORY,
    "plan": PLANNER_CATEGORY,
    "planning": PLANNER_CATEGORY,
    "planner": PLANNER_CATEGORY,
    "curobo": CUROBO_CATEGORY,
    "cu_robo": CUROBO_CATEGORY,
    "cu-robo": CUROBO_CATEGORY,
    "other": OTHER_CATEGORY,
}


def _safe_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive for hostile __str__.
        return f"<{type(value).__name__}>"


def json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert common runtime values into values accepted by ``json.dumps``.

    This intentionally has no NumPy or Isaac import.  ``tolist``/``item`` are
    handled by duck typing so telemetry remains importable in offline tests and
    in lightweight runtime processes.
    """

    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, enum.Enum):
        return json_safe(value.value, _seen)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, numbers.Real):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return _safe_text(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (Path, os.PathLike)):
        return _safe_text(os.fspath(value))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    object_id = id(value)
    if object_id in _seen:
        return "<recursive>"

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _seen.add(object_id)
        try:
            return {
                field.name: json_safe(getattr(value, field.name), _seen)
                for field in dataclasses.fields(value)
            }
        finally:
            _seen.discard(object_id)

    if isinstance(value, Mapping):
        _seen.add(object_id)
        try:
            return {
                _safe_text(key): json_safe(item, _seen)
                for key, item in value.items()
            }
        finally:
            _seen.discard(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(object_id)
        try:
            return [json_safe(item, _seen) for item in value]
        finally:
            _seen.discard(object_id)

    # NumPy arrays/scalars, tensors, and USD vector-like values commonly expose
    # one of these methods.  Keep the conversion best-effort and non-fatal.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method(), _seen)
        except Exception:
            pass
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return json_safe(tolist_method(), _seen)
        except Exception:
            pass

    try:
        return [json_safe(item, _seen) for item in value]
    except (TypeError, ValueError, AttributeError):
        return _safe_text(value)


def _normalize_category(category: Any) -> str:
    value = _safe_text(category).strip().lower().replace(" ", "_")
    return _CATEGORY_ALIASES.get(value, value or OTHER_CATEGORY)


def _elapsed(start: float, end: float) -> float:
    """Return a finite non-negative duration even if a custom clock misbehaves."""

    try:
        duration = float(end) - float(start)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return duration if math.isfinite(duration) and duration >= 0.0 else 0.0


def _simulation_step_values(dt_sec: Any, count: Any) -> tuple[int, float]:
    """Validate one simulation-step update and return count plus total time."""

    try:
        physics_steps = operator.index(count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("simulation step count must be an integer") from exc
    try:
        dt = float(dt_sec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("simulation step dt_sec must be a finite number") from exc
    if physics_steps < 0:
        raise ValueError("simulation step count must be non-negative")
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("simulation step dt_sec must be non-negative and finite")
    simulated_time_sec = dt * physics_steps
    if not math.isfinite(simulated_time_sec):
        raise ValueError("simulation step duration must be finite")
    return physics_steps, simulated_time_sec


def _simulation_totals(phases: list["PhaseTiming"] | tuple["PhaseTiming", ...]) -> tuple[int, float]:
    """Return direct simulation counters recorded by a collection of phases."""

    physics_steps = sum(phase.physics_steps for phase in phases)
    simulated_time_sec = sum(phase.simulated_time_sec for phase in phases)
    return physics_steps, simulated_time_sec


def _percentile(samples: list[float], quantile: float) -> float:
    """Return an interpolated percentile from a bounded sample list."""

    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclasses.dataclass(frozen=True)
class PhaseTiming:
    """One completed phase within a skill."""

    name: str
    category: str
    duration_sec: float
    exclusive_duration_sec: float
    status: str = "success"
    depth: int = 0
    parent: str | None = None
    metadata: Any = None
    physics_steps: int = 0
    simulated_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class SkillFailureEvent:
    """Failure information retained without affecting the runtime error path."""

    skill_name: str
    reason: str
    duration_sec: float
    error_type: str | None = None
    phases: tuple[PhaseTiming, ...] = ()
    metadata: Any = None
    physics_steps: int = 0
    simulated_time_sec: float = 0.0

    @property
    def event(self) -> str:
        return "skill_failed"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "event": self.event,
                "skill_name": self.skill_name,
                "reason": self.reason,
                "duration_sec": self.duration_sec,
                "error_type": self.error_type,
                "phases": [phase.to_dict() for phase in self.phases],
                "metadata": self.metadata,
                "physics_steps": self.physics_steps,
                "simulated_time_sec": self.simulated_time_sec,
            }
        )


# Short public name for callers that already use the generic event vocabulary.
FailureEvent = SkillFailureEvent


class _DurationAggregate:
    __slots__ = (
        "_alternate_label",
        "count",
        "total_sec",
        "min_sec",
        "max_sec",
        "_alternate_total_sec",
        "_alternate_min_sec",
        "_alternate_max_sec",
        "_sample_limit",
        "_samples",
        "_alternate_samples",
        "physics_steps",
        "simulated_time_sec",
    )

    def __init__(self, alternate_label: str | None = None, sample_limit: int = 128):
        self._alternate_label = alternate_label
        self._sample_limit = max(0, int(sample_limit))
        self.count = 0
        self.total_sec = 0.0
        self.min_sec: float | None = None
        self.max_sec: float | None = None
        self._alternate_total_sec = 0.0
        self._alternate_min_sec: float | None = None
        self._alternate_max_sec: float | None = None
        self._samples: list[float] = []
        self._alternate_samples: list[float] = []
        self.physics_steps = 0
        self.simulated_time_sec = 0.0

    def observe(
        self,
        duration_sec: float,
        alternate_sec: float | None = None,
        *,
        physics_steps: int = 0,
        simulated_time_sec: float = 0.0,
    ) -> None:
        duration = max(0.0, float(duration_sec))
        self.count += 1
        self.total_sec += duration
        self.min_sec = duration if self.min_sec is None else min(self.min_sec, duration)
        self.max_sec = duration if self.max_sec is None else max(self.max_sec, duration)
        if len(self._samples) < self._sample_limit:
            self._samples.append(duration)
        try:
            step_count = operator.index(physics_steps)
        except (TypeError, ValueError, OverflowError):
            step_count = 0
        self.physics_steps += max(0, step_count)
        try:
            simulated = float(simulated_time_sec)
        except (TypeError, ValueError, OverflowError):
            simulated = 0.0
        if math.isfinite(simulated) and simulated >= 0.0:
            self.simulated_time_sec += simulated
        if self._alternate_label is not None:
            alternate = duration if alternate_sec is None else max(0.0, float(alternate_sec))
            self._alternate_total_sec += alternate
            self._alternate_min_sec = (
                alternate
                if self._alternate_min_sec is None
                else min(self._alternate_min_sec, alternate)
            )
            self._alternate_max_sec = (
                alternate
                if self._alternate_max_sec is None
                else max(self._alternate_max_sec, alternate)
            )
            if len(self._alternate_samples) < self._sample_limit:
                self._alternate_samples.append(alternate)

    def to_dict(self) -> dict[str, Any]:
        count = self.count
        result: dict[str, Any] = {
            "count": count,
            "total_sec": self.total_sec,
            "mean_sec": self.total_sec / count if count else 0.0,
            "min_sec": self.min_sec if self.min_sec is not None else 0.0,
            "max_sec": self.max_sec if self.max_sec is not None else 0.0,
            "p50_sec": _percentile(self._samples, 0.50),
            "p95_sec": _percentile(self._samples, 0.95),
            "physics_steps": self.physics_steps,
            "simulated_time_sec": self.simulated_time_sec,
        }
        if self._alternate_label is not None:
            label = self._alternate_label
            result.update(
                {
                    f"{label}_total_sec": self._alternate_total_sec,
                    f"{label}_mean_sec": self._alternate_total_sec / count if count else 0.0,
                    f"{label}_min_sec": (
                        self._alternate_min_sec if self._alternate_min_sec is not None else 0.0
                    ),
                    f"{label}_max_sec": (
                        self._alternate_max_sec if self._alternate_max_sec is not None else 0.0
                    ),
                    f"{label}_p50_sec": _percentile(self._alternate_samples, 0.50),
                    f"{label}_p95_sec": _percentile(self._alternate_samples, 0.95),
                }
            )
        return result


class _SkillAggregate:
    __slots__ = ("total", "phases", "categories", "_sample_limit")

    def __init__(self, sample_limit: int = 128):
        self._sample_limit = sample_limit
        self.total = _DurationAggregate(sample_limit=sample_limit)
        # Phase aggregates use inclusive duration as their primary value and
        # expose exclusive duration for nested-phase accounting.
        self.phases: dict[str, _DurationAggregate] = {}
        # Category aggregates use exclusive duration as their primary value and
        # expose inclusive duration for callers that want the layered view.
        self.categories: dict[str, _DurationAggregate] = {}

    def observe(self, duration_sec: float, phases: list[PhaseTiming]) -> None:
        physics_steps, simulated_time_sec = _simulation_totals(phases)
        self.total.observe(
            duration_sec,
            physics_steps=physics_steps,
            simulated_time_sec=simulated_time_sec,
        )
        for phase in phases:
            phase_aggregate = self.phases.setdefault(
                phase.name, _DurationAggregate("exclusive", self._sample_limit)
            )
            phase_aggregate.observe(
                phase.duration_sec,
                phase.exclusive_duration_sec,
                physics_steps=phase.physics_steps,
                simulated_time_sec=phase.simulated_time_sec,
            )
            category_aggregate = self.categories.setdefault(
                phase.category, _DurationAggregate("inclusive", self._sample_limit)
            )
            category_aggregate.observe(
                phase.exclusive_duration_sec,
                phase.duration_sec,
                physics_steps=phase.physics_steps,
                simulated_time_sec=phase.simulated_time_sec,
            )

    def to_dict(self) -> dict[str, Any]:
        phases = {name: aggregate.to_dict() for name, aggregate in self.phases.items()}
        categories = {
            name: aggregate.to_dict() for name, aggregate in self.categories.items()
        }
        planner_sec = sum(
            categories.get(name, {}).get("total_sec", 0.0)
            for name in _PLANNER_CATEGORIES
        )
        result = self.total.to_dict()
        result.update(
            {
                "simulation_sec": categories.get(SIMULATION_CATEGORY, {}).get(
                    "total_sec", 0.0
                ),
                "execution_sec": categories.get(EXECUTION_CATEGORY, {}).get(
                    "total_sec", 0.0
                ),
                "planner_sec": planner_sec,
                "planner_curobo_sec": planner_sec,
                "curobo_sec": categories.get(CUROBO_CATEGORY, {}).get("total_sec", 0.0),
                "simulation_mean_sec": categories.get(SIMULATION_CATEGORY, {}).get(
                    "mean_sec", 0.0
                ),
                "simulation_p50_sec": categories.get(SIMULATION_CATEGORY, {}).get(
                    "p50_sec", 0.0
                ),
                "execution_mean_sec": categories.get(EXECUTION_CATEGORY, {}).get(
                    "mean_sec", 0.0
                ),
                "execution_p50_sec": categories.get(EXECUTION_CATEGORY, {}).get(
                    "p50_sec", 0.0
                ),
                "planner_mean_sec": categories.get(PLANNER_CATEGORY, {}).get(
                    "mean_sec", 0.0
                ),
                "planner_p50_sec": categories.get(PLANNER_CATEGORY, {}).get(
                    "p50_sec", 0.0
                ),
                "curobo_mean_sec": categories.get(CUROBO_CATEGORY, {}).get(
                    "mean_sec", 0.0
                ),
                "curobo_p50_sec": categories.get(CUROBO_CATEGORY, {}).get(
                    "p50_sec", 0.0
                ),
                "phases": phases,
                "categories": categories,
            }
        )
        return result


class _NoopPhase:
    """Reusable no-op phase returned by disabled/unscoped recorders."""

    __slots__ = ()

    def __enter__(self) -> "_NoopPhase":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def start(self) -> "_NoopPhase":
        return self

    def finish(self, success: bool = True, **kwargs: Any) -> "_NoopPhase":
        return self

    end = finish

    def fail(self, reason: str = "phase_failed", **kwargs: Any) -> "_NoopPhase":
        return self

    def record_simulation_step(self, dt_sec: float, count: int = 1) -> "_NoopPhase":
        return self

    record_simulation_steps = record_simulation_step

    @property
    def record(self) -> None:
        return None

    @property
    def physics_steps(self) -> int:
        return 0

    @property
    def simulated_time_sec(self) -> float:
        return 0.0


_NOOP_PHASE = _NoopPhase()


class PhaseTimingScope:
    """Context manager for one nested phase of a :class:`SkillTimingScope`."""

    __slots__ = (
        "_skill",
        "name",
        "category",
        "metadata",
        "_start",
        "_finished",
        "_failed",
        "_failure_reason",
        "_failure_error_type",
        "_parent",
        "_depth",
        "_child_elapsed_sec",
        "_physics_steps",
        "_simulated_time_sec",
        "_record",
    )

    def __init__(
        self,
        skill: "SkillTimingScope",
        name: Any,
        category: Any,
        metadata: Any,
    ):
        self._skill = skill
        self.name = _safe_text(name)
        self.category = _normalize_category(category)
        self.metadata = metadata
        self._start: float | None = None
        self._finished = False
        self._failed = False
        self._failure_reason = "phase_failed"
        self._failure_error_type: str | None = None
        self._parent: PhaseTimingScope | None = None
        self._depth = 0
        self._child_elapsed_sec = 0.0
        self._physics_steps = 0
        self._simulated_time_sec = 0.0
        self._record: PhaseTiming | None = None

    def __enter__(self) -> "PhaseTimingScope":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.fail(
                reason=_safe_text(exc_value) or exc_type.__name__,
                error_type=exc_type.__name__,
            )
        self.finish(success=not self._failed)
        return False

    def start(self) -> "PhaseTimingScope":
        if self._start is not None or self._finished:
            return self
        if not self._skill._is_active:
            self._finished = True
            return self
        recorder = self._skill._recorder
        self._start = recorder._now()
        if self._skill._active_phases:
            self._parent = self._skill._active_phases[-1]
            self._depth = self._parent._depth + 1
        self._skill._active_phases.append(self)
        return self

    def fail(
        self,
        reason: str = "phase_failed",
        *,
        error: BaseException | None = None,
        error_type: str | None = None,
    ) -> "PhaseTimingScope":
        if self._finished:
            return self
        self._failed = True
        self._failure_reason = _safe_text(reason) or "phase_failed"
        if error_type is not None:
            self._failure_error_type = _safe_text(error_type)
        elif error is not None:
            self._failure_error_type = type(error).__name__
        self._skill._mark_failed(
            self._failure_reason,
            error_type=self._failure_error_type,
        )
        return self

    mark_failed = fail

    def record_simulation_step(self, dt_sec: float, count: int = 1) -> "PhaseTimingScope":
        """Record physics steps without taking another wall-clock sample.

        Invalid values are ignored in best-effort mode, matching the recorder's
        non-fatal telemetry behavior.  A strict recorder raises ``ValueError``.
        Calls on an inactive or finished phase are no-ops.
        """

        if self._start is None or self._finished or not self._skill._is_active:
            return self
        try:
            physics_steps, simulated_time_sec = _simulation_step_values(dt_sec, count)
        except Exception:
            if not self._skill._recorder.best_effort:
                raise
            return self
        self._physics_steps += physics_steps
        self._simulated_time_sec += simulated_time_sec
        return self

    record_simulation_steps = record_simulation_step

    def finish(
        self,
        success: bool = True,
        *,
        reason: str | None = None,
        error: BaseException | None = None,
    ) -> "PhaseTimingScope":
        if self._finished:
            return self
        if not success:
            self.fail(reason or self._failure_reason, error=error)
        if self._start is None:
            self._finished = True
            return self

        recorder = self._skill._recorder
        end = recorder._now()
        duration_sec = _elapsed(self._start, end)
        exclusive_sec = max(0.0, duration_sec - self._child_elapsed_sec)
        status = "failed" if self._failed else "success"
        self._record = PhaseTiming(
            name=self.name,
            category=self.category,
            duration_sec=duration_sec,
            exclusive_duration_sec=exclusive_sec,
            status=status,
            depth=self._depth,
            parent=self._parent.name if self._parent is not None else None,
            metadata=self.metadata,
            physics_steps=self._physics_steps,
            simulated_time_sec=self._simulated_time_sec,
        )
        self._finished = True
        try:
            if self._parent is not None:
                self._parent._child_elapsed_sec += duration_sec
            try:
                self._skill._active_phases.remove(self)
            except ValueError:  # A skill may have closed the phase defensively.
                pass
            self._skill._phase_records.append(self._record)
        except Exception:
            if not recorder.best_effort:
                raise
        return self

    end = finish

    @property
    def record(self) -> PhaseTiming | None:
        return self._record

    @property
    def physics_steps(self) -> int:
        return self._physics_steps

    @property
    def simulated_time_sec(self) -> float:
        return self._simulated_time_sec


class SkillTimingScope:
    """Context manager and explicit handle for one skill invocation."""

    __slots__ = (
        "_recorder",
        "name",
        "metadata",
        "_start",
        "_duration_sec",
        "_finished",
        "_is_active",
        "_failed",
        "_failure_reason",
        "_failure_error_type",
        "_failure_metadata",
        "_active_phases",
        "_phase_records",
        "_token",
    )

    def __init__(self, recorder: "SkillTimingRecorder", name: Any, metadata: Any):
        self._recorder = recorder
        self.name = _safe_text(name)
        self.metadata = metadata
        self._start: float | None = None
        self._duration_sec = 0.0
        self._finished = False
        self._is_active = False
        self._failed = False
        self._failure_reason = "skill_failed"
        self._failure_error_type: str | None = None
        self._failure_metadata: Any = None
        self._active_phases: list[PhaseTimingScope] = []
        self._phase_records: list[PhaseTiming] = []
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "SkillTimingScope":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.fail(
                reason=_safe_text(exc_value) or exc_type.__name__,
                error=exc_value,
            )
        self.finish(success=not self._failed)
        # Never suppress or replace the business exception.
        return False

    def start(self) -> "SkillTimingScope":
        if self._start is not None or self._finished:
            return self
        if not self._recorder.enabled:
            self._finished = True
            return self
        self._start = self._recorder._now()
        self._is_active = True
        self._token = self._recorder._active_scope.set(self)
        return self

    def phase(
        self,
        name: Any,
        category: Any = OTHER_CATEGORY,
        *,
        kind: Any | None = None,
        metadata: Any = None,
    ) -> PhaseTimingScope | _NoopPhase:
        if kind is not None:
            category = kind
        if not self._recorder.enabled or not self._is_active or self._finished:
            return _NOOP_PHASE
        return PhaseTimingScope(self, name, category, metadata)

    def simulation(self, name: Any = SIMULATION_CATEGORY, **kwargs: Any):
        return self.phase(name, category=SIMULATION_CATEGORY, **kwargs)

    def execution(self, name: Any = EXECUTION_CATEGORY, **kwargs: Any):
        return self.phase(name, category=EXECUTION_CATEGORY, **kwargs)

    def planner(self, name: Any = PLANNER_CATEGORY, **kwargs: Any):
        return self.phase(name, category=PLANNER_CATEGORY, **kwargs)

    def curobo(self, name: Any = CUROBO_CATEGORY, **kwargs: Any):
        return self.phase(name, category=CUROBO_CATEGORY, **kwargs)

    def start_phase(self, name: Any, category: Any = OTHER_CATEGORY, **kwargs: Any):
        return self.phase(name, category=category, **kwargs).start()

    def fail(
        self,
        reason: str = "skill_failed",
        *,
        error: BaseException | None = None,
        error_type: str | None = None,
        metadata: Any = None,
    ) -> "SkillTimingScope":
        if self._finished:
            return self
        self._mark_failed(reason, error=error, error_type=error_type, metadata=metadata)
        return self

    mark_failed = fail

    def _mark_failed(
        self,
        reason: str,
        *,
        error: BaseException | None = None,
        error_type: str | None = None,
        metadata: Any = None,
    ) -> None:
        if self._finished:
            return
        was_failed = self._failed
        self._failed = True
        if not was_failed:
            self._failure_reason = _safe_text(reason) or "skill_failed"
        if error_type is not None and self._failure_error_type is None:
            self._failure_error_type = _safe_text(error_type)
        elif error is not None and self._failure_error_type is None:
            self._failure_error_type = type(error).__name__
        if metadata is not None:
            self._failure_metadata = metadata

    def finish(
        self,
        success: bool = True,
        *,
        reason: str | None = None,
        error: BaseException | None = None,
    ) -> "SkillTimingScope":
        if self._finished:
            return self
        if self._start is None and self._recorder.enabled:
            self.start()
        if not success:
            self._mark_failed(reason or self._failure_reason, error=error)

        # Defensive cleanup makes explicit finish() safe even if a caller
        # forgot to close a phase.  Normal nested context managers do not use
        # this path because their inner __exit__ runs first.
        if self._active_phases:
            for phase in reversed(self._active_phases[:]):
                phase.finish(success=False, reason="skill_finished_before_phase")

        end = self._recorder._now() if self._start is not None else 0.0
        start = self._start if self._start is not None else end
        duration_sec = _elapsed(start, end)
        self._duration_sec = duration_sec
        self._finished = True
        self._is_active = False
        try:
            self._recorder._complete_skill(self, duration_sec)
        except Exception:
            if not self._recorder.best_effort:
                raise
        finally:
            if self._token is not None:
                try:
                    self._recorder._active_scope.reset(self._token)
                except Exception:
                    if not self._recorder.best_effort:
                        raise
                self._token = None
        return self

    end = finish

    @property
    def duration_sec(self) -> float:
        if self._start is None:
            return 0.0
        if not self._finished:
            return _elapsed(self._start, self._recorder._now())
        return self._duration_sec


class SkillTimingRecorder:
    """Aggregate successful skill and nested phase durations.

    ``best_effort=True`` makes telemetry failures non-fatal at context-manager
    boundaries.  It is useful in simulation workers where observability must
    never turn a successful episode into a runtime failure.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        clock: Callable[[], float] | None = None,
        best_effort: bool = True,
        retain_records: bool = False,
        record_failures: bool = True,
        max_samples: int = 128,
    ):
        self.enabled = bool(enabled)
        self.best_effort = bool(best_effort)
        self.retain_records = bool(retain_records)
        self.record_failures = bool(record_failures)
        try:
            self.max_samples = min(4096, max(0, operator.index(max_samples)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_samples must be a non-negative integer") from exc
        self._clock = clock or time.perf_counter
        self._active_scope: contextvars.ContextVar[SkillTimingScope | None] = (
            contextvars.ContextVar(f"skill_timing_scope_{id(self)}", default=None)
        )
        self._successful_skill_count = 0
        self._failed_skill_count = 0
        self._by_skill: dict[str, _SkillAggregate] = {}
        self._by_phase: dict[str, _DurationAggregate] = {}
        self._by_category: dict[str, _DurationAggregate] = {}
        self._failure_events: list[SkillFailureEvent] = []
        self._completed_records: list[dict[str, Any]] = []
        self._successful_skill_aggregate = _DurationAggregate(
            sample_limit=self.max_samples
        )
        # One counter per simulator step, independent of how many Skills are
        # active.  Per-Skill counters remain in the phase aggregates below.
        self._episode_physics_steps = 0
        self._episode_simulated_time_sec = 0.0

    def _now(self) -> float:
        try:
            value = float(self._clock())
            if math.isfinite(value):
                return value
        except Exception:
            if not self.best_effort:
                raise
        return time.perf_counter()

    def skill(self, name: Any, *, metadata: Any = None) -> SkillTimingScope:
        """Create a skill scope; use it as a context manager or call ``start``."""

        return SkillTimingScope(self, name, metadata)

    measure = skill

    def start_skill(self, name: Any, *, metadata: Any = None) -> SkillTimingScope:
        """Start a skill for callers using explicit ``start``/``finish`` flow."""

        return self.skill(name, metadata=metadata).start()

    def phase(
        self,
        name: Any,
        category: Any = OTHER_CATEGORY,
        *,
        kind: Any | None = None,
        metadata: Any = None,
    ):
        """Create a phase on the currently active skill, or a no-op otherwise."""

        scope = self._active_scope.get()
        if scope is None:
            return _NOOP_PHASE
        return scope.phase(name, category, kind=kind, metadata=metadata)

    def simulation(self, name: Any = SIMULATION_CATEGORY, **kwargs: Any):
        return self.phase(name, category=SIMULATION_CATEGORY, **kwargs)

    def execution(self, name: Any = EXECUTION_CATEGORY, **kwargs: Any):
        return self.phase(name, category=EXECUTION_CATEGORY, **kwargs)

    def planner(self, name: Any = PLANNER_CATEGORY, **kwargs: Any):
        return self.phase(name, category=PLANNER_CATEGORY, **kwargs)

    def curobo(self, name: Any = CUROBO_CATEGORY, **kwargs: Any):
        return self.phase(name, category=CUROBO_CATEGORY, **kwargs)

    def record_episode_simulation_step(self, dt_sec: float, count: int = 1):
        """Record unique simulator steps without attributing them to a Skill.

        A concurrent DAG can have several active Skills during one physics
        step.  This episode-level counter therefore remains the authoritative
        global simulation clock, while each Skill phase still records its own
        execution steps for latency analysis.
        """

        if not self.enabled:
            return self
        try:
            physics_steps, simulated_time_sec = _simulation_step_values(dt_sec, count)
        except Exception:
            if not self.best_effort:
                raise
            return self
        self._episode_physics_steps += physics_steps
        self._episode_simulated_time_sec += simulated_time_sec
        return self

    record_episode_physics_step = record_episode_simulation_step

    def _complete_skill(self, scope: SkillTimingScope, duration_sec: float) -> None:
        physics_steps, simulated_time_sec = _simulation_totals(scope._phase_records)
        if not scope._failed:
            self._successful_skill_count += 1
            self._successful_skill_aggregate.observe(
                duration_sec,
                physics_steps=physics_steps,
                simulated_time_sec=simulated_time_sec,
            )
            skill_aggregate = self._by_skill.setdefault(
                scope.name, _SkillAggregate(self.max_samples)
            )
            skill_aggregate.observe(duration_sec, scope._phase_records)
            for phase in scope._phase_records:
                phase_aggregate = self._by_phase.setdefault(
                    phase.name, _DurationAggregate("exclusive", self.max_samples)
                )
                phase_aggregate.observe(
                    phase.duration_sec,
                    phase.exclusive_duration_sec,
                    physics_steps=phase.physics_steps,
                    simulated_time_sec=phase.simulated_time_sec,
                )
                category_aggregate = self._by_category.setdefault(
                    phase.category, _DurationAggregate("inclusive", self.max_samples)
                )
                category_aggregate.observe(
                    phase.exclusive_duration_sec,
                    phase.duration_sec,
                    physics_steps=phase.physics_steps,
                    simulated_time_sec=phase.simulated_time_sec,
                )
            if self.retain_records:
                self._completed_records.append(
                    {
                        "skill_name": scope.name,
                        "duration_sec": duration_sec,
                        "physics_steps": physics_steps,
                        "simulated_time_sec": simulated_time_sec,
                        "phases": [phase.to_dict() for phase in scope._phase_records],
                        "metadata": json_safe(scope.metadata),
                    }
                )
            return

        self._failed_skill_count += 1
        if self.record_failures:
            self._failure_events.append(
                SkillFailureEvent(
                    skill_name=scope.name,
                    reason=scope._failure_reason,
                    duration_sec=duration_sec,
                    error_type=scope._failure_error_type,
                    phases=tuple(scope._phase_records),
                    metadata=(
                        scope._failure_metadata
                        if scope._failure_metadata is not None
                        else scope.metadata
                    ),
                    physics_steps=physics_steps,
                    simulated_time_sec=simulated_time_sec,
                )
            )

    @staticmethod
    def _planner_duration(categories: Mapping[str, Any]) -> float:
        return sum(
            float(categories.get(name, {}).get("total_sec", 0.0))
            for name in _PLANNER_CATEGORIES
        )

    def summary(self) -> dict[str, Any]:
        """Return a fresh, JSON-safe aggregate summary."""

        by_skill = {name: aggregate.to_dict() for name, aggregate in self._by_skill.items()}
        by_phase = {name: aggregate.to_dict() for name, aggregate in self._by_phase.items()}
        by_category = {
            name: aggregate.to_dict() for name, aggregate in self._by_category.items()
        }
        result: dict[str, Any] = {
            "schema_version": 1,
            "enabled": self.enabled,
            "successful_skill_count": self._successful_skill_count,
            "failed_skill_count": self._failed_skill_count,
            "by_skill": by_skill,
            "by_phase": by_phase,
            "by_category": by_category,
            "planner_sec": self._planner_duration(by_category),
            "curobo_sec": by_category.get(CUROBO_CATEGORY, {}).get("total_sec", 0.0),
            "failure_events": [event.to_dict() for event in self._failure_events],
        }
        if self.enabled:
            skill_physics_steps = sum(
                aggregate.total.physics_steps for aggregate in self._by_skill.values()
            )
            skill_simulated_time_sec = sum(
                aggregate.total.simulated_time_sec for aggregate in self._by_skill.values()
            )
            result.update(
                {
                    # Preserve the historical per-Skill totals and expose the
                    # unique episode clock explicitly for concurrent Skills.
                    "physics_steps": skill_physics_steps,
                    "simulated_time_sec": skill_simulated_time_sec,
                    "episode_physics_steps": self._episode_physics_steps,
                    "episode_simulated_time_sec": self._episode_simulated_time_sec,
                    "simulation_sec": by_category.get(SIMULATION_CATEGORY, {}).get(
                        "total_sec", 0.0
                    ),
                    "execution_sec": by_category.get(EXECUTION_CATEGORY, {}).get(
                        "total_sec", 0.0
                    ),
                    "successful_skill": self._successful_skill_aggregate.to_dict(),
                }
            )
        if self.retain_records:
            result["completed_skill_records"] = json_safe(self._completed_records)
        return json_safe(result)

    to_dict = summary

    def aggregate_by_skill(self) -> dict[str, dict[str, Any]]:
        return self.summary()["by_skill"]

    def aggregate_by_phase(self) -> dict[str, dict[str, Any]]:
        return self.summary()["by_phase"]

    def aggregate_by_category(self) -> dict[str, dict[str, Any]]:
        return self.summary()["by_category"]

    @property
    def failure_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._failure_events]

    def reset(self) -> None:
        """Clear completed timing aggregates and failure events."""

        self._successful_skill_count = 0
        self._failed_skill_count = 0
        self._by_skill.clear()
        self._by_phase.clear()
        self._by_category.clear()
        self._failure_events.clear()
        self._completed_records.clear()
        self._successful_skill_aggregate = _DurationAggregate(
            sample_limit=self.max_samples
        )
        self._episode_physics_steps = 0
        self._episode_simulated_time_sec = 0.0


def _comparison_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _compare_values(before: Any, after: Any) -> dict[str, Any]:
    before_value = _comparison_number(before)
    after_value = _comparison_number(after)
    delta = after_value - before_value
    ratio = (after_value / before_value) if before_value else (1.0 if after_value == 0.0 else None)
    return {
        "before": before_value,
        "after": after_value,
        "absolute": abs(delta),
        "delta": delta,
        "ratio": ratio,
    }


def _metric_aggregate(summary: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    if metric == "total":
        aggregate = summary.get("successful_skill", {})
        if isinstance(aggregate, Mapping):
            return aggregate
        return {}
    categories = summary.get("by_category", {})
    if not isinstance(categories, Mapping):
        return {}
    aggregate = categories.get(metric, {})
    return aggregate if isinstance(aggregate, Mapping) else {}


def _skill_metric_aggregate(aggregate: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    if metric == "total":
        return aggregate
    categories = aggregate.get("categories", {})
    if not isinstance(categories, Mapping):
        return {}
    category = categories.get(metric, {})
    return category if isinstance(category, Mapping) else {}


def _compare_metric_set(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        field: _compare_values(before.get(field, 0.0), after.get(field, 0.0))
        for field in ("mean_sec", "p50_sec")
    }


def compare_timing_summaries(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare offline timing summaries without importing simulator packages.

    ``absolute`` is the absolute change, ``delta`` is ``after - before`` and
    ``ratio`` is ``after / before`` (``None`` when the baseline is zero).
    The top-level metrics compare all successful skills; ``by_skill`` provides
    the same fields for each successful skill name.
    """

    before_summary = json_safe(before)
    after_summary = json_safe(after)
    if not isinstance(before_summary, Mapping):
        before_summary = {}
    if not isinstance(after_summary, Mapping):
        after_summary = {}
    metrics = ("total", "planner", "curobo", "simulation", "execution")
    result: dict[str, Any] = {
        "schema_version": 1,
        "metrics": {
            metric: _compare_metric_set(
                _metric_aggregate(before_summary, metric),
                _metric_aggregate(after_summary, metric),
            )
            for metric in metrics
        },
        "by_skill": {},
    }
    before_skills = before_summary.get("by_skill", {})
    after_skills = after_summary.get("by_skill", {})
    if not isinstance(before_skills, Mapping):
        before_skills = {}
    if not isinstance(after_skills, Mapping):
        after_skills = {}
    for skill_name in sorted(set(before_skills) | set(after_skills), key=_safe_text):
        before_skill = before_skills.get(skill_name, {})
        after_skill = after_skills.get(skill_name, {})
        if not isinstance(before_skill, Mapping):
            before_skill = {}
        if not isinstance(after_skill, Mapping):
            after_skill = {}
        result["by_skill"][_safe_text(skill_name)] = {
            metric: _compare_metric_set(
                _skill_metric_aggregate(before_skill, metric),
                _skill_metric_aggregate(after_skill, metric),
            )
            for metric in metrics
        }
    return json_safe(result)


# Friendly aliases keep the public surface discoverable without duplicating the
# implementation or adding another wrapper layer.
SkillTiming = SkillTimingRecorder
SkillTimer = SkillTimingRecorder


__all__ = [
    "CUROBO_CATEGORY",
    "EXECUTION_CATEGORY",
    "FailureEvent",
    "OTHER_CATEGORY",
    "PLANNER_CATEGORY",
    "PhaseTiming",
    "PhaseTimingScope",
    "SIMULATION_CATEGORY",
    "SkillFailureEvent",
    "SkillTimer",
    "SkillTiming",
    "SkillTimingRecorder",
    "SkillTimingScope",
    "compare_timing_summaries",
    "json_safe",
]
