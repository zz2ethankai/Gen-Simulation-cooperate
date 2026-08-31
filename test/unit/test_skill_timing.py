"""Offline tests for the standalone Skill Timing telemetry module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.telemetry import SkillTimingRecorder, compare_timing_summaries


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_successful_skill_and_nested_phases_are_aggregated_by_layer():
    clock = _Clock()
    timing = SkillTimingRecorder(clock=clock, retain_records=True)

    with timing.skill("pick") as skill:
        clock.advance(1.0)
        with skill.execution("control") as execution:
            clock.advance(2.0)
            with skill.curobo("curobo_plan") as curobo:
                clock.advance(3.0)
            clock.advance(1.0)
        clock.advance(2.0)

    summary = timing.summary()
    pick = summary["by_skill"]["pick"]

    assert summary["successful_skill_count"] == 1
    assert summary["failed_skill_count"] == 0
    assert pick["count"] == 1
    assert pick["total_sec"] == pytest.approx(9.0)
    assert pick["execution_sec"] == pytest.approx(3.0)
    assert pick["planner_sec"] == pytest.approx(3.0)
    assert pick["curobo_sec"] == pytest.approx(3.0)
    assert summary["by_category"]["execution"]["total_sec"] == pytest.approx(3.0)
    assert summary["by_category"]["execution"]["inclusive_total_sec"] == pytest.approx(6.0)
    assert summary["by_phase"]["control"]["total_sec"] == pytest.approx(6.0)
    assert summary["by_phase"]["control"]["exclusive_total_sec"] == pytest.approx(3.0)
    assert execution.record is not None
    assert execution.record.depth == 0
    assert execution.record.status == "success"
    assert curobo.record is not None
    assert curobo.record.depth == 1
    assert curobo.record.parent == "control"
    assert summary["completed_skill_records"][0]["phases"]
    json.dumps(summary, allow_nan=False)


def test_only_successful_skills_update_aggregates_and_exceptions_are_reraised():
    clock = _Clock()
    timing = SkillTimingRecorder(clock=clock)

    with timing.skill("pick"):
        clock.advance(1.0)

    with pytest.raises(RuntimeError, match="planner failed"):
        with timing.skill("pick"):
            with timing.planner("curobo") as planner:
                planner.record_simulation_step(0.01, count=2)
                clock.advance(2.0)
                raise RuntimeError("planner failed")

    summary = timing.summary()
    assert summary["successful_skill_count"] == 1
    assert summary["failed_skill_count"] == 1
    assert summary["by_skill"]["pick"]["count"] == 1
    assert summary["by_skill"]["pick"]["total_sec"] == pytest.approx(1.0)
    assert summary["by_category"] == {}
    assert summary["by_phase"] == {}
    assert len(summary["failure_events"]) == 1
    failure = summary["failure_events"][0]
    assert failure["event"] == "skill_failed"
    assert failure["skill_name"] == "pick"
    assert failure["error_type"] == "RuntimeError"
    assert failure["reason"] == "planner failed"
    assert failure["phases"][0]["status"] == "failed"
    assert failure["physics_steps"] == 2
    assert failure["simulated_time_sec"] == pytest.approx(0.02)
    json.dumps(summary, allow_nan=False)


def test_explicit_failure_is_recorded_without_success_statistics():
    clock = _Clock()
    timing = SkillTimingRecorder(clock=clock)

    with timing.skill("place") as skill:
        clock.advance(0.5)
        skill.fail(
            "target_out_of_bounds",
            metadata={"path": Path("/tmp/target"), "not_a_number": math.nan},
        )
        clock.advance(0.5)

    summary = timing.to_dict()
    assert summary["successful_skill_count"] == 0
    assert summary["failed_skill_count"] == 1
    assert summary["by_skill"] == {}
    assert summary["failure_events"][0]["reason"] == "target_out_of_bounds"
    assert summary["failure_events"][0]["metadata"] == {
        "path": "/tmp/target",
        "not_a_number": None,
    }
    json.dumps(summary, allow_nan=False)


def test_disabled_recorder_is_a_noop_and_does_not_call_clock():
    clock = _Clock()
    timing = SkillTimingRecorder(enabled=False, clock=clock)

    with timing.skill("disabled"):
        with timing.phase("ignored", category="planner") as phase:
            phase.record_simulation_step(0.01, count=100)
            clock.advance(10.0)

    assert clock.calls == 0
    assert timing.summary() == {
        "schema_version": 1,
        "enabled": False,
        "successful_skill_count": 0,
        "failed_skill_count": 0,
        "by_skill": {},
        "by_phase": {},
        "by_category": {},
        "planner_sec": 0.0,
        "curobo_sec": 0.0,
        "failure_events": [],
    }


def test_explicit_start_finish_and_aggregation_helpers():
    clock = _Clock()
    timing = SkillTimingRecorder(clock=clock)

    scope = timing.start_skill("home")
    clock.advance(2.5)
    scope.finish()

    assert timing.aggregate_by_skill()["home"]["count"] == 1
    assert timing.aggregate_by_skill()["home"]["total_sec"] == pytest.approx(2.5)
    assert timing.aggregate_by_phase() == {}
    assert timing.aggregate_by_category() == {}


def test_simulation_counters_percentiles_and_offline_comparison():
    before_clock = _Clock()
    after_clock = _Clock()
    before = SkillTimingRecorder(clock=before_clock, retain_records=True, max_samples=4)
    after = SkillTimingRecorder(clock=after_clock, retain_records=True, max_samples=4)

    for timing, clock, durations in (
        (before, before_clock, (1.0, 3.0)),
        (after, after_clock, (2.0, 4.0)),
    ):
        for duration in durations:
            with timing.skill("pick"):
                with timing.simulation("physics") as phase:
                    phase.record_simulation_step(0.01, count=2)
                    clock.advance(duration)

    summary = before.summary()
    assert summary["physics_steps"] == 4
    assert summary["simulated_time_sec"] == pytest.approx(0.04)
    before.record_episode_simulation_step(0.01, count=4)
    summary = before.summary()
    assert summary["episode_physics_steps"] == 4
    assert summary["episode_simulated_time_sec"] == pytest.approx(0.04)
    assert summary["by_skill"]["pick"]["physics_steps"] == 4
    assert summary["by_phase"]["physics"]["physics_steps"] == 4
    assert summary["by_category"]["simulation"]["physics_steps"] == 4
    assert summary["completed_skill_records"][0]["physics_steps"] == 2
    assert summary["by_skill"]["pick"]["p50_sec"] == pytest.approx(2.0)
    assert summary["by_skill"]["pick"]["p95_sec"] == pytest.approx(2.9)

    comparison = compare_timing_summaries(before.summary(), after.summary())
    assert comparison["metrics"]["total"]["mean_sec"]["delta"] == pytest.approx(1.0)
    assert comparison["metrics"]["total"]["p50_sec"]["ratio"] == pytest.approx(1.5)
    assert comparison["by_skill"]["pick"]["simulation"]["p50_sec"]["delta"] == pytest.approx(1.0)
    json.dumps(comparison, allow_nan=False)
