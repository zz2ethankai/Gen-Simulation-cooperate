"""Focused measured-state tests for the navigation/manipulation barrier."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.mobile.navigation_settle import (  # noqa: E402
    NavigationBaseState,
    NavigationSettleBarrier,
    NavigationSettlePort,
    NavigationSettleStatus,
)


class _FakeSettleSource:
    def __init__(self, samples):
        self.samples = list(samples)
        self.stop_count = 0
        self.finalize_count = 0

    def measure(self):
        if not self.samples:
            raise AssertionError("fake source ran out of measured samples")
        return self.samples.pop(0)

    def stop(self):
        self.stop_count += 1

    def finalize(self):
        self.finalize_count += 1


def _sample(x, *, yaw=0.0, vx=0.0, vy=0.0, wz=0.0):
    return NavigationBaseState((x, 0.0, 0.0), yaw, (vx, vy, wz))


def test_navigation_settle_waits_through_pose_drift_then_dwell():
    source = _FakeSettleSource(
        [
            _sample(0.0),
            # Twist is already small, but measured pose drift must break the
            # dwell instead of releasing the dependent arm phase.
            _sample(0.01),
            _sample(0.01),
            _sample(0.01),
        ]
    )
    barrier = NavigationSettleBarrier(
        NavigationSettlePort(source.measure, source.stop, source.finalize),
        consecutive_steps=2,
        timeout_sec=1.0,
    )

    assert barrier.step(now_sec=0.0).status == NavigationSettleStatus.WAITING
    drift = barrier.step(now_sec=0.01)
    assert drift.status == NavigationSettleStatus.WAITING
    assert drift.stable_steps == 0
    assert np.isclose(drift.pose_delta_m, 0.01)
    assert barrier.step(now_sec=0.02).status == NavigationSettleStatus.WAITING
    settled = barrier.step(now_sec=0.03)

    assert settled.status == NavigationSettleStatus.SETTLED
    assert settled.complete and settled.success
    assert source.stop_count == 4


def test_navigation_settle_reports_timeout_when_base_keeps_drifting():
    source = _FakeSettleSource([_sample(0.0), _sample(0.01), _sample(0.02)])
    barrier = NavigationSettleBarrier(
        NavigationSettlePort(source.measure, source.stop, source.finalize),
        consecutive_steps=4,
        timeout_sec=0.02,
    )

    assert barrier.step(now_sec=0.0).status == NavigationSettleStatus.WAITING
    assert barrier.step(now_sec=0.01).status == NavigationSettleStatus.WAITING
    result = barrier.step(now_sec=0.02)

    assert result.status == NavigationSettleStatus.TIMED_OUT
    assert result.complete and not result.success
    assert result.reason == "settle_timeout"
    assert source.stop_count == 3
