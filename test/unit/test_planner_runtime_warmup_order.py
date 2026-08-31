"""Regression tests for native planner scene materialization order."""

from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.domain_types import PlannerKind, PlannerRuntimeProfile  # noqa: E402
from core.planning.planner_runtime import PlannerRuntime  # noqa: E402


class _WarmupOrderPlanner:
    def __init__(self, events, kind):
        self.events = events
        self.kind = kind
        self.joint_names = ["joint_0"]

    def update_world(self, world):
        self.events.append(("update_world", self.kind, world))

    def warmup(self, **kwargs):
        self.events.append(("warmup", self.kind, kwargs))
        assert self.events[-2][0] == "update_world"

    def plan_pose(self, *_args, **_kwargs):
        return SimpleNamespace(success=True, path=[[0.0]])

    def destroy(self):
        pass


def test_native_warmup_follows_world_update_and_batch_stays_lazy():
    events = []

    def factory(profile=None, kind=None):
        del profile
        events.append(("construct", kind))
        return _WarmupOrderPlanner(events, kind)

    runtime = PlannerRuntime(
        PlannerRuntimeProfile(
            planner_factory=factory,
            batch_planner_factory=factory,
            warmup_config={"enable_graph": False, "num_warmup_iterations": 1},
        ),
        world={"world": 1},
    )
    runtime.ensure_planner()

    assert runtime.batch_planner is None
    assert [event[0] for event in events] == [
        "construct",
        "update_world",
        "warmup",
    ]
    assert events[1][2] == {"world": 1}

    runtime.ensure_batch_planner()
    assert [event[0] for event in events] == [
        "construct",
        "update_world",
        "warmup",
        "construct",
        "update_world",
        "warmup",
    ]
    assert events[4][2] == {"world": 1}
