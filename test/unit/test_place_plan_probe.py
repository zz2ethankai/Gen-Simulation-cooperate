"""Behavior contract for the attachment-gated Place planning probe."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "workflows/simbox/core/skills/place_plan_probe.py"


class _State(str, Enum):
    WORLD_OBSTACLE = "world_obstacle"
    ACTIVE_TARGET_APPROACH = "active_target_approach"
    ATTACHED = "attached"


def _load_probe_module(monkeypatch):
    base_module = types.ModuleType("core.skills.base_skill")
    base_module.register_skill = lambda cls: cls
    place_module = types.ModuleType("core.skills.place")

    class FakePlace:
        def simple_generate_manip_cmds(self):
            self.super_called = True
            self.failure_reason = ""
            self._pending_target_intent = {
                "selected_index": 4,
                "preplace_position": np.array([0.1, 0.2, 0.3]),
                "place_position": np.array([0.1, 0.2, 0.2]),
            }
            self.manip_list = [
                SimpleNamespace(phase=SimpleNamespace(value="transit_preplace")),
                SimpleNamespace(phase=SimpleNamespace(value="terminal_place_descent")),
                SimpleNamespace(phase=SimpleNamespace(value="gripper_open")),
            ]

    place_module.Place = FakePlace
    monkeypatch.setitem(sys.modules, "core.skills.base_skill", base_module)
    monkeypatch.setitem(sys.modules, "core.skills.place", place_module)
    spec = importlib.util.spec_from_file_location("place_plan_probe_contract", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _probe(module, tmp_path, state, events):
    probe = module.PlacePlanProbe()
    probe.pick_obj = SimpleNamespace(name="object")
    probe.skill_cfg = {
        "candidate_id": "candidate_7",
        "objects": ["object", "support"],
        "result_path": str(tmp_path / "place_probe.json"),
    }
    manager = SimpleNamespace(
        records={
            "object": SimpleNamespace(
                state=state,
                owner_robot="robot",
                owner_arm="left",
            )
        },
        object_state_events=events,
    )
    probe.controller = SimpleNamespace(
        collision_world_mode="physics_schema",
        collision_scene_manager=manager,
        name="robot",
        lr_name="left",
        get_ee_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    probe.failure_reason = ""
    probe._pending_target_intent = None
    probe.manip_list = []
    return probe, manager


def test_place_probe_plans_only_after_real_attachment_event(monkeypatch, tmp_path):
    module = _load_probe_module(monkeypatch)
    event = {
        "entity": "object",
        "from": "active_target_approach",
        "to": "attached",
        "reason": "attach",
        "owner_robot": "robot",
        "owner_arm": "left",
        "world_revision": 9,
    }
    probe, manager = _probe(module, tmp_path, _State.ATTACHED, [event])

    probe.simple_generate_manip_cmds()

    result = json.loads((tmp_path / "place_probe.json").read_text(encoding="utf-8"))
    assert probe.super_called is True
    assert result["feasible"] is True
    assert result["failure_code"] is None
    assert result["attachment"] == event
    assert result["selected_target"]["selected_index"] == 4
    assert result["planned_phases"] == [
        "transit_preplace",
        "terminal_place_descent",
        "gripper_open",
    ]
    assert manager.records["object"].state is _State.ATTACHED
    assert probe.manip_list[0][2] == "update_pose_cost_metric"


def test_place_probe_never_forges_attached_state(monkeypatch, tmp_path):
    module = _load_probe_module(monkeypatch)
    probe, manager = _probe(module, tmp_path, _State.WORLD_OBSTACLE, [])

    probe.simple_generate_manip_cmds()

    result = json.loads((tmp_path / "place_probe.json").read_text(encoding="utf-8"))
    assert not hasattr(probe, "super_called")
    assert result["feasible"] is False
    assert result["failure_code"] == "PLACE_PROBE_OBJECT_NOT_ATTACHED"
    assert manager.records["object"].state is _State.WORLD_OBSTACLE
