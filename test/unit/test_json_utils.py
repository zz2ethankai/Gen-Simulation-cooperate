"""Tests for debug-artifact JSON conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.json_utils import json_ready  # noqa: E402


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class JointState:
    def __init__(self):
        self.position = FakeTensor([1.0, 2.0])
        self.velocity = FakeTensor([0.1, 0.2])
        self.acceleration = None
        self.jerk = None
        self.joint_names = ["joint_a", "joint_b"]


JointState.__module__ = "curobo.types.state"


def test_curobo_joint_state_is_expanded_to_json_safe_fields():
    result = json_ready(JointState())

    assert result == {
        "position": [1.0, 2.0],
        "velocity": [0.1, 0.2],
        "acceleration": None,
        "jerk": None,
        "joint_names": ["joint_a", "joint_b"],
    }
    json.dumps(result)


def test_unknown_debug_value_degrades_to_string():
    class Unknown:
        def __str__(self):
            return "unknown-debug-value"

    assert json_ready({"value": Unknown()}) == {"value": "unknown-debug-value"}
