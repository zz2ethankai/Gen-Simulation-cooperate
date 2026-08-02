"""JSON conversion helpers for non-critical runtime debug artifacts."""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np


def json_ready(value: Any):
    """Recursively convert simulator and planner values to JSON-safe data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return json_ready(value.value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())

    value_type = type(value)
    if value_type.__module__.startswith("curobo") and value_type.__name__ == "JointState":
        payload = {}
        for field in ("position", "velocity", "acceleration", "jerk", "joint_names"):
            if hasattr(value, field):
                payload[field] = json_ready(getattr(value, field))
        return payload

    detach = getattr(value, "detach", None)
    if callable(detach):
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        if callable(cpu):
            detached = cpu()
        numpy_value = getattr(detached, "numpy", None)
        if callable(numpy_value):
            return json_ready(numpy_value())

    if value_type.__module__.startswith("pxr") and value_type.__name__.startswith("Vec"):
        return [json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return str(value)
