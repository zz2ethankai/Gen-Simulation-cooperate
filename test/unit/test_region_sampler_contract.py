"""Mechanical contracts for placement offsets consumed by Banana regions."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))


class _PoseObject:
    def __init__(self, prim: str, z: float):
        self.prim = prim
        self._z = z

    def get_local_pose(self):
        return np.array([0.0, 0.0, self._z]), np.array([1.0, 0.0, 0.0, 0.0])


def test_a_on_b_region_sampler_adds_contact_offset_to_final_z(monkeypatch):
    boxes = {
        "robot": types.SimpleNamespace(
            min=np.array([-0.2, -0.2, 0.10]), max=np.array([0.2, 0.2, 1.10])
        ),
        "support": types.SimpleNamespace(
            min=np.array([-0.5, -0.5, 0.40]), max=np.array([0.5, 0.5, 0.52])
        ),
    }
    geometry_module = types.ModuleType("core.utils.usd_geom_utils")
    geometry_module.compute_bbox = lambda prim: boxes[prim]
    monkeypatch.setitem(sys.modules, "core.utils.usd_geom_utils", geometry_module)
    sys.modules.pop("core.utils.region_sampler", None)
    sampler_module = importlib.import_module("core.utils.region_sampler")

    class _Rotation:
        @classmethod
        def from_euler(cls, *args, **kwargs):
            return cls()

        @classmethod
        def from_quat(cls, *args, **kwargs):
            return cls()

        def __mul__(self, other):
            return self

        def as_quat(self, *args, **kwargs):
            return np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(sampler_module, "R", _Rotation)

    offset_m = 0.037
    position, _ = sampler_module.RandomRegionSampler.A_on_B_region_sampler(
        _PoseObject("robot", z=0.20),
        _PoseObject("support", z=0.0),
        pos_range=[[0.0, 0.0, offset_m], [0.0, 0.0, offset_m]],
        yaw_rotation=[0.0, 0.0],
    )

    authored_bottom_to_origin = 0.20 - 0.10
    assert position[2] == pytest.approx(
        0.52 + authored_bottom_to_origin + 0.001 + offset_m
    )
    monkeypatch.delitem(sys.modules, "core.utils.region_sampler")
