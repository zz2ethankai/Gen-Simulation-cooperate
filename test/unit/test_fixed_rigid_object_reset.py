"""Regression coverage for fixed Scene-4 rigid-object reset metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]

from workflows.simbox.core.utils.region_metadata import (  # noqa: E402
    merge_source_region_sampling_metadata,
)
from workflows.simbox.core.utils.rigid_pose import upright_world_orientation  # noqa: E402


def _load_region_sampler_without_usd_dependencies():
    """Load the pure sampler math while replacing its USD bbox adapter."""

    module_name = "_fixed_reset_region_sampler"
    module_path = ROOT / "workflows/simbox/core/utils/region_sampler.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    usd_geom_utils = types.ModuleType("core.utils.usd_geom_utils")
    usd_geom_utils.compute_bbox = lambda prim: prim  # replaced per test
    previous = sys.modules.get("core.utils.usd_geom_utils")
    sys.modules["core.utils.usd_geom_utils"] = usd_geom_utils
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if previous is None:
            sys.modules.pop("core.utils.usd_geom_utils", None)
        else:
            sys.modules["core.utils.usd_geom_utils"] = previous
    return module


region_sampler = _load_region_sampler_without_usd_dependencies()


class _FakePrim:
    def __init__(self, minimum, maximum):
        self.bbox = (np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float))


class _FakeObject:
    def __init__(self, position, orientation, bbox):
        self.prim = _FakePrim(*bbox)
        self._position = np.asarray(position, dtype=float)
        self._orientation = np.asarray(orientation, dtype=float)

    def get_world_pose(self):
        return self._position.copy(), self._orientation.copy()

    def set_world_pose(self, position, orientation):
        self._position = np.asarray(position, dtype=float)
        self._orientation = np.asarray(orientation, dtype=float)


def _quat_close(first, second):
    # q and -q describe the same rotation.
    return np.isclose(abs(float(np.dot(first, second))), 1.0, atol=1e-7)


def test_scene4_source_sampling_metadata_matches_normalized_object_names():
    path = (
        ROOT
        / "InternDataAssets/assets/custom/scene_4/01_kitchen/assets/basic/"
        / "kitchen_apple_to_tray/simbox_task.yaml"
    )
    task = yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"][0]
    active = next(item for item in task["regions"] if item["object"] == "apple_0_id9008")
    assert "sampling" not in active

    merge_source_region_sampling_metadata(task)

    assert active["sampling"]["keep_upright"] is True
    assert active["sampling"]["mode"] == "fixed"


def test_fixed_region_pose_is_repeatable_for_same_seed(monkeypatch):
    """A fixed, upright reset must not inherit a prior roll/pitch."""

    minimum = np.asarray([3.0, 0.5, 0.8])
    maximum = np.asarray([3.4, 0.9, 1.0])
    monkeypatch.setattr(
        region_sampler,
        "compute_bbox",
        lambda prim: type("BBox", (), {"min": prim.bbox[0], "max": prim.bbox[1]})(),
    )

    configured_orientation = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_quat(
        scalar_first=True
    )
    object_pose = _FakeObject(
        [3.2, 0.8, 1.1],
        R.from_euler("xyz", [38.0, -22.0, 17.0], degrees=True).as_quat(scalar_first=True),
        ([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]),
    )
    target = _FakeObject([0.0, 0.0, 0.0], configured_orientation, (minimum, maximum))

    results = []
    for _ in range(2):
        # reset_fixed_rigid_objects establishes the configured fixed pose
        # before asking the keep_upright sampler for the final world pose.
        object_pose.set_world_pose(object_pose.get_world_pose()[0], configured_orientation)
        np.random.seed(23)
        results.append(
            region_sampler.RandomRegionSampler.A_on_B_region_sampler(
                object_pose,
                target,
                pos_range=[[-0.2, 0.308333, 0.000139], [-0.2, 0.308333, 0.000139]],
                yaw_rotation=[0.0, 0.0],
                keep_upright=True,
            )
        )

    assert np.allclose(results[0][0], results[1][0], atol=1e-9)
    assert _quat_close(np.asarray(results[0][1]), np.asarray(results[1][1]))
    assert _quat_close(np.asarray(results[0][1]), configured_orientation)


def test_keep_upright_normalizes_tilted_captured_pose_and_preserves_yaw():
    tilted = R.from_euler("xyz", [38.0, -22.0, 17.0], degrees=True).as_quat(
        scalar_first=True
    )

    restored = upright_world_orientation(tilted)
    euler = R.from_quat(restored, scalar_first=True).as_euler("xyz", degrees=True)

    np.testing.assert_allclose(euler[:2], [0.0, 0.0], atol=1e-7)
    assert np.isclose(euler[2], 17.0, atol=1e-7)


def test_fixed_restore_is_wired_after_normal_layout_warmup():
    """Keep the fixed-object restore/audit on the effective reset path."""

    workflow_source = (ROOT / "workflows/simbox_dual_workflow.py").read_text(encoding="utf-8")
    task_source = (ROOT / "workflows/simbox/core/tasks/banana.py").read_text(encoding="utf-8")

    assert 'def _restore_fixed_rigid_objects_after_warmup(self, label):' in workflow_source
    assert '_restore_fixed_rigid_objects_after_warmup("layout_warmup_complete")' in workflow_source
    assert 'audit_fixed_rigid_object_reset' in workflow_source
    assert 'def restore_fixed_rigid_object_states(self, label="warmup"):' in task_source
    assert 'def audit_fixed_rigid_object_reset(self, label="audit"):' in task_source
    assert 'LOGGER = logging.getLogger("de_logger")' in task_source
