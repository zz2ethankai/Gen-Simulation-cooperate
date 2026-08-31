"""Behavior coverage for native batch chunking and original-index intersection."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("isaacsim")

from core.planning.domain_types import BatchPlanResult, JointTrajectory
from core.skills.pick import Pick
from core.skills.place import Place


def _path():
    return JointTrajectory(
        positions=[[0.0, 0.0], [0.1, 0.1]],
        joint_names=("joint_0", "joint_1"),
    )


class _BatchRuntime:
    grasp_approach_axis = 2

    def __init__(self, fail_global=None):
        self.calls = []
        self.fail_global = fail_global

    def transition_target(self, *args, **kwargs):
        self.calls.append(("transition", args, kwargs))

    def plan_pose_batch(self, positions, orientations, **kwargs):
        positions = np.asarray(positions)
        phase = kwargs["phase_id"]
        start = len([item for item in self.calls if item[0] == phase])
        offset = start * 20
        mask = np.ones(len(positions), dtype=bool)
        if self.fail_global is not None and offset <= self.fail_global < offset + len(positions):
            mask[self.fail_global - offset] = False
        self.calls.append((phase, len(positions), kwargs.get("start_paths")))
        return BatchPlanResult(
            success=tuple(bool(value) for value in mask),
            trajectories=tuple(_path() for _ in positions),
        )


def test_pick_chunks_all_candidates_and_intersects_original_indices():
    count = 41
    runtime = _BatchRuntime(fail_global=21)
    skill = object.__new__(Pick)
    skill.skill_runtime = runtime
    skill.pick_obj = SimpleNamespace(name="apple")
    skill.skill_cfg = {"pre_grasp_offset": 0.1}
    skill.fixed_orientation = None
    skill.sampled_scores = np.zeros(count)
    skill._candidate_raw_indices = np.arange(count)
    selected_inputs = []
    def select(_positions, _orientations, _transforms, valid):
        selected_inputs.append(np.asarray(valid))
        return int(valid[0]) if len(valid) else None
    skill._select_grasp_index = select
    transforms = np.repeat(np.eye(4, dtype=float)[None], count, axis=0)
    transforms[:, 0, 3] = np.arange(count, dtype=float)

    state = skill._plan_candidates(transforms)

    pre = [item for item in runtime.calls if item[0] == "pick_pregrasp_batch"]
    terminal = [item for item in runtime.calls if item[0] == "pick_grasp_batch"]
    assert [item[1] for item in pre] == [20, 20, 1]
    assert [item[1] for item in terminal] == [20, 20, 1]
    assert state["result"]["grasp_count"] == count
    assert state["result"]["selected_grasp_index"] == 0
    assert all(21 not in valid for valid in selected_inputs)


def test_place_keeps_original_indices_when_pre_batch_rejects_candidate():
    count = 41
    runtime = _BatchRuntime(fail_global=22)
    skill = object.__new__(Place)
    skill.skill_runtime = runtime
    skill.pick_obj = SimpleNamespace(name="apple")
    skill.place_obj = SimpleNamespace(name="tray")
    positions = np.repeat(np.eye(4, dtype=float)[None], count, axis=0)
    positions[:, 0, 3] = np.arange(count, dtype=float)
    geometry = {
        "pre_positions": positions[:, :3, 3].copy(),
        "place_positions": positions[:, :3, 3].copy() + np.array([0.0, 0.0, 0.05]),
        "pre_orientations": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "place_orientations": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
    }

    # Different target positions force the terminal query and exercise the
    # pre-index -> terminal-index mapping rather than a positional zip.
    selected = skill._plan_candidates(geometry)

    pre = [item for item in runtime.calls if item[0] == "place_preplace_batch"]
    terminal = [item for item in runtime.calls if item[0] == "place_terminal_batch"]
    assert [item[1] for item in pre] == [20, 20, 1]
    assert [item[1] for item in terminal] == [20, 20, 1]
    assert selected == 0
    assert terminal[1][2][2] is not None
