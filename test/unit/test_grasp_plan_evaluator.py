"""Unit tests for the shared Pick/Probe grasp-plan decision logic."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

# The evaluator only needs matrix-to-pose conversion.  Stub the Isaac-backed
# utility so this decision logic remains testable outside the simulator.
transform_module = types.ModuleType("core.utils.transformation_utils")


def _poses_from_tf_matrices(values):
    positions = np.asarray(values)[:, :3, 3]
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(values), 1))
    return positions, orientations


transform_module.poses_from_tf_matrices = _poses_from_tf_matrices
sys.modules["core.utils.transformation_utils"] = transform_module
plan_utils_module = types.ModuleType("core.utils.plan_utils")
plan_utils_module.select_index_by_priority_single = lambda result: int(np.flatnonzero(result.success)[0])
plan_utils_module.select_index_by_priority_dual = (
    lambda pre, final: int(np.flatnonzero(pre.success & final.success)[0])
)
sys.modules["core.utils.plan_utils"] = plan_utils_module

from core.planning.grasp_plan_evaluator import GraspPlanEvaluator  # noqa: E402


class FakeResult:
    def __init__(self, success, paths=None):
        self.success = np.asarray(success, dtype=bool)
        self._paths = paths

    def get_paths(self):
        return self._paths

    def get_interpolated_plan(self):
        return self._paths


class FakeWorld:
    def __init__(self, names):
        self.names = set(names)

    def get_obstacle(self, name):
        return object() if name in self.names else None


class FakeController:
    use_batch = True
    robot_file = "split_aloha_right.yml"
    lr_name = "right"

    def __init__(self, results, attach_names=("/target/mesh",)):
        self.results = iter(results)
        self.world_cfg = FakeWorld(attach_names)
        self.motion_gen = types.SimpleNamespace(world_model=FakeWorld(attach_names))

    def test_batch_forward(self, _positions, _orientations):
        return next(self.results)


def _grasps(count=3):
    values = np.tile(np.eye(4), (count, 1, 1))
    values[:, 0, 3] = np.arange(count) * 0.1
    return values


def test_joint_success_is_required_without_fallback():
    controller = FakeController(
        [FakeResult([True, False, False]), FakeResult([False, True, False])], attach_names=("/target/mesh",)
    )
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.1, 0.2, 0.3]), 0.1, "/target/mesh"
    ).result
    assert not result.feasible
    assert result.joint_success_count == 0
    assert result.selected_grasp_index is None
    assert result.failure_code == "NO_JOINT_GRASP_PLAN"


def test_joint_candidate_and_attach_prim_are_both_required():
    controller = FakeController([FakeResult([True, True, False]), FakeResult([False, True, True])])
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    ).result
    assert result.feasible
    assert result.joint_success_count == 1
    assert result.selected_grasp_index == 1
    assert result.selected_grasp_score == 0.1
    assert result.attach_prim_valid


def test_attach_prim_mismatch_is_reported_before_pick_execution():
    controller = FakeController(
        [FakeResult([True, False, False]), FakeResult([True, False, False])], attach_names=()
    )
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.1, 0.2, 0.3]), 0.1, "/target/mesh"
    ).result
    assert not result.feasible
    assert result.joint_success_count == 0
    assert result.failure_code == "ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD"
    assert result.missing_attach_prim_paths == ["/target/mesh"]


def test_all_attach_prims_must_exist_for_multi_prim_object():
    controller = FakeController(
        [FakeResult([True, False, False]), FakeResult([True, False, False])],
        attach_names=("/target/part_0",),
    )
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(),
        np.array([0.1, 0.2, 0.3]),
        0.1,
        ["/target/part_0", "/target/part_1"],
    ).result
    assert not result.feasible
    assert result.attach_prim_path is None
    assert result.attach_prim_paths == ["/target/part_0", "/target/part_1"]
    assert result.missing_attach_prim_paths == ["/target/part_1"]


def test_config_failure_is_preserved_without_launching_motion_plans():
    controller = FakeController([])
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(),
        np.array([0.1, 0.2, 0.3]),
        0.1,
        [],
        attach_config_failure_code="ATTACH_COLLISION_PRIM_AMBIGUOUS",
        attach_candidate_paths=["/target/a", "/target/b"],
    ).result
    assert not result.feasible
    assert result.failure_code == "ATTACH_COLLISION_PRIM_AMBIGUOUS"
    assert result.attach_candidate_paths == ["/target/a", "/target/b"]


def test_pre_ignore_attach_check_can_be_reused_during_grasp_planning():
    controller = FakeController(
        [FakeResult([True, True, False]), FakeResult([False, True, True])],
        attach_names=(),
    )
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(),
        np.array([0.3, 0.1, 0.2]),
        0.1,
        ["/target/mesh"],
        attach_missing_paths=[],
    ).result
    assert result.feasible
    assert result.attach_prim_valid


def test_pregrasp_and_terminal_world_callbacks_wrap_the_two_plan_sets():
    order = []

    class OrderedController(FakeController):
        def test_batch_forward(self, positions, orientations):
            order.append(("plan", float(positions[0, 0])))
            return super().test_batch_forward(positions, orientations)

    controller = OrderedController(
        [FakeResult([True, False, False]), FakeResult([True, False, False])]
    )
    result = GraspPlanEvaluator(controller).evaluate(
        _grasps(),
        np.array([0.1, 0.2, 0.3]),
        0.1,
        "/target/mesh",
        prepare_pregrasp_world=lambda: order.append(("world", "full")),
        prepare_grasp_world=lambda: order.append(("world", "terminal")),
    ).result
    assert result.feasible
    assert [item[1] for item in order if item[0] == "world"] == ["full", "terminal"]
    assert order[0] == ("world", "full")
    assert order[2] == ("world", "terminal")


def test_chained_terminal_plan_rejects_nonstraight_path_and_keeps_selected_path():
    paths = [object(), object(), object()]

    class ChainedController(FakeController):
        def test_batch_forward_from_paths(self, _positions, _orientations, starts):
            assert starts is paths
            return FakeResult([True, True, False], paths=paths)

        def measure_cartesian_path(self, path, _start, _goal):
            return (1.8, 0.02) if path is paths[0] else (1.1, 0.004)

    controller = ChainedController(
        [FakeResult([True, True, False], paths=paths)],
    )
    evaluation = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )
    assert evaluation.result.feasible
    assert evaluation.result.selected_grasp_index == 1
    assert evaluation.terminal_path is paths[1]
    assert evaluation.terminal_path_length_ratio == 1.1
    assert evaluation.terminal_path_max_deviation_m == 0.004


def test_chained_terminal_all_failed_returns_safe_failure_without_paths():
    paths = [object(), object(), object()]

    class FailedChainedController(FakeController):
        def test_batch_forward_from_paths(self, _positions, _orientations, _starts):
            return FakeResult([False, False, False], paths=None)

    controller = FailedChainedController(
        [FakeResult([True, True, True], paths=paths)],
    )
    evaluation = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )
    assert not evaluation.result.feasible
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"
    assert evaluation.terminal_path is None


def test_all_pregrasps_failed_does_not_call_curobo_get_paths_or_terminal_world():
    class NoInterpolatedPlanResult(FakeResult):
        def get_paths(self):
            raise TypeError("object of type 'NoneType' has no len()")

    class ChainedController(FakeController):
        def test_batch_forward_from_paths(self, *_args):
            raise AssertionError("terminal planning must not run without a pregrasp path")

    terminal_world_calls = []
    controller = ChainedController(
        [NoInterpolatedPlanResult([False, False, False], paths=None)]
    )
    evaluation = GraspPlanEvaluator(controller).evaluate(
        _grasps(),
        np.array([0.1, 0.2, 0.3]),
        0.1,
        "/target/mesh",
        prepare_grasp_world=lambda: terminal_world_calls.append(True),
    )
    assert not evaluation.result.feasible
    assert evaluation.result.pregrasp_success_count == 0
    assert evaluation.result.grasp_success_count == 0
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"
    assert terminal_world_calls == []


def test_nonbatch_terminal_plan_starts_from_matching_pregrasp_path():
    pre_paths = [object(), object(), object()]
    terminal_paths = [None, object(), None]

    class NonBatchController(FakeController):
        use_batch = False

        def __init__(self):
            super().__init__([])
            self.pre_index = 0
            self.terminal_starts = []

        def test_single_forward_result(self, _position, _orientation):
            index = self.pre_index
            self.pre_index += 1
            return FakeResult([index in {0, 1}], paths=pre_paths[index])

        def test_single_forward_from_path(self, _position, _orientation, start_path):
            self.terminal_starts.append(start_path)
            index = pre_paths.index(start_path)
            return FakeResult([index == 1], paths=terminal_paths[index])

        def measure_cartesian_path(self, path, _start, _goal):
            assert path is terminal_paths[1]
            return 1.2, 0.005

    evaluation = GraspPlanEvaluator(NonBatchController()).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )
    assert evaluation.result.feasible
    assert evaluation.result.pregrasp_success_count == 2
    assert evaluation.result.grasp_success_count == 1
    assert evaluation.result.selected_grasp_index == 1
    assert evaluation.terminal_path is terminal_paths[1]
    assert evaluation.terminal_path_length_ratio == 1.2
    assert evaluation.terminal_path_max_deviation_m == 0.005


def test_nonbatch_terminal_plan_rejects_nonstraight_path():
    pre_path = object()
    terminal_path = object()

    class NonStraightController(FakeController):
        use_batch = False

        def __init__(self):
            super().__init__([])

        def test_single_forward_result(self, _position, _orientation):
            return FakeResult([True], paths=pre_path)

        def test_single_forward_from_path(self, _position, _orientation, start_path):
            assert start_path is pre_path
            return FakeResult([True], paths=terminal_path)

        def measure_cartesian_path(self, _path, _start, _goal):
            return 1.7, 0.02

    evaluation = GraspPlanEvaluator(NonStraightController()).evaluate(
        _grasps(1), np.array([0.1]), 0.1, "/target/mesh"
    )
    assert not evaluation.result.feasible
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"
    assert evaluation.terminal_path is None


def test_pick_and_probe_import_the_same_evaluator():
    pick_source = (ROOT / "workflows/simbox/core/skills/pick.py").read_text(encoding="utf-8")
    probe_source = (ROOT / "workflows/simbox/core/skills/pick_plan_probe.py").read_text(encoding="utf-8")
    assert "from core.planning.grasp_plan_evaluator import GraspPlanEvaluator" in pick_source
    assert "super().simple_generate_manip_cmds()" in probe_source
    assert "test_batch_forward" not in probe_source
