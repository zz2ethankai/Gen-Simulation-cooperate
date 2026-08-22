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

from core.planning.domain_types import (  # noqa: E402
    BatchPlanResult,
    CollisionPolicy,
)
from core.planning.grasp_plan_evaluator import GraspPlanEvaluator  # noqa: E402


class FakeResult(BatchPlanResult):
    _default_paths = object()

    def __init__(self, success, paths=_default_paths, *, selected=None, metrics=None):
        success_values = None if success is None else np.asarray(success, dtype=bool)
        if paths is self._default_paths:
            path_values = (
                [object()] * len(success_values)
                if success_values is not None
                else None
            )
        else:
            path_values = paths
        super().__init__(
            success=success_values,
            trajectories=None,
            status="ok",
            error=None,
            selected_candidate_index=selected,
            metrics=metrics,
            world_revision=0,
        )
        # Keep opaque sentinels in these tests so callback identity and
        # start-path forwarding remain observable without native planner data.
        self.trajectory = path_values
        self.trajectories = path_values
        self.status = "ok"
        self.error = None
        self.world_revision = 0
        self.candidate_indices = ()


def test_normalized_batch_selection_uses_selected_index_and_metrics():
    paths = [object(), object(), object()]
    pre_result = FakeResult([True, True, False], paths=paths)
    selected_result = FakeResult(
        [True, True, False],
        paths=paths,
        selected=1,
        metrics={"path_costs": [4.0, 1.0, 8.0]},
    )
    assert GraspPlanEvaluator.choose_candidate_index(pre_result, selected_result) == 1
    assert GraspPlanEvaluator.choose_candidate_index(
        pre_result,
        FakeResult([True, True, False], paths=paths, metrics={"path_costs": [4.0, 1.0, 8.0]}),
    ) == 1

class FakeSceneManager:
    def __init__(self, names):
        self.names = set(names)

    def has_native_obstacle(self, _controller, name):
        return name in self.names


class FakeController:
    batch_capability = True
    robot_file = "split_aloha_right.yml"
    lr_name = "right"

    def __init__(self, results, attach_names=("/target/mesh",)):
        self.results = iter(results)
        self.collision_scene_manager = FakeSceneManager(attach_names)

    def has_native_obstacle(self, name):
        return self.collision_scene_manager.has_native_obstacle(self, name)

    def plan_pose_batch(
        self,
        _positions,
        _orientations,
        *,
        start_paths=None,
        collision_policy=None,
        active_target=None,
    ):
        return next(self.results)

    @staticmethod
    def measure_cartesian_path(_path, _start, _goal):
        return 1.0, 0.0


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
        def plan_pose_batch(
            self,
            positions,
            orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            order.append(("plan", float(positions[0, 0])))
            return super().plan_pose_batch(
                positions,
                orientations,
                start_paths=start_paths,
                collision_policy=collision_policy,
                active_target=active_target,
            )

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


def test_typed_query_port_receives_world_and_target_policies():
    class TypedPort(FakeController):
        def __init__(self):
            super().__init__([])
            self.policies = []

        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            self.policies.append(collision_policy)
            if start_paths is None:
                return FakeResult([True, True, False])
            return FakeResult(
                [True, True, False],
                paths=[object(), object(), None],
            )

    port = TypedPort()
    result = GraspPlanEvaluator(port).evaluate(
        _grasps(), np.array([0.1, 0.2, 0.3]), 0.1, "/target/mesh"
    ).result
    assert result.feasible
    assert port.policies == [
        CollisionPolicy.WORLD_TRANSIT,
        CollisionPolicy.TARGET_APPROACH,
    ]


def test_chained_terminal_plan_rejects_nonstraight_path_and_keeps_selected_path():
    paths = [object(), object(), object()]

    class ChainedController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            starts = start_paths
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
    assert evaluation.result.joint_success_count == 1
    assert evaluation.result.selected_grasp_index == 1
    assert evaluation.terminal_path is paths[1]
    assert evaluation.terminal_path_length_ratio == 1.1
    assert evaluation.terminal_path_max_deviation_m == 0.004


def test_batch_candidate_selector_can_choose_physical_candidate():
    paths = [object(), object(), object()]
    calls = []

    def selector(pre_result, terminal_result, valid_indices, positions, orientations, transforms):
        calls.append(
            (
                pre_result,
                terminal_result,
                valid_indices.tolist(),
                positions.shape,
                orientations.shape,
                transforms.shape,
            )
        )
        return 1

    class SelectableController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            starts = start_paths
            assert starts is paths
            return FakeResult([True, True, False], paths=paths)

    evaluation = GraspPlanEvaluator(
        SelectableController([FakeResult([True, True, False], paths=paths)])
    ).evaluate(
        _grasps(),
        np.array([0.3, 0.1, 0.2]),
        0.1,
        "/target/mesh",
        candidate_selector=selector,
    )
    assert evaluation.result.selected_grasp_index == 1
    assert len(calls) == 1
    assert calls[0][2:] == ([0, 1], (3, 3), (3, 4), (3, 4, 4))


def test_batch_candidate_selector_cannot_choose_filtered_candidate():
    paths = [object(), object(), object()]

    class SelectableController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            starts = start_paths
            assert starts is paths
            return FakeResult([True, True, False], paths=paths)

        def measure_cartesian_path(self, path, _start, _goal):
            return (1.8, 0.02) if path is paths[0] else (1.1, 0.004)

    evaluation = GraspPlanEvaluator(
        SelectableController([FakeResult([True, True, False], paths=paths)])
    ).evaluate(
        _grasps(),
        np.array([0.3, 0.1, 0.2]),
        0.1,
        "/target/mesh",
        candidate_selector=lambda *_args: 0,
    )
    assert evaluation.result.selected_grasp_index == 1
    assert evaluation.terminal_path is paths[1]


def test_postgrasp_validator_filters_terminal_candidates_before_selection():
    paths = [object(), object(), object()]

    class AttachedController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            starts = start_paths
            assert starts is paths
            return FakeResult([True, True, False], paths=paths)

    checked = []

    def validate(candidate_index, path):
        checked.append((candidate_index, path))
        return {"success": candidate_index == 1, "mode": "test"}

    evaluation = GraspPlanEvaluator(
        AttachedController([FakeResult([True, True, False], paths=paths)])
    ).evaluate(
        _grasps(),
        np.array([0.3, 0.1, 0.2]),
        0.1,
        "/target/mesh",
        postgrasp_validator=validate,
    )

    assert evaluation.result.feasible
    assert evaluation.result.grasp_success_count == 1
    assert evaluation.result.joint_success_count == 1
    assert evaluation.result.selected_grasp_index == 1
    assert checked == [(0, paths[0]), (1, paths[1])]
    assert [item["candidate_index"] for item in evaluation.post_grasp_validation] == [0, 1]


def test_chained_terminal_all_failed_returns_safe_failure_without_paths():
    paths = [object(), object(), object()]

    class FailedChainedController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
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


def test_terminal_empty_success_mask_fails_closed_without_broadcast_error():
    paths = [object(), object(), object()]

    class EmptyTerminalController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            return FakeResult(None, paths=None)

    controller = EmptyTerminalController(
        [FakeResult([True, True, True], paths=paths)],
    )
    evaluation = GraspPlanEvaluator(controller).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )

    assert not evaluation.result.feasible
    assert evaluation.result.pregrasp_success_count == 3
    assert evaluation.result.grasp_success_count == 0
    assert evaluation.result.joint_success_count == 0
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"
    assert evaluation.terminal_path is None


def test_terminal_literal_empty_success_mask_fails_closed():
    paths = [object(), object(), object()]

    class EmptyTerminalController(FakeController):
        def plan_pose_batch(
            self,
            _positions,
            _orientations,
            *,
            start_paths=None,
            collision_policy=None,
            active_target=None,
        ):
            if start_paths is None:
                return super().plan_pose_batch(
                    _positions,
                    _orientations,
                    collision_policy=collision_policy,
                    active_target=active_target,
                )
            return FakeResult(np.empty((0,), dtype=bool), paths=None)

    evaluation = GraspPlanEvaluator(
        EmptyTerminalController([FakeResult([True, True, True], paths=paths)])
    ).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )

    assert not evaluation.result.feasible
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"


def test_pregrasp_path_count_mismatch_fails_closed():
    class MismatchedController(FakeController):
        def plan_pose_batch(self, *_args, **_kwargs):
            if _kwargs.get("start_paths") is None:
                return super().plan_pose_batch(*_args)
            raise AssertionError("terminal planning must not run with mismatched paths")

    evaluation = GraspPlanEvaluator(
        MismatchedController([FakeResult([True, True, True], paths=[object()])])
    ).evaluate(
        _grasps(), np.array([0.3, 0.1, 0.2]), 0.1, "/target/mesh"
    )

    assert not evaluation.result.feasible
    assert evaluation.result.pregrasp_success_count == 3
    assert evaluation.result.grasp_success_count == 0
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"


def test_all_pregrasps_failed_does_not_call_terminal_world():
    class ChainedController(FakeController):
        def plan_pose_batch(self, *_args, **_kwargs):
            if _kwargs.get("start_paths") is None:
                return super().plan_pose_batch(*_args)
            raise AssertionError("terminal planning must not run without a pregrasp path")

    terminal_world_calls = []
    controller = ChainedController(
        [FakeResult([False, False, False], paths=None)]
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
        batch_capability = False

        def __init__(self):
            super().__init__([])
            self.pre_index = 0
            self.terminal_starts = []

        def plan_pose_result(
            self, _position, _orientation, *, collision_policy=None, active_target=None
        ):
            index = self.pre_index
            self.pre_index += 1
            return FakeResult([index in {0, 1}], paths=pre_paths[index])

        def plan_pose_from_path(
            self,
            _position,
            _orientation,
            start_path,
            *,
            collision_policy=None,
            active_target=None,
        ):
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
        batch_capability = False

        def __init__(self):
            super().__init__([])

        def plan_pose_result(
            self, _position, _orientation, *, collision_policy=None, active_target=None
        ):
            return FakeResult([True], paths=pre_path)

        def plan_pose_from_path(
            self,
            _position,
            _orientation,
            start_path,
            *,
            collision_policy=None,
            active_target=None,
        ):
            assert start_path is pre_path
            return FakeResult([True], paths=terminal_path)

        def measure_cartesian_path(self, _path, _start, _goal):
            return 1.7, 0.02

    evaluation = GraspPlanEvaluator(NonStraightController()).evaluate(
        _grasps(1), np.array([0.1]), 0.1, "/target/mesh"
    )
    assert not evaluation.result.feasible
    assert evaluation.result.grasp_success_count == 0
    assert evaluation.result.joint_success_count == 0
    assert evaluation.result.failure_code == "NO_JOINT_GRASP_PLAN"
    assert evaluation.terminal_path is None


def test_nonbatch_postgrasp_validator_filters_terminal_candidate():
    pre_paths = [object(), object()]
    terminal_paths = [object(), object()]

    class NonBatchController(FakeController):
        batch_capability = False

        def __init__(self):
            super().__init__([])
            self.pre_index = 0

        def plan_pose_result(
            self, _position, _orientation, *, collision_policy=None, active_target=None
        ):
            path = pre_paths[self.pre_index]
            self.pre_index += 1
            return FakeResult([True], paths=path)

        def plan_pose_from_path(
            self,
            _position,
            _orientation,
            start_path,
            *,
            collision_policy=None,
            active_target=None,
        ):
            index = pre_paths.index(start_path)
            return FakeResult([True], paths=terminal_paths[index])

        @staticmethod
        def measure_cartesian_path(_path, _start, _goal):
            return 1.0, 0.0

    evaluation = GraspPlanEvaluator(NonBatchController()).evaluate(
        _grasps(2),
        np.array([0.3, 0.1]),
        0.1,
        "/target/mesh",
        postgrasp_validator=lambda candidate_index, _path: candidate_index == 1,
    )

    assert evaluation.result.feasible
    assert evaluation.result.grasp_success_count == 1
    assert evaluation.result.joint_success_count == 1
    assert evaluation.result.selected_grasp_index == 1
    assert evaluation.terminal_path is terminal_paths[1]


def test_pick_and_probe_import_the_same_evaluator():
    pick_source = (ROOT / "workflows/simbox/core/skills/pick.py").read_text(encoding="utf-8")
    probe_source = (ROOT / "workflows/simbox/core/skills/pick_plan_probe.py").read_text(encoding="utf-8")
    assert "from core.planning.grasp_plan_evaluator import GraspPlanEvaluator" in pick_source
    assert "super().simple_generate_manip_cmds()" in probe_source
    assert "test_batch_forward" not in probe_source
