"""Host-side regression test for strict Pick terminal candidate recovery."""

from __future__ import annotations

import ast
import logging
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

# Keep the evaluator import independent of Isaac Sim on the host.
if "core.utils.transformation_utils" not in sys.modules:
    transform_module = types.ModuleType("core.utils.transformation_utils")
    transform_module.poses_from_tf_matrices = lambda values: (
        np.asarray(values)[:, :3, 3],
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(values), 1)),
    )
    sys.modules["core.utils.transformation_utils"] = transform_module

from core.planning.domain_types import (  # noqa: E402
    CollisionPolicy,
    PlanResult,
)
from core.controllers.pick_planning import PickPlanningPort  # noqa: E402
from core.planning.collision_scene_manager import PlannerScenePort  # noqa: E402
from core.planning.grasp_plan_evaluator import (  # noqa: E402
    GraspPlanEvaluation,
    GraspPlanEvaluator,
    GraspPlanResult,
)
from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402
from core.planning.planner_runtime import PlannerRuntime  # noqa: E402


def _pick_class():
    """Load Pick's class body without importing simulator-only dependencies."""

    path = ROOT / "workflows" / "simbox" / "core" / "skills" / "pick.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pick_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Pick"
    )

    class _BaseSkill:
        pass

    namespace = {
        "BaseSkill": _BaseSkill,
        "register_skill": lambda value: value,
        "Robot": object,
        "BaseTask": object,
        "DictConfig": object,
        "MotionPhase": MotionPhase,
        "MotionPhaseCommand": MotionPhaseCommand,
        "CollisionPolicy": CollisionPolicy,
        "GraspPlanEvaluator": GraspPlanEvaluator,
        "CUROBO_BATCH_SIZE": 20,
        "np": np,
        "random": random,
        "LOGGER": logging.getLogger("pick-fallback-test"),
    }

    def _tf_matrix_from_pose(position, _orientation):
        transform = np.eye(4)
        transform[:3, 3] = np.asarray(position, dtype=float).reshape(3)
        return transform

    namespace["tf_matrix_from_pose"] = _tf_matrix_from_pose
    namespace["pose_from_tf_matrix"] = lambda transform: (
        np.asarray(transform, dtype=float)[:3, 3],
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    exec(
        compile(ast.Module(body=[pick_class], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["Pick"]


class _TensorArgs:
    """Small device adapter required by the formal planner scene port."""

    @staticmethod
    def to_device(value):
        return value


class _SceneManager:
    """Typed-port test manager; it deliberately has no controller aliases."""

    def __init__(self, owner):
        self.owner = owner

    def refresh_controller_reference_world(self, _port, *, force=False):
        self.owner.calls.append(("refresh", force))
        return True

    def sync_dynamic_poses(self, step_id, *, interval_steps, force=False):
        self.owner.calls.append(("sync", step_id, interval_steps, force))
        return ["target"]

    def begin_target_transit(self, object_name, robot_name, arm_name):
        self.owner.calls.append(
            ("transition", object_name, CollisionPolicy.WORLD_TRANSIT)
        )

    def begin_target_approach(self, object_name, robot_name, arm_name):
        self.owner.calls.append(
            ("transition", object_name, CollisionPolicy.TARGET_APPROACH)
        )

    def restore_world(self, object_name):
        self.owner.calls.append(("restore", object_name))

    def has_native_obstacle(self, _port, path):
        return path == "/target/mesh"


class _Planning(PickPlanningPort):
    """Formal Pick port backed by a dependency-injected planner runtime."""

    def __init__(self, transforms=None):
        self.calls = []
        self.transforms = transforms
        self.shift_on_retarget = False
        self.expected_plan_x = 16.0
        self._tensor_args = _TensorArgs()
        self._runtime = PlannerRuntime(scene_revision=1, name="pick-fallback-test")
        # PickPlanningPort exposes the runtime's robot/device contract for
        # timing and grasp queries.  Keep the native runtime itself free of
        # simulator objects in this host-side test.
        self._runtime.robot_port = SimpleNamespace(
            tensor_args=self._tensor_args,
            interpolation_dt=0.01,
        )
        scene_port = PlannerScenePort(
            name="robot",
            lr_name="right",
            reference_prim_path="/World/robot/base",
            robot_ee_path="/World/robot/ee",
            tensor_args=self._tensor_args,
            robot=SimpleNamespace(),
            runtime=self._runtime,
        )
        self._scene_manager = _SceneManager(self)
        super().__init__(
            scene_port=scene_port,
            collision_scene_manager=self._scene_manager,
            update_pose_cost_metric=lambda _value: None,
            build_commands=lambda **kwargs: kwargs,
            arm_base_transform=lambda: np.eye(4),
            frame_debug=lambda: {},
            capture_reference=lambda _name: None,
            retarget_commands=self._retarget_queue,
            replan_after_safety=self._retarget_commands,
            execution_ee_pose=lambda: (
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            initial_ee_pose=lambda: (
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            phase_complete=lambda _command: True,
            robot_file="panda_right.yml",
            plan_pose_result=self._plan_pose_result,
            measure_cartesian_path=lambda _path, _start, _goal: (1.0, 0.0),
        )

    def _retarget_commands(self, object_name, command, commands):
        self.calls.append(("retarget", object_name))
        if self.shift_on_retarget and self.transforms is not None:
            self.transforms[:, 0, 3] += 1.0
            for pending in commands:
                if pending.target_position is not None:
                    pending.target_position = pending.target_position.copy()
                    pending.target_position[0] += 1.0
        return True

    @staticmethod
    def _retarget_queue(_object_name, commands):
        return commands

    def _plan_pose_result(
        self,
        position,
        orientation,
        *,
        request_metadata,
    ):
        candidate_position = float(np.asarray(position)[0])
        collision_policy = request_metadata["collision_policy"]
        active_target = request_metadata["active_target"]
        self.calls.append(("plan", candidate_position, collision_policy, active_target))
        assert candidate_position == self.expected_plan_x
        return PlanResult(
            success=True,
            trajectory=[[0.0, 0.0], [1.0, 1.0]],
            collision_policy=collision_policy,
            phase_id=request_metadata["phase_id"],
            world_revision=self.world_revision,
        )


def test_terminal_forbidden_contact_replans_candidate_15_to_16_without_relaxing_policy():
    pick_class = _pick_class()
    result = GraspPlanResult(
        feasible=True,
        arm="right",
        grasp_count=17,
        pregrasp_success_count=17,
        grasp_success_count=17,
        joint_success_count=17,
        selected_grasp_index=15,
        selected_grasp_score=0.15,
        attach_prim_valid=True,
        attach_prim_path="/target/mesh",
        attach_prim_paths=["/target/mesh"],
        missing_attach_prim_paths=[],
        attach_candidate_paths=[],
    )
    evaluation = GraspPlanEvaluation(
        result=result,
        pregrasp_positions=np.zeros((17, 3)),
        pregrasp_orientations=np.zeros((17, 4)),
        grasp_positions=np.arange(17, dtype=float)[:, None] * np.ones((17, 3)),
        grasp_orientations=np.zeros((17, 4)),
        terminal_paths=[object() for _ in range(17)],
    )
    pregrasp = MotionPhaseCommand(
        MotionPhase.TRANSIT_PREGRASP,
        np.array([15.0, 0.0, -0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
    )
    terminal = MotionPhaseCommand(
        MotionPhase.TERMINAL_GRASP_APPROACH,
        np.array([15.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
        allow_target_finger_contact=True,
    )
    terminal.params["candidate_index"] = 15
    close = MotionPhaseCommand(
        MotionPhase.GRIPPER_CLOSE,
        terminal.target_position.copy(),
        terminal.target_orientation.copy(),
        active_object="target",
        allow_target_finger_contact=True,
        replan_allowed=False,
    )
    attach = MotionPhaseCommand(
        MotionPhase.ATTACH,
        active_object="target",
        allow_target_finger_contact=True,
        replan_allowed=False,
    )
    current_transforms = np.tile(np.eye(4), (17, 1, 1))
    current_transforms[:, 0, 3] = np.arange(17, dtype=float)
    planning = _Planning(current_transforms)
    planning.shift_on_retarget = True
    planning.expected_plan_x = 17.0
    lift = MotionPhaseCommand(
        MotionPhase.POST_GRASP_LIFT,
        np.array([15.0, 0.0, 0.25]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
    )
    pick = SimpleNamespace(
        plan_evaluation=evaluation,
        planning=planning,
        pick_obj=SimpleNamespace(name="target"),
        manip_list=[pregrasp, terminal, close, attach, lift],
        _candidate_rank_order=[15, 16],
        _candidate_raw_indices=np.arange(17, dtype=int),
        _candidate_failed_indices=set(),
        _candidate_failure_diagnostics=[],
        _candidate_replan_diagnostics=[],
        _terminal_replan_count=0,
        _candidate_replan_limit=0,
        sampled_scores=np.arange(17, dtype=float),
        _selected_candidate_debug={},
        _write_debug_artifact=lambda *_args, **_kwargs: None,
        failure_reason="",
        error_message="",
        process_valid=True,
        fixed_orientation=None,
        skill_cfg={"pre_grasp_offset": 0.1},
        get_ee_poses=lambda _frame: planning.transforms,
    )
    for method_name in (
        "_record_terminal_candidate_event",
        "_candidate_pose_key",
        "_terminal_candidate_order",
        "_current_candidate_poses",
        "_sync_replacement_candidate_targets",
        "_replan_terminal_candidate",
    ):
        setattr(pick, method_name, getattr(pick_class, method_name).__get__(pick))

    assert pick._replan_terminal_candidate(
        terminal, reason="forbidden_link_contact"
    )
    assert evaluation.result.selected_grasp_index == 16
    assert terminal.params["candidate_index"] == 16
    assert terminal.params["candidate_replan_limit"] == 0
    assert terminal.metadata["candidate_replan_limit"] == 0
    assert terminal.planning_request_metadata["candidate_replan_limit"] == 0
    assert terminal.planning_request_metadata["replan_policy"] == (
        "terminal_candidate_fallback"
    )
    assert terminal.params["preplanned_joint_path"] is not None
    assert [command.phase for command in pick.manip_list] == [
        MotionPhase.TRANSIT_PREGRASP,
        MotionPhase.TERMINAL_GRASP_APPROACH,
        MotionPhase.GRIPPER_CLOSE,
        MotionPhase.ATTACH,
        MotionPhase.POST_GRASP_LIFT,
    ]
    assert np.allclose(pregrasp.target_position, [17.0, 0.0, -0.1])
    assert np.allclose(terminal.target_position, [17.0, 0.0, 0.0])
    assert np.allclose(close.target_position, [17.0, 0.0, 0.0])
    assert np.allclose(lift.target_position, [17.0, 0.0, 0.25])
    assert terminal.allow_target_finger_contact
    assert not terminal.allow_target_robot_contact
    assert not close.allow_target_robot_contact
    assert not attach.allow_target_robot_contact
    assert any(
        event["candidate_index"] == 15
        and event["reason"] == "forbidden_link_contact"
        for event in pick._candidate_failure_diagnostics
    )
    assert any(
        event["candidate_index"] == 16
        and event["success"]
        for event in pick._candidate_replan_diagnostics
    )
    assert planning.calls == [
        ("retarget", "target"),
        ("transition", "target", CollisionPolicy.TARGET_APPROACH),
        ("plan", 17.0, CollisionPolicy.TARGET_APPROACH, "target"),
    ]


def test_duplicate_raw_slots_are_removed_before_sampling_and_recovery_ranking():
    pick_class = _pick_class()
    raw_transforms = np.tile(np.eye(4), (17, 1, 1))
    raw_transforms[:, 0, 3] = np.arange(17, dtype=float)
    raw_transforms[13] = raw_transforms[12]
    sampled = SimpleNamespace(
        T_obj_ee=raw_transforms,
        _raw_grasp_keys=pick_class._build_raw_grasp_keys(raw_transforms),
        _candidate_raw_indices=np.empty((0,), dtype=int),
        scores=np.arange(1.0, 18.0),
        sampled_scores=np.empty((0,), dtype=float),
        skill_cfg={},
        pick_obj=SimpleNamespace(name="target"),
        _sample_debug={},
        _debug_log=lambda _message: None,
        get_ee_poses=lambda _frame: raw_transforms,
    )
    for method_name in (
        "_deduplicated_raw_grasp_indices",
        "sample_ee_pose",
    ):
        setattr(sampled, method_name, getattr(pick_class, method_name).__get__(sampled))

    random.seed(7)
    sampled.sample_ee_pose(max_length=17)
    raw_indices = sampled._candidate_raw_indices.tolist()
    # Sampling follows the pre-migration score-weighted ``random.choices``
    # behavior and therefore may repeat a physical candidate to fill the
    # requested batch.  Duplicate annotation slots are still removed from the
    # source pool before sampling.
    assert len(raw_indices) == 17
    assert set(raw_indices).issubset(set(range(17)) - {13})
    assert 13 not in raw_indices
    assert sampled._sample_debug["deduplicated_raw_indices"] == [
        *range(13),
        *range(14, 17),
    ]
    assert sampled._sample_debug["unique_candidate_count"] == 16

    positions = np.arange(17, dtype=float)[:, None] * np.ones((17, 3))
    positions[13] = positions[12]
    evaluation = GraspPlanEvaluation(
        result=GraspPlanResult(
            feasible=True,
            arm="right",
            grasp_count=17,
            pregrasp_success_count=17,
            grasp_success_count=17,
            joint_success_count=17,
            selected_grasp_index=12,
            selected_grasp_score=0.12,
            attach_prim_valid=True,
            attach_prim_path="/target/mesh",
            attach_prim_paths=["/target/mesh"],
            missing_attach_prim_paths=[],
            attach_candidate_paths=[],
        ),
        pregrasp_positions=positions.copy(),
        pregrasp_orientations=np.zeros((17, 4)),
        grasp_positions=positions,
        grasp_orientations=np.zeros((17, 4)),
    )
    ranked = SimpleNamespace(
        plan_evaluation=evaluation,
        _candidate_rank_order=[12, 13, 16],
        _candidate_failed_indices={12},
    )
    ranked._candidate_pose_key = pick_class._candidate_pose_key.__get__(ranked)
    ranked._terminal_candidate_order = pick_class._terminal_candidate_order.__get__(ranked)
    recovery_order = ranked._terminal_candidate_order(12)
    assert recovery_order[0] == 16
    assert 13 not in recovery_order


def test_terminal_recovery_reaches_fourth_unique_candidate_within_typed_budget():
    pick_class = _pick_class()
    count = 4
    positions = np.arange(count, dtype=float)[:, None] * np.ones((count, 3))
    evaluation = GraspPlanEvaluation(
        result=GraspPlanResult(
            feasible=True,
            arm="right",
            grasp_count=count,
            pregrasp_success_count=count,
            grasp_success_count=count,
            joint_success_count=count,
            selected_grasp_index=0,
            selected_grasp_score=0.0,
            attach_prim_valid=True,
            attach_prim_path="/target/mesh",
            attach_prim_paths=["/target/mesh"],
            missing_attach_prim_paths=[],
            attach_candidate_paths=[],
        ),
        pregrasp_positions=positions.copy(),
        pregrasp_orientations=np.zeros((count, 4)),
        grasp_positions=positions.copy(),
        grasp_orientations=np.zeros((count, 4)),
        terminal_paths=[object() for _ in range(count)],
    )
    pregrasp = MotionPhaseCommand(
        MotionPhase.TRANSIT_PREGRASP,
        np.array([0.0, 0.0, -0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
    )
    terminal = MotionPhaseCommand(
        MotionPhase.TERMINAL_GRASP_APPROACH,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
        allow_target_finger_contact=True,
    )
    terminal.params["candidate_index"] = 0
    close = MotionPhaseCommand(
        MotionPhase.GRIPPER_CLOSE,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
        allow_target_finger_contact=True,
        replan_allowed=False,
    )
    attach = MotionPhaseCommand(
        MotionPhase.ATTACH,
        active_object="target",
        allow_target_finger_contact=True,
        replan_allowed=False,
    )
    lift = MotionPhaseCommand(
        MotionPhase.POST_GRASP_LIFT,
        np.array([0.0, 0.0, 0.25]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        active_object="target",
    )
    transforms = np.tile(np.eye(4), (count, 1, 1))
    transforms[:, 0, 3] = np.arange(count, dtype=float)
    planning = _Planning(transforms)
    pick = SimpleNamespace(
        plan_evaluation=evaluation,
        planning=planning,
        pick_obj=SimpleNamespace(name="target"),
        manip_list=[pregrasp, terminal, close, attach, lift],
        _candidate_rank_order=[0, 1, 2, 3],
        _candidate_raw_indices=np.arange(count, dtype=int),
        _candidate_failed_indices=set(),
        _candidate_failure_diagnostics=[],
        _candidate_replan_diagnostics=[],
        _terminal_replan_count=0,
        _candidate_replan_limit=3,
        sampled_scores=np.arange(count, dtype=float),
        _selected_candidate_debug={},
        _write_debug_artifact=lambda *_args, **_kwargs: None,
        failure_reason="",
        error_message="",
        process_valid=True,
        fixed_orientation=None,
        skill_cfg={"pre_grasp_offset": 0.1},
        get_ee_poses=lambda _frame: planning.transforms,
    )
    for method_name in (
        "_record_terminal_candidate_event",
        "_candidate_pose_key",
        "_terminal_candidate_order",
        "_terminal_candidate_replan_budget",
        "_current_candidate_poses",
        "_sync_replacement_candidate_targets",
        "_replan_terminal_candidate",
    ):
        setattr(pick, method_name, getattr(pick_class, method_name).__get__(pick))

    assert pick._terminal_candidate_replan_budget() == 3

    # Each safety event retires only the active physical candidate.  The
    # supervisor's typed limit of N-1 therefore reaches candidate 3 (the
    # fourth unique candidate) after candidates 0, 1, and 2 fail.
    for current_index, replacement_index in ((0, 1), (1, 2), (2, 3)):
        terminal.params["candidate_index"] = current_index
        planning.expected_plan_x = float(replacement_index)
        assert pick._replan_terminal_candidate(
            terminal, reason="forbidden_link_contact"
        )
        assert terminal.params["candidate_index"] == replacement_index

    assert evaluation.result.selected_grasp_index == 3
    assert pick._candidate_failed_indices == {0, 1, 2}
    assert pick._terminal_replan_count == 3
    assert terminal.candidate_replan_limit == 3
    assert terminal.replan_policy == "terminal_candidate_fallback"
    assert terminal.params["candidate_replan_limit"] == 3
    assert terminal.metadata["candidate_replan_limit"] == 3
    assert terminal.metadata["replan_policy"] == "terminal_candidate_fallback"
    assert terminal.planning_request_metadata["candidate_replan_limit"] == 3
    assert terminal.planning_request_metadata["replan_policy"] == (
        "terminal_candidate_fallback"
    )
    assert not terminal.allow_target_robot_contact
    assert any(
        event["candidate_index"] == 3 and event["success"]
        for event in pick._candidate_replan_diagnostics
    )
