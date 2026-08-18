"""Single CuRobo grasp-plan evaluation implementation used by Pick and Probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from core.utils.plan_utils import select_index_by_priority_dual, select_index_by_priority_single
from core.utils.transformation_utils import poses_from_tf_matrices


@dataclass
class GraspPlanResult:
    feasible: bool
    arm: str
    grasp_count: int
    pregrasp_success_count: int
    grasp_success_count: int
    joint_success_count: int
    selected_grasp_index: int | None
    selected_grasp_score: float | None
    attach_prim_valid: bool
    attach_prim_path: str | None
    attach_prim_paths: list[str]
    missing_attach_prim_paths: list[str]
    attach_candidate_paths: list[str]
    failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraspPlanEvaluation:
    result: GraspPlanResult
    pregrasp_positions: np.ndarray
    pregrasp_orientations: np.ndarray
    grasp_positions: np.ndarray
    grasp_orientations: np.ndarray
    pregrasp_path: Any | None = None
    pregrasp_path_index: int | None = None
    terminal_path: Any | None = None
    terminal_path_length_ratio: float | None = None
    terminal_path_max_deviation_m: float | None = None


class GraspPlanEvaluator:
    """Evaluate pre-grasp and grasp paths without executing robot commands."""

    def __init__(self, controller: Any, debug_log: Callable[[str], None] | None = None):
        self.controller = controller
        self.debug_log = debug_log or (lambda _message: None)

    @property
    def arm(self) -> str:
        value = getattr(self.controller, "lr_name", None)
        if value in {"left", "right"}:
            return value
        raise ValueError("grasp-plan controller must declare arm_id")

    @staticmethod
    def _normalize_attach_paths(prim_paths: str | Sequence[str]) -> list[str]:
        if isinstance(prim_paths, str):
            return [prim_paths] if prim_paths else []
        return [str(path) for path in prim_paths]

    def missing_attach_prims(self, prim_paths: str | Sequence[str]) -> list[str]:
        motion_gen = getattr(self.controller, "motion_gen", None)
        world = getattr(motion_gen, "world_model", None)
        paths = self._normalize_attach_paths(prim_paths)
        if world is None:
            return paths
        return [path for path in paths if world.get_obstacle(path) is None]

    def attach_prims_valid(self, prim_paths: str | Sequence[str]) -> bool:
        paths = self._normalize_attach_paths(prim_paths)
        return bool(paths) and not self.missing_attach_prims(paths)

    def attach_prim_valid(self, prim_path: str) -> bool:
        """Compatibility wrapper for callers that still pass one path."""

        return self.attach_prims_valid([prim_path])

    @staticmethod
    def _count(result: Any) -> int:
        success = getattr(result, "success", None)
        return int(success.sum().item()) if success is not None else 0

    @staticmethod
    def _planner_diagnostic(result: Any) -> str:
        """Return the CuRobo failure layer without changing planner behavior."""

        status = getattr(result, "status", None)
        status = getattr(status, "value", status)
        return "status=%s valid_query=%s success=%d" % (
            status,
            getattr(result, "valid_query", None),
            GraspPlanEvaluator._count(result),
        )

    @staticmethod
    def _log_terminal_candidates(debug_log, result: Any, pre_result: Any, grasp_positions: np.ndarray) -> None:
        """Per-candidate terminal-grasp detail for the pregrasp-successful
        subset.  Prints grasp height (arm-base z) against pos/rot error and
        success so a terminal failure can be read as orientation-unreachable
        (rot_error above threshold at every candidate) vs collision-only
        (rot_error ~ 0 but TrajOpt still fails).  Diagnostics must never
        break planning."""
        try:
            success = getattr(result, "success", None)
            pre_ok = getattr(pre_result, "success", None)
            pos_err = getattr(result, "position_error", None)
            rot_err = getattr(result, "rotation_error", None)
            if success is None or pre_ok is None:
                return
            success_np = np.asarray(success.detach().cpu().numpy(), dtype=bool).reshape(-1)
            pre_ok_np = np.asarray(pre_ok.detach().cpu().numpy(), dtype=bool).reshape(-1)
            pos_np = pos_err.detach().cpu().numpy().reshape(-1) if pos_err is not None else None
            rot_np = rot_err.detach().cpu().numpy().reshape(-1) if rot_err is not None else None
            rows = []
            for i in np.nonzero(pre_ok_np)[0]:
                rows.append(
                    "c%d z=%.4f pos=%s rot=%s ok=%d"
                    % (
                        i,
                        float(grasp_positions[i, 2]),
                        f"{float(pos_np[i]):.4f}" if pos_np is not None else "-",
                        f"{float(rot_np[i]):.4f}" if rot_np is not None else "-",
                        int(bool(success_np[i])),
                    )
                )
            if rows:
                debug_log("terminal-candidates " + " ".join(rows))
        except Exception as exc:  # diagnostics must never break planning
            debug_log("terminal-candidates <diag failed: %r>" % (exc,))

    def evaluate(
        self,
        grasp_transforms: np.ndarray,
        grasp_scores: np.ndarray,
        pregrasp_offset_m: float,
        attach_prim_paths: str | Sequence[str],
        fixed_orientation: np.ndarray | None = None,
        test_mode: str = "forward",
        attach_config_failure_code: str | None = None,
        attach_candidate_paths: Sequence[str] | None = None,
        attach_missing_paths: Sequence[str] | None = None,
        prepare_pregrasp_world: Callable[[], None] | None = None,
        prepare_grasp_world: Callable[[], None] | None = None,
        cartesian_ratio_limit: float = 1.5,
        cartesian_deviation_m: float = 0.01,
    ) -> GraspPlanEvaluation:
        paths = self._normalize_attach_paths(attach_prim_paths)
        candidates = [str(path) for path in (attach_candidate_paths or [])]
        if attach_config_failure_code is not None:
            return self._empty(attach_config_failure_code, paths, candidates)
        if not paths:
            return self._empty("ATTACH_COLLISION_CONFIG_MISSING", paths, candidates)
        missing_paths = (
            self.missing_attach_prims(paths)
            if attach_missing_paths is None
            else [str(path) for path in attach_missing_paths]
        )
        if missing_paths:
            return self._empty(
                "ATTACH_COLLISION_PRIM_NOT_IN_CUROBO_WORLD",
                paths,
                candidates,
                missing_paths=missing_paths,
            )

        grasps = np.asarray(grasp_transforms)
        scores = np.asarray(grasp_scores).reshape(-1)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) == 0:
            return self._empty("NO_GRASP_CANDIDATE", paths, candidates)
        if len(scores) != len(grasps):
            raise ValueError("grasp_scores length must match grasp_transforms")

        pregrasps = grasps.copy()
        approach_axis = int(self.controller.grasp_approach_axis)
        pregrasps[:, :3, 3] -= pregrasps[:, :3, approach_axis] * float(
            pregrasp_offset_m
        )
        pre_positions, pre_orientations = poses_from_tf_matrices(pregrasps)
        positions, orientations = poses_from_tf_matrices(grasps)
        if fixed_orientation is not None:
            pre_orientations[:] = fixed_orientation
            orientations[:] = fixed_orientation

        pre_success_count = 0
        grasp_success_count = 0
        joint_success_count = 0
        selected_index: int | None = None
        pregrasp_path_index: int | None = None
        pregrasp_paths = None
        terminal_paths = None
        path_metrics: dict[int, tuple[float, float]] = {}
        if prepare_pregrasp_world is not None:
            prepare_pregrasp_world()
        if self.controller.use_batch:
            if np.array_equal(pre_positions, positions) and np.array_equal(pre_orientations, orientations):
                result = self.controller.test_batch_forward(positions, orientations)
                self.debug_log("grasp-only " + self._planner_diagnostic(result))
                grasp_success_count = self._count(result)
                pre_success_count = grasp_success_count
                joint_success_count = grasp_success_count
                if joint_success_count:
                    selected_index = int(select_index_by_priority_single(result))
                    terminal_paths = result.get_paths()
            else:
                pre_result = self.controller.test_batch_forward(pre_positions, pre_orientations)
                self.debug_log("pregrasp " + self._planner_diagnostic(pre_result))
                pre_success_count = self._count(pre_result)
                # CuRobo leaves ``interpolated_plan`` as None when the whole
                # pre-grasp batch fails. MotionGenResult.get_paths() assumes
                # it is present and raises TypeError instead of returning an
                # empty list. Treat this as the expected no-path outcome and
                # never enter the terminal-world planning pass.
                if pre_success_count:
                    pregrasp_paths = pre_result.get_paths()
                    pregrasp_path_index = next(
                        index
                        for index, success in enumerate(pre_result.success)
                        if bool(success)
                    )
                    if prepare_grasp_world is not None:
                        prepare_grasp_world()
                    if hasattr(self.controller, "test_batch_forward_from_paths"):
                        result = self.controller.test_batch_forward_from_paths(
                            positions, orientations, pregrasp_paths
                        )
                        self.debug_log("terminal-grasp " + self._planner_diagnostic(result))
                        self._log_terminal_candidates(self.debug_log, result, pre_result, positions)
                         # Diagnostic only: is the grasp pose reachable from a fresh
                        # home start?  The chained start (pregrasp endpoint) can land
                        # in a local IK minimum even when the target is reachable from
                        # home, which changes the fix (relax/waypoint vs move target).
                        if hasattr(self.controller, "test_batch_forward"):
                            try:
                                grasp_home = self.controller.test_batch_forward(positions, orientations)
                                self.debug_log("grasp-from-home " + self._planner_diagnostic(grasp_home))
                            except Exception as exc:  # diagnostics must never break planning
                                self.debug_log("grasp-from-home <diag failed: %r>" % (exc,))
                        if self._count(result):
                            terminal_paths = result.get_paths()
                            for candidate_index, path in enumerate(terminal_paths):
                                if bool(pre_result.success[candidate_index]) and bool(result.success[candidate_index]):
                                    ratio, deviation = self.controller.measure_cartesian_path(
                                        path, pre_positions[candidate_index], positions[candidate_index]
                                    )
                                    path_metrics[candidate_index] = (ratio, deviation)
                                    if ratio > cartesian_ratio_limit or deviation > cartesian_deviation_m:
                                        result.success[candidate_index] = False
                    else:
                        # Compatibility for legacy/mock controllers. Standard
                        # Pick controllers implement the chained-start API.
                        result = self.controller.test_batch_forward(positions, orientations)
                    grasp_success_count = self._count(result)
                    both = pre_result.success & result.success
                    joint_success_count = int(both.sum().item())
                    if joint_success_count:
                        selected_index = int(select_index_by_priority_dual(pre_result, result))
                        pregrasp_path_index = selected_index
        else:
            pre_results = []
            pre_paths: list[Any | None] = []
            pregrasp_paths = pre_paths
            for pre_position, pre_orientation in zip(pre_positions, pre_orientations):
                if test_mode == "forward":
                    if hasattr(self.controller, "test_single_forward_result"):
                        pre_result = self.controller.test_single_forward_result(
                            pre_position, pre_orientation
                        )
                        pre_ok = bool(pre_result.success.item())
                        pre_path = (
                            pre_result.get_interpolated_plan() if pre_ok else None
                        )
                    else:
                        pre_ok = bool(
                            self.controller.test_single_forward(
                                pre_position, pre_orientation
                            )
                        )
                        pre_path = None
                elif test_mode == "ik":
                    pre_ok = bool(self.controller.test_single_ik(pre_position, pre_orientation))
                    pre_path = None
                else:
                    raise ValueError(f"unsupported grasp test_mode: {test_mode}")
                pre_success_count += int(pre_ok)
                pre_results.append(pre_ok)
                pre_paths.append(pre_path)
                if pre_ok and pre_path is not None and pregrasp_path_index is None:
                    pregrasp_path_index = len(pre_paths) - 1
            if prepare_grasp_world is not None:
                prepare_grasp_world()
            for index, (position, orientation) in enumerate(zip(positions, orientations)):
                if test_mode == "forward":
                    if (
                        pre_results[index]
                        and pre_paths[index] is not None
                        and hasattr(self.controller, "test_single_forward_from_path")
                    ):
                        grasp_result = self.controller.test_single_forward_from_path(
                            position, orientation, pre_paths[index]
                        )
                        grasp_ok = bool(grasp_result.success.item())
                        terminal_path = (
                            grasp_result.get_interpolated_plan() if grasp_ok else None
                        )
                        if terminal_path is not None:
                            ratio, deviation = self.controller.measure_cartesian_path(
                                terminal_path,
                                pre_positions[index],
                                positions[index],
                            )
                            path_metrics[index] = (ratio, deviation)
                            if ratio > cartesian_ratio_limit or deviation > cartesian_deviation_m:
                                grasp_ok = False
                                terminal_path = None
                    elif pre_results[index]:
                        # Legacy/mock compatibility. Runtime Physics-schema
                        # controllers always provide the chained-start method.
                        grasp_ok = bool(
                            self.controller.test_single_forward(position, orientation)
                        )
                        terminal_path = None
                    else:
                        grasp_ok = False
                        terminal_path = None
                else:
                    grasp_ok = bool(self.controller.test_single_ik(position, orientation))
                    terminal_path = None
                grasp_success_count += int(grasp_ok)
                if pre_results[index] and grasp_ok:
                    joint_success_count += 1
                    if selected_index is None:
                        selected_index = index
                        pregrasp_path_index = index
                        if terminal_path is not None:
                            terminal_paths = {index: terminal_path}

        attach_valid = True
        if joint_success_count == 0:
            failure_code = "NO_JOINT_GRASP_PLAN"
        else:
            failure_code = None
        feasible = failure_code is None
        result_value = GraspPlanResult(
            feasible=feasible,
            arm=self.arm,
            grasp_count=len(grasps),
            pregrasp_success_count=pre_success_count,
            grasp_success_count=grasp_success_count,
            joint_success_count=joint_success_count,
            selected_grasp_index=selected_index,
            selected_grasp_score=float(scores[selected_index]) if selected_index is not None else None,
            attach_prim_valid=attach_valid,
            attach_prim_path=paths[0] if len(paths) == 1 else None,
            attach_prim_paths=paths,
            missing_attach_prim_paths=[],
            attach_candidate_paths=candidates,
            failure_code=failure_code,
        )
        self.debug_log(
            "grasp-plan arm=%s grasps=%d pre=%d grasp=%d joint=%d selected=%s attach=%s failure=%s"
            % (
                result_value.arm,
                result_value.grasp_count,
                pre_success_count,
                grasp_success_count,
                joint_success_count,
                selected_index,
                attach_valid,
                failure_code,
            )
        )
        return GraspPlanEvaluation(
            result=result_value,
            pregrasp_positions=pre_positions,
            pregrasp_orientations=pre_orientations,
            grasp_positions=positions,
            grasp_orientations=orientations,
            pregrasp_path=(
                pregrasp_paths[pregrasp_path_index]
                if pregrasp_paths is not None and pregrasp_path_index is not None
                else None
            ),
            pregrasp_path_index=pregrasp_path_index,
            terminal_path=(
                terminal_paths[selected_index]
                if terminal_paths is not None and selected_index is not None
                else None
            ),
            terminal_path_length_ratio=(
                path_metrics[selected_index][0] if selected_index in path_metrics else None
            ),
            terminal_path_max_deviation_m=(
                path_metrics[selected_index][1] if selected_index in path_metrics else None
            ),
        )

    def _empty(
        self,
        failure_code: str,
        attach_prim_paths: Sequence[str],
        attach_candidate_paths: Sequence[str] | None = None,
        missing_paths: Sequence[str] | None = None,
    ) -> GraspPlanEvaluation:
        paths = [str(path) for path in attach_prim_paths]
        empty_positions = np.empty((0, 3), dtype=float)
        empty_orientations = np.empty((0, 4), dtype=float)
        return GraspPlanEvaluation(
            result=GraspPlanResult(
                feasible=False,
                arm=self.arm,
                grasp_count=0,
                pregrasp_success_count=0,
                grasp_success_count=0,
                joint_success_count=0,
                selected_grasp_index=None,
                selected_grasp_score=None,
                attach_prim_valid=False,
                attach_prim_path=paths[0] if len(paths) == 1 else None,
                attach_prim_paths=paths,
                missing_attach_prim_paths=[str(path) for path in (missing_paths or [])],
                attach_candidate_paths=[str(path) for path in (attach_candidate_paths or [])],
                failure_code=failure_code,
            ),
            pregrasp_positions=empty_positions,
            pregrasp_orientations=empty_orientations,
            grasp_positions=empty_positions.copy(),
            grasp_orientations=empty_orientations.copy(),
        )
