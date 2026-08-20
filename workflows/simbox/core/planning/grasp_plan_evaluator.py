"""Single CuRobo grasp-plan evaluation implementation used by Pick and Probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from core.utils.plan_utils import (
    extract_result_paths,
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
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
    terminal_path: Any | None = None
    terminal_paths: list[Any | None] = field(default_factory=list)
    terminal_path_length_ratio: float | None = None
    terminal_path_max_deviation_m: float | None = None
    pregrasp_plan_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    terminal_plan_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    post_grasp_validation: list[dict[str, Any]] = field(default_factory=list)


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
        return "right" if "right" in str(self.controller.robot_file) else "left"

    @staticmethod
    def _normalize_attach_paths(prim_paths: str | Sequence[str]) -> list[str]:
        if isinstance(prim_paths, str):
            return [prim_paths] if prim_paths else []
        return [str(path) for path in prim_paths]

    def missing_attach_prims(self, prim_paths: str | Sequence[str]) -> list[str]:
        world = getattr(self.controller, "world_cfg", None)
        paths = self._normalize_attach_paths(prim_paths)
        if world is None:
            return paths
        return [path for path in paths if world.get_obstacle(path) is None]

    def attach_prims_valid(self, prim_paths: str | Sequence[str]) -> bool:
        paths = self._normalize_attach_paths(prim_paths)
        return bool(paths) and not self.missing_attach_prims(paths)

    @staticmethod
    def _count(result: Any) -> int:
        success = GraspPlanEvaluator._success_mask(result)
        return int(np.count_nonzero(success))

    @staticmethod
    def _success_mask(result: Any) -> np.ndarray:
        success = getattr(result, "success", None)
        if success is None:
            return np.zeros(0, dtype=bool)
        if hasattr(success, "detach"):
            success = success.detach().cpu().numpy()
        success = np.asarray(success, dtype=bool)
        return success.any(axis=-1) if success.ndim > 1 else success

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
    def _tensor_summary(value: Any) -> dict[str, Any] | None:
        """Summarize a native CuRobo tensor without leaking device objects."""

        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            values = np.asarray(value)
            if values.size == 0:
                return {"shape": list(values.shape), "count": 0}
            finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
            summary: dict[str, Any] = {"shape": list(values.shape)}
            if np.issubdtype(values.dtype, np.bool_):
                summary.update({"true_count": int(values.sum()), "count": int(values.size)})
            elif np.issubdtype(values.dtype, np.number):
                summary["finite_count"] = int(finite.size)
                if finite.size:
                    summary.update(
                        {
                            "min": float(finite.min()),
                            "max": float(finite.max()),
                            "mean": float(finite.mean()),
                        }
                    )
            return summary
        except (TypeError, ValueError, RuntimeError):
            return {"type": type(value).__name__}

    @classmethod
    def _result_diagnostic(cls, result: Any, candidate_index: int) -> dict[str, Any]:
        """Capture native failure layers for a candidate planning attempt."""

        status = getattr(result, "status", None)
        status = getattr(status, "value", status)
        if status is not None:
            status = str(status)
        success = getattr(result, "success", None)
        feasible = getattr(result, "feasible", None)
        debug_info = getattr(result, "debug_info", None)
        trajectory_info = {}
        for name in ("interpolated_trajectory", "js_solution"):
            state = getattr(result, name, None)
            position = getattr(state, "position", None)
            trajectory_info[name] = {
                "present": state is not None,
                "position_shape": list(position.shape) if position is not None else None,
            }
        return {
            "candidate_index": int(candidate_index),
            "success": cls._tensor_summary(success),
            "feasible": cls._tensor_summary(feasible),
            "status": status,
            "valid_query": getattr(result, "valid_query", None),
            "position_error": cls._tensor_summary(getattr(result, "position_error", None)),
            "rotation_error": cls._tensor_summary(getattr(result, "rotation_error", None)),
            "cspace_error": cls._tensor_summary(getattr(result, "cspace_error", None)),
            "goalset_index": cls._tensor_summary(getattr(result, "goalset_index", None)),
            "num_seeds": getattr(result, "num_seeds", None),
            "batch_size": getattr(result, "batch_size", None),
            "trajectory": trajectory_info,
            "interpolated_last_tstep": cls._tensor_summary(
                getattr(result, "interpolated_last_tstep", None)
            ),
            "debug_info_keys": sorted(str(key) for key in debug_info.keys())
            if isinstance(debug_info, dict)
            else None,
        }

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
        candidate_selector: Callable[
            [Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], int
        ]
        | None = None,
        postgrasp_validator: Callable[[int, Any], bool | dict[str, Any]] | None = None,
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
        if "r5a" in str(self.controller.robot_file):
            pregrasps[:, :3, 3] -= pregrasps[:, :3, 0] * float(pregrasp_offset_m)
        else:
            pregrasps[:, :3, 3] -= pregrasps[:, :3, 2] * float(pregrasp_offset_m)
        pre_positions, pre_orientations = poses_from_tf_matrices(pregrasps)
        positions, orientations = poses_from_tf_matrices(grasps)
        if fixed_orientation is not None:
            pre_orientations[:] = fixed_orientation
            orientations[:] = fixed_orientation

        pre_success_count = 0
        grasp_success_count = 0
        joint_success_count = 0
        selected_index: int | None = None
        pregrasp_paths: list[Any | None] = []
        terminal_paths: list[Any | None] = []
        path_metrics: dict[int, tuple[float, float]] = {}
        pregrasp_plan_diagnostics: list[dict[str, Any]] = []
        terminal_plan_diagnostics: list[dict[str, Any]] = []
        post_grasp_validation: list[dict[str, Any]] = []

        def _validate_postgrasp(candidate_index: int, terminal_path: Any) -> bool:
            """Run optional post-grasp validation and normalize its diagnostics."""
            if postgrasp_validator is None:
                return True

            try:
                validation = postgrasp_validator(int(candidate_index), terminal_path)
                if isinstance(validation, dict):
                    diagnostic = dict(validation)
                    success = bool(diagnostic.get("success", False))
                else:
                    success = bool(validation)
                    diagnostic = {"success": success}
            except Exception as exc:  # Runtime callback failures are candidate failures.
                success = False
                diagnostic = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            diagnostic["candidate_index"] = int(candidate_index)
            post_grasp_validation.append(diagnostic)
            return success

        if prepare_pregrasp_world is not None:
            prepare_pregrasp_world()
        if self.controller.use_batch:
            if np.array_equal(pre_positions, positions) and np.array_equal(pre_orientations, orientations):
                # A zero pre-grasp offset still has terminal-grasp semantics.
                # The legacy evaluator switched to its grasp-specific world
                # before testing this path, where the object being picked was
                # excluded from the owner's collision world.  Keep that
                # transition in the native v2 path as well; otherwise a goal
                # pose that is intentionally inside the target's collider is
                # rejected by IK before the gripper can close around it.
                if prepare_grasp_world is not None:
                    prepare_grasp_world()
                result = self.controller.test_batch_forward(positions, orientations)
                self.debug_log("grasp-only " + self._planner_diagnostic(result))
                grasp_success_count = self._count(result)
                pre_success_count = grasp_success_count
                joint_success_count = grasp_success_count
                if joint_success_count:
                    selected_index = int(select_index_by_priority_single(result))
            else:
                pre_result = self.controller.test_batch_forward(pre_positions, pre_orientations)
                self.debug_log("pregrasp " + self._planner_diagnostic(pre_result))
                pregrasp_plan_diagnostics.append(self._result_diagnostic(pre_result, -1))
                pre_success_count = self._count(pre_result)
                pre_paths = extract_result_paths(pre_result)
                candidate_count = len(grasps)
                if len(pre_paths) != candidate_count:
                    self.debug_log(
                        "pregrasp path-count mismatch "
                        f"expected={candidate_count} got={len(pre_paths)}; "
                        "treating all terminal candidates as failed"
                    )
                    pre_paths = [None] * candidate_count
                pregrasp_paths = pre_paths
                pre_success = self._success_mask(pre_result)
                if pre_success.shape != (candidate_count,) or not np.any(pre_success):
                    pre_success = np.zeros(candidate_count, dtype=bool)
                # BatchMotionPlanner returns None when its batched IK graph
                # finds no seed.  That is not proof that every candidate is
                # invalid: the single native planner uses its normal seed
                # budget and can solve a candidate from the same world.  Try
                # that bounded fallback before declaring the whole Pick
                # infeasible, and retain the result diagnostics in the
                # snapshot for post-run review.
                if not np.any(pre_success) or not any(path is not None for path in pre_paths):
                    single_forward = getattr(
                        self.controller, "test_single_forward_result", None
                    )
                    if callable(single_forward):
                        pre_paths = [None] * candidate_count
                        pregrasp_paths = pre_paths
                        for candidate_index in range(candidate_count):
                            single_result = single_forward(
                                pre_positions[candidate_index],
                                pre_orientations[candidate_index],
                            )
                            diagnostic = self._result_diagnostic(
                                single_result, candidate_index
                            )
                            diagnostic["mode"] = "native_single_pregrasp_fallback"
                            pregrasp_plan_diagnostics.append(diagnostic)
                            single_paths = extract_result_paths(single_result)
                            if (
                                self._count(single_result)
                                and single_paths
                                and single_paths[0] is not None
                            ):
                                pre_success[candidate_index] = True
                                pre_paths[candidate_index] = single_paths[0]
                        pre_success_count = int(np.count_nonzero(pre_success))
                # Native v2 may leave ``interpolated_trajectory`` unset when
                # every candidate fails.
                # Treat this as the expected no-path outcome and
                # never enter the terminal-world planning pass.
                result = None
                if pre_success_count and any(path is not None for path in pre_paths):
                    if prepare_grasp_world is not None:
                        prepare_grasp_world()
                    result = self.controller.test_batch_forward_from_paths(
                        positions, orientations, pre_paths
                    )
                    self.debug_log("terminal-grasp " + self._planner_diagnostic(result))
                    result_success = self._success_mask(result)
                    # Native v2 returns ``success=None`` when every batch
                    # IK/TrajOpt attempt fails.  A batch failure is not
                    # necessarily a per-candidate failure: the batch solver
                    # uses one padded CUDA-graph query, while the ordinary
                    # native planner can still solve an individual goal
                    # from the same named pre-grasp endpoint.  Retry only
                    # this exceptional all-failed case; successful batch
                    # planning keeps the fast path and its exact result.
                    single_fallback = False
                    if result_success.shape != pre_success.shape:
                        self.debug_log(
                            "terminal-grasp success-mask mismatch "
                            f"pre={pre_success.shape} final={result_success.shape}; "
                            "treating terminal candidates as failed"
                        )
                        result_success = np.zeros_like(pre_success)
                    if not np.any(result_success):
                        single_fallback = True
                        self.debug_log(
                            "terminal-grasp batch returned no valid candidate; "
                            "retrying eligible candidates with native single planner"
                        )
                    path_available = np.zeros_like(pre_success)
                    terminal_paths = [None] * len(pre_success)
                    single_from_path = getattr(
                        self.controller, "test_single_forward_from_path", None
                    )
                    if single_fallback and not callable(single_from_path):
                        self.debug_log(
                            "terminal-grasp single fallback unavailable; "
                            "controller has no native path query"
                        )
                    elif single_fallback:
                        for candidate_index, (pre_ok, pre_path) in enumerate(
                            zip(pre_success, pre_paths)
                        ):
                            if not pre_ok or pre_path is None:
                                continue
                            single_result = single_from_path(
                                positions[candidate_index],
                                orientations[candidate_index],
                                pre_path,
                            )
                            diagnostic = self._result_diagnostic(
                                single_result, candidate_index
                            )
                            diagnostic["mode"] = "native_single_fallback"
                            single_paths = extract_result_paths(single_result)
                            single_path = (
                                single_paths[0]
                                if self._count(single_result) and single_paths
                                else None
                            )
                            if single_path is None:
                                diagnostic["path_available"] = False
                                terminal_plan_diagnostics.append(diagnostic)
                                continue
                            ratio, deviation = self.controller.measure_cartesian_path(
                                single_path,
                                pre_positions[candidate_index],
                                positions[candidate_index],
                            )
                            diagnostic["path_available"] = True
                            diagnostic["path_length_ratio"] = float(ratio)
                            diagnostic["path_max_deviation_m"] = float(deviation)
                            terminal_plan_diagnostics.append(diagnostic)
                            if ratio <= 1.5 and deviation <= 0.01:
                                result_success[candidate_index] = True
                                terminal_paths[candidate_index] = single_path
                                path_available[candidate_index] = True
                    elif np.any(result_success):
                        candidate_paths = extract_result_paths(result)
                        if len(candidate_paths) != len(pre_success):
                            self.debug_log(
                                "terminal path-count mismatch "
                                f"expected={len(pre_success)} got={len(candidate_paths)}; "
                                "treating all terminal candidates as failed"
                            )
                            result_success = np.zeros_like(pre_success)
                        else:
                            terminal_paths = [
                                path if pre_success[index] else None
                                for index, path in enumerate(candidate_paths)
                            ]
                            path_available[:] = [path is not None for path in terminal_paths]
                    # Path-shape validation is deliberately kept in a local
                    # mask.  Native result tensors may be CUDA-graph-backed;
                    # mutating ``result.success`` here made stale candidate
                    # state observable to later calls.
                    if result_success.shape != pre_success.shape:
                        result_success = np.zeros_like(pre_success)
                    for candidate_index, path in enumerate(terminal_paths):
                        if not (
                            pre_success[candidate_index]
                            and result_success[candidate_index]
                            and path is not None
                        ):
                            continue
                        if candidate_index not in path_metrics:
                            ratio, deviation = self.controller.measure_cartesian_path(
                                path,
                                pre_positions[candidate_index],
                                positions[candidate_index],
                            )
                            path_metrics[candidate_index] = (ratio, deviation)
                        ratio, deviation = path_metrics[candidate_index]
                        if ratio > 1.5 or deviation > 0.01:
                            result_success[candidate_index] = False
                    result_success &= path_available
                    grasp_success_count = int(np.count_nonzero(result_success))
                    both = pre_success & result_success & path_available
                    terminal_paths = [
                        path if both[index] else None
                        for index, path in enumerate(terminal_paths)
                    ]
                    joint_success_count = int(np.count_nonzero(both))

                    if postgrasp_validator is not None and joint_success_count:
                        post_valid = np.zeros_like(both, dtype=bool)
                        for candidate_index in np.flatnonzero(both):
                            terminal_path = terminal_paths[int(candidate_index)]
                            if _validate_postgrasp(candidate_index, terminal_path):
                                post_valid[int(candidate_index)] = True

                        both = post_valid
                        result_success = np.logical_and(result_success, post_valid)
                        terminal_paths = [
                            path if post_valid[index] else None
                            for index, path in enumerate(terminal_paths)
                        ]
                        grasp_success_count = int(np.count_nonzero(result_success))
                        joint_success_count = int(np.count_nonzero(both))

                    if joint_success_count:
                        if single_fallback:
                            proposed_index = int(np.flatnonzero(both)[0])
                        elif candidate_selector is not None:
                            valid_indices = np.flatnonzero(both).astype(int)
                            try:
                                proposed_index = int(
                                    candidate_selector(
                                        pre_result,
                                        result,
                                        valid_indices,
                                        positions,
                                        orientations,
                                        grasps,
                                    )
                                )
                                self.debug_log(
                                    "candidate selector proposed "
                                    f"index={proposed_index} valid={valid_indices.tolist()}"
                                )
                            except Exception as exc:  # Selection must not mask a valid plan.
                                self.debug_log(
                                    "candidate selector failed; using native priority "
                                    f"fallback error={exc!r}"
                                )
                                proposed_index = int(
                                    select_index_by_priority_dual(pre_result, result)
                                )
                        else:
                            proposed_index = int(select_index_by_priority_dual(pre_result, result))
                        if 0 <= proposed_index < len(both) and both[proposed_index]:
                            selected_index = proposed_index
                        else:
                            # A selector must never turn a malformed or stale
                            # native result into an executable candidate.
                            selected_index = int(np.flatnonzero(both)[0])
        else:
            pre_results = []
            pre_paths: list[Any | None] = []
            pregrasp_paths = pre_paths
            terminal_paths = [None] * len(grasps)
            for pre_position, pre_orientation in zip(pre_positions, pre_orientations):
                if test_mode == "forward":
                    pre_result = self.controller.test_single_forward_result(
                        pre_position, pre_orientation
                    )
                    pregrasp_plan_diagnostics.append(
                        self._result_diagnostic(pre_result, len(pregrasp_plan_diagnostics))
                    )
                    pre_ok = bool(self._success_mask(pre_result).any())
                    pre_paths_for_result = extract_result_paths(pre_result)
                    pre_path = pre_paths_for_result[0] if pre_ok and pre_paths_for_result else None
                elif test_mode == "ik":
                    pre_ok = bool(self.controller.test_single_ik(pre_position, pre_orientation))
                    pre_path = None
                else:
                    raise ValueError(f"unsupported grasp test_mode: {test_mode}")
                pre_success_count += int(pre_ok)
                pre_results.append(pre_ok)
                pre_paths.append(pre_path)
            if prepare_grasp_world is not None:
                prepare_grasp_world()
            for index, (position, orientation) in enumerate(zip(positions, orientations)):
                if test_mode == "forward":
                    if pre_results[index] and pre_paths[index] is not None:
                        grasp_result = self.controller.test_single_forward_from_path(
                            position, orientation, pre_paths[index]
                        )
                        terminal_plan_diagnostics.append(
                            self._result_diagnostic(grasp_result, index)
                        )
                        grasp_ok = bool(self._success_mask(grasp_result).any())
                        terminal_paths_for_result = extract_result_paths(grasp_result)
                        terminal_path = terminal_paths_for_result[0] if grasp_ok and terminal_paths_for_result else None
                        if terminal_path is None:
                            grasp_ok = False
                        if terminal_path is not None:
                            ratio, deviation = self.controller.measure_cartesian_path(
                                terminal_path,
                                pre_positions[index],
                                positions[index],
                            )
                            path_metrics[index] = (ratio, deviation)
                            if ratio > 1.5 or deviation > 0.01:
                                grasp_ok = False
                                terminal_path = None
                        if grasp_ok and terminal_path is not None:
                            grasp_ok = _validate_postgrasp(index, terminal_path)
                            if not grasp_ok:
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
                        if terminal_path is not None:
                            terminal_paths[index] = terminal_path

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
                pregrasp_paths[selected_index]
                if selected_index is not None and selected_index < len(pregrasp_paths)
                else None
            ),
            terminal_path=(
                terminal_paths[selected_index]
                if selected_index is not None and selected_index < len(terminal_paths)
                else None
            ),
            terminal_paths=terminal_paths,
            terminal_path_length_ratio=(
                path_metrics[selected_index][0] if selected_index in path_metrics else None
            ),
            terminal_path_max_deviation_m=(
                path_metrics[selected_index][1] if selected_index in path_metrics else None
            ),
            pregrasp_plan_diagnostics=pregrasp_plan_diagnostics,
            terminal_plan_diagnostics=terminal_plan_diagnostics,
            post_grasp_validation=post_grasp_validation,
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
