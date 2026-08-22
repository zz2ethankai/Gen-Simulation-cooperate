"""Single CuRobo grasp-plan evaluation implementation used by Pick and Probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Sequence

import numpy as np

from core.controllers.pick_planning import PickPlanningQueryPort
from core.planning.domain_types import BatchPlanResult, CollisionPolicy, PlanResult
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
    """Evaluate grasp paths through the narrow typed Pick planning port."""

    def __init__(
        self,
        planning_port: PickPlanningQueryPort,
        debug_log: Callable[[str], None] | None = None,
    ):
        self.planning_port = planning_port
        self.debug_log = debug_log or (lambda _message: None)

    @property
    def arm(self) -> str:
        value = self.planning_port.lr_name
        if value not in {"left", "right"}:
            raise ValueError(f"Pick planning port has invalid arm identity: {value!r}")
        return value

    @staticmethod
    def _normalize_attach_paths(prim_paths: str | Sequence[str]) -> list[str]:
        if isinstance(prim_paths, str):
            return [prim_paths] if prim_paths else []
        return [str(path) for path in prim_paths]

    def missing_attach_prims(self, prim_paths: str | Sequence[str]) -> list[str]:
        paths = self._normalize_attach_paths(prim_paths)
        return [path for path in paths if not self.planning_port.has_native_obstacle(path)]

    def attach_prims_valid(self, prim_paths: str | Sequence[str]) -> bool:
        paths = self._normalize_attach_paths(prim_paths)
        return bool(paths) and not self.missing_attach_prims(paths)

    @staticmethod
    def _count(result: Any) -> int:
        success = GraspPlanEvaluator._success_mask(result)
        return int(np.count_nonzero(success))

    @staticmethod
    def _success_mask(result: Any) -> np.ndarray:
        """Read the normalized success mask without touching native results."""

        if isinstance(result, BatchPlanResult):
            success = result.success_mask
        elif isinstance(result, PlanResult):
            success = result.success
        else:
            raise TypeError(
                "GraspPlanEvaluator expects a normalized PlanResult or "
                "BatchPlanResult"
            )
        if success is None:
            return np.zeros(0, dtype=bool)
        if hasattr(success, "detach"):
            success = success.detach().cpu().numpy()
        success = np.asarray(success, dtype=bool)
        if success.ndim == 0:
            return success.reshape(1)
        return success.any(axis=-1) if success.ndim > 1 else success

    @staticmethod
    def _result_paths(result: PlanResult | Any) -> list[Any | None]:
        """Return paths from the normalized planner result only.

        Batch trajectories are represented by a sequence or by a named
        trajectory object whose first dimension is the candidate index.  The
        evaluator never reaches through the result wrapper to planner-native
        fields.
        """

        if isinstance(result, BatchPlanResult):
            trajectory = result.trajectories
        elif isinstance(result, PlanResult):
            trajectory = result.trajectory
        else:
            raise TypeError(
                "GraspPlanEvaluator expects a normalized PlanResult or "
                "BatchPlanResult"
            )
        if trajectory is None:
            return []
        if isinstance(trajectory, list):
            return trajectory
        if isinstance(trajectory, tuple):
            return list(trajectory)
        position = getattr(trajectory, "position", None)
        if getattr(position, "ndim", 0) >= 3:
            try:
                return [trajectory[index] for index in range(int(position.shape[0]))]
            except (IndexError, TypeError, ValueError):
                return []
        return [trajectory]

    @staticmethod
    def _result_metrics(result: Any) -> Any:
        """Return normalized per-candidate metrics when the result supplies them."""

        if isinstance(result, BatchPlanResult):
            return result.metrics
        if isinstance(result, PlanResult):
            return result.metrics
        raise TypeError(
            "GraspPlanEvaluator expects a normalized PlanResult or BatchPlanResult"
        )

    @staticmethod
    def _result_selected_index(result: Any) -> int | None:
        """Return the planner-selected candidate from a normalized batch result."""

        if isinstance(result, BatchPlanResult):
            value = result.selected_candidate_index
        elif isinstance(result, PlanResult):
            value = result.selected_candidate_index
        else:
            raise TypeError(
                "GraspPlanEvaluator expects a normalized PlanResult or "
                "BatchPlanResult"
            )
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metric_vector(
        metrics: Mapping[str, Any], keys: Sequence[str]
    ) -> np.ndarray | None:
        """Extract one plain numeric value per candidate from result metrics."""

        value = next((metrics[key] for key in keys if key in metrics), None)
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            values = np.asarray(value, dtype=float)
            if values.ndim == 0:
                values = values.reshape(1)
            elif values.ndim > 1:
                values = values[:, 0]
            return values.reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _trajectory_cost(path: Any) -> float | None:
        """Return cumulative joint movement for a normalized trajectory."""

        positions = getattr(path, "positions", None)
        if positions is None:
            positions = getattr(path, "position", None)
        if positions is None:
            return None
        try:
            values = np.asarray(positions, dtype=float)
            if values.ndim < 2 or values.shape[0] < 2:
                return 0.0
            return float(np.abs(np.diff(values, axis=0)).sum())
        except (TypeError, ValueError):
            return None

    @classmethod
    def choose_candidate_index(
        cls,
        pre_result: Any,
        result: Any,
        valid_indices: Sequence[int] | None = None,
    ) -> int:
        """Choose one candidate from normalized pre/terminal plan results.

        The planner owns batch success and selected-index semantics.  The
        evaluator only intersects the normalized masks, honors the planner's
        selected index when valid, and uses deterministic lowest-index
        fallback for malformed or incomplete diagnostics.
        """

        pre_success = cls._success_mask(pre_result)
        result_success = cls._success_mask(result)
        common_count = min(len(pre_success), len(result_success))
        common = np.logical_and(
            pre_success[:common_count], result_success[:common_count]
        )
        if valid_indices is not None:
            requested = np.asarray(valid_indices, dtype=int).reshape(-1)
            common_indices = [
                int(index)
                for index in requested
                if 0 <= int(index) < len(common) and bool(common[int(index)])
            ]
        else:
            common_indices = [int(index) for index in np.flatnonzero(common)]
        if not common_indices:
            return 0

        selected = cls._result_selected_index(result)
        if selected in common_indices:
            return int(selected)
        selected = cls._result_selected_index(pre_result)
        if selected in common_indices:
            return int(selected)

        metrics = cls._result_metrics(result)
        if not isinstance(metrics, Mapping):
            metrics = {}

        # Preserve the old error-filter behavior using only normalized
        # per-candidate metrics.  If a metric is present but filters every
        # common candidate, retain the complete common set as a safe fallback.
        filtered_indices = list(common_indices)
        for keys in (
            ("position_error", "position_errors", "goal_position_error", "pose_error"),
            ("rotation_error", "rotation_errors", "orientation_error", "orientation_errors"),
        ):
            values = cls._metric_vector(metrics, keys)
            if values is None:
                continue
            eligible = [
                index
                for index in filtered_indices
                if index < len(values) and np.isfinite(values[index])
            ]
            if not eligible:
                continue
            threshold = float(np.mean([values[index] for index in eligible]))
            within_threshold = [
                index for index in eligible if values[index] <= threshold
            ]
            if within_threshold:
                filtered_indices = within_threshold

        # If the normalized result publishes a numeric priority/cost metric,
        # use it without interpreting native solver fields.  Lower is
        # preferred; unavailable/non-finite values fall back to trajectory
        # movement and then candidate order.
        common_indices = filtered_indices or common_indices
        metric_values = None
        for key in (
            "priority",
            "path_cost",
            "path_costs",
            "trajectory_cost",
            "cost",
            "costs",
            "joint_difference",
            "score",
            "scores",
        ):
            if key in metrics:
                metric_values = metrics[key]
                break
        if metric_values is not None:
            try:
                values = np.asarray(metric_values, dtype=float)
                if values.ndim > 1:
                    values = values[:, 0]
                values = values.reshape(-1)
                ranked = [
                    index
                    for index in common_indices
                    if index < len(values) and np.isfinite(values[index])
                ]
                if ranked:
                    return min(ranked, key=lambda index: (float(values[index]), index))
            except (TypeError, ValueError):
                pass

        paths = cls._result_paths(result)
        path_costs = {
            index: cls._trajectory_cost(paths[index])
            for index in common_indices
            if index < len(paths) and paths[index] is not None
        }
        ranked_paths = [
            index
            for index, cost in path_costs.items()
            if cost is not None and np.isfinite(cost)
        ]
        if ranked_paths:
            return min(ranked_paths, key=lambda index: (path_costs[index], index))
        return min(common_indices)

    @staticmethod
    def _planner_diagnostic(result: Any) -> str:
        """Return the typed planner status without exposing native details."""

        if not isinstance(result, (PlanResult, BatchPlanResult)):
            raise TypeError(
                "GraspPlanEvaluator expects a normalized PlanResult or "
                "BatchPlanResult"
            )
        status = result.status
        status = getattr(status, "value", status)
        return "status=%s reason=%s success=%d" % (
            status,
            result.reason,
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
        """Capture typed planner status for a candidate planning attempt."""

        if not isinstance(result, (PlanResult, BatchPlanResult)):
            raise TypeError(
                "GraspPlanEvaluator expects a normalized PlanResult or "
                "BatchPlanResult"
            )
        status = result.status
        status = getattr(status, "value", status)
        if status is not None:
            status = str(status)
        success = cls._success_mask(result)
        paths = cls._result_paths(result)
        metrics = cls._result_metrics(result)
        trajectory_info = {
            "count": len(paths),
            "available_count": sum(path is not None for path in paths),
        }
        metadata = result.metadata
        return {
            "candidate_index": int(candidate_index),
            "success": cls._tensor_summary(success),
            "status": status,
            "reason": result.reason,
            "world_revision": result.world_revision,
            "candidate_indices": list(result.candidate_indices or ()),
            "selected_index": cls._result_selected_index(result),
            "metrics": cls._tensor_summary(metrics),
            "trajectory": trajectory_info,
            "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        }

    def evaluate(
        self,
        grasp_transforms: np.ndarray,
        grasp_scores: np.ndarray,
        pregrasp_offset_m: float,
        attach_prim_paths: str | Sequence[str],
        fixed_orientation: np.ndarray | None = None,
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
        if "r5a" in str(self.planning_port.robot_file):
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
        if self.planning_port.batch_capability:
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
                result = self.planning_port.plan_pose_batch(
                    positions,
                    orientations,
                    collision_policy=CollisionPolicy.TARGET_APPROACH,
                )
                self.debug_log("grasp-only " + self._planner_diagnostic(result))
                grasp_success_count = self._count(result)
                pre_success_count = grasp_success_count
                joint_success_count = grasp_success_count
                if joint_success_count:
                    selected_index = self._result_selected_index(result)
                    result_mask = self._success_mask(result)
                    if (
                        selected_index is None
                        or not 0 <= selected_index < len(result_mask)
                        or not bool(result_mask[selected_index])
                    ):
                        selected_index = self.choose_candidate_index(result, result)
            else:
                pre_result = self.planning_port.plan_pose_batch(
                    pre_positions,
                    pre_orientations,
                    collision_policy=CollisionPolicy.WORLD_TRANSIT,
                )
                self.debug_log("pregrasp " + self._planner_diagnostic(pre_result))
                pregrasp_plan_diagnostics.append(self._result_diagnostic(pre_result, -1))
                pre_success_count = self._count(pre_result)
                pre_paths = self._result_paths(pre_result)
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
                # A typed batch result may carry no trajectory when every
                # candidate fails.
                # Treat this as the expected no-path outcome and
                # never enter the terminal-world planning pass.
                result = None
                if pre_success_count and any(path is not None for path in pre_paths):
                    if prepare_grasp_world is not None:
                        prepare_grasp_world()
                    result = self.planning_port.plan_pose_batch(
                        positions,
                        orientations,
                        collision_policy=CollisionPolicy.TARGET_APPROACH,
                        start_paths=pre_paths,
                    )
                    self.debug_log("terminal-grasp " + self._planner_diagnostic(result))
                    batch_diagnostic = self._result_diagnostic(result, -1)
                    batch_diagnostic["mode"] = "native_batch"
                    terminal_plan_diagnostics.append(batch_diagnostic)
                    result_success = self._success_mask(result)
                    if result_success.shape != pre_success.shape:
                        self.debug_log(
                            "terminal-grasp success-mask mismatch "
                            f"pre={pre_success.shape} final={result_success.shape}; "
                            "treating terminal candidates as failed"
                        )
                        result_success = np.zeros_like(pre_success)
                    # A complete batch terminal miss is not evidence that
                    # every candidate is unreachable.  The single typed
                    # planner uses its normal seed budget and can still solve
                    # a candidate from the same named pre-grasp endpoint.
                    # Retry only this exceptional all-failed case so the
                    # normal batch path remains unchanged.
                    single_fallback = not np.any(result_success)
                    if single_fallback:
                        self.debug_log(
                            "terminal-grasp batch returned no valid candidate; "
                            "retrying eligible candidates with typed single planner"
                        )
                    path_available = np.zeros_like(pre_success)
                    terminal_paths = [None] * len(pre_success)
                    if single_fallback:
                        single_from_path = getattr(
                            self.planning_port, "plan_pose_from_path", None
                        )
                        if not callable(single_from_path):
                            self.debug_log(
                                "terminal-grasp typed single fallback unavailable; "
                                "planning port has no path query"
                            )
                        else:
                            for candidate_index, (pre_ok, pre_path) in enumerate(
                                zip(pre_success, pre_paths)
                            ):
                                if not pre_ok or pre_path is None:
                                    continue
                                try:
                                    single_result = single_from_path(
                                        positions[candidate_index],
                                        orientations[candidate_index],
                                        pre_path,
                                        collision_policy=CollisionPolicy.TARGET_APPROACH,
                                    )
                                except Exception as exc:  # Evidence only; keep trying candidates.
                                    diagnostic = {
                                        "candidate_index": int(candidate_index),
                                        "mode": "native_single_fallback",
                                        "success": False,
                                        "path_available": False,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                    terminal_plan_diagnostics.append(diagnostic)
                                    continue

                                diagnostic = self._result_diagnostic(
                                    single_result, candidate_index
                                )
                                diagnostic["mode"] = "native_single_fallback"
                                single_paths = self._result_paths(single_result)
                                single_path = (
                                    single_paths[0]
                                    if self._count(single_result) and single_paths
                                    else None
                                )
                                if single_path is None:
                                    diagnostic["path_available"] = False
                                    terminal_plan_diagnostics.append(diagnostic)
                                    continue

                                try:
                                    ratio, deviation = (
                                        self.planning_port.measure_cartesian_path(
                                            single_path,
                                            pre_positions[candidate_index],
                                            positions[candidate_index],
                                        )
                                    )
                                except Exception as exc:  # Evidence only; keep trying candidates.
                                    diagnostic["path_available"] = False
                                    diagnostic[
                                        "error"
                                    ] = f"{type(exc).__name__}: {exc}"
                                    terminal_plan_diagnostics.append(diagnostic)
                                    continue

                                ratio = float(ratio)
                                deviation = float(deviation)
                                path_metrics[candidate_index] = (ratio, deviation)
                                diagnostic["path_available"] = True
                                diagnostic["path_length_ratio"] = ratio
                                diagnostic["path_max_deviation_m"] = deviation
                                terminal_plan_diagnostics.append(diagnostic)
                                if ratio <= 1.5 and deviation <= 0.01:
                                    result_success[candidate_index] = True
                                    terminal_paths[candidate_index] = single_path
                                    path_available[candidate_index] = True
                    elif np.any(result_success):
                        candidate_paths = self._result_paths(result)
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
                            ratio, deviation = self.planning_port.measure_cartesian_path(
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
                            # The batch result has no successful candidate to
                            # rank.  Select deterministically from the valid
                            # fallback paths instead of asking the normalized
                            # batch result to rank an empty intersection.
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
                                proposed_index = self.choose_candidate_index(
                                    pre_result, result, valid_indices
                                )
                        else:
                            proposed_index = self.choose_candidate_index(pre_result, result)
                        if 0 <= proposed_index < len(both) and both[proposed_index]:
                            selected_index = proposed_index
                        else:
                            # A selector must never turn a malformed or stale
                            # planner result into an executable candidate.
                            selected_index = int(np.flatnonzero(both)[0])
        else:
            pre_results = []
            pre_paths: list[Any | None] = []
            pregrasp_paths = pre_paths
            terminal_paths = [None] * len(grasps)
            for pre_position, pre_orientation in zip(pre_positions, pre_orientations):
                pre_result = self.planning_port.plan_pose_result(
                    pre_position,
                    pre_orientation,
                    collision_policy=CollisionPolicy.WORLD_TRANSIT,
                )
                pregrasp_plan_diagnostics.append(
                    self._result_diagnostic(pre_result, len(pregrasp_plan_diagnostics))
                )
                pre_ok = bool(self._success_mask(pre_result).any())
                pre_paths_for_result = self._result_paths(pre_result)
                pre_path = pre_paths_for_result[0] if pre_ok and pre_paths_for_result else None
                pre_success_count += int(pre_ok)
                pre_results.append(pre_ok)
                pre_paths.append(pre_path)
            if prepare_grasp_world is not None:
                prepare_grasp_world()
            for index, (position, orientation) in enumerate(zip(positions, orientations)):
                if pre_results[index] and pre_paths[index] is not None:
                    grasp_result = self.planning_port.plan_pose_from_path(
                        position,
                        orientation,
                        pre_paths[index],
                        collision_policy=CollisionPolicy.TARGET_APPROACH,
                    )
                    terminal_plan_diagnostics.append(
                        self._result_diagnostic(grasp_result, index)
                    )
                    grasp_ok = bool(self._success_mask(grasp_result).any())
                    terminal_paths_for_result = self._result_paths(grasp_result)
                    terminal_path = terminal_paths_for_result[0] if grasp_ok and terminal_paths_for_result else None
                    if terminal_path is None:
                        grasp_ok = False
                    if terminal_path is not None:
                        ratio, deviation = self.planning_port.measure_cartesian_path(
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
