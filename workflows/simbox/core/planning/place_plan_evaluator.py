"""Chained CuRobo validation for pre-place transit and terminal descent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)


@dataclass(frozen=True)
class PlacePlanEvaluation:
    feasible: bool
    selected_index: int | None
    preplace_success_count: int
    descent_success_count: int
    joint_success_count: int
    failure_code: str | None


def _success_count(result: Any) -> int:
    success = getattr(result, "success", None)
    return int(success.sum().item()) if success is not None else 0


def evaluate_place_paths(
    controller: Any,
    collision_scene_manager: Any,
    object_name: str,
    support_name: str,
    preplace_positions: np.ndarray,
    preplace_orientations: np.ndarray,
    descent_positions: np.ndarray,
    descent_orientations: np.ndarray,
    *,
    test_mode: str,
) -> PlacePlanEvaluation:
    """Require a common candidate through the transit and contact planning worlds."""

    same_target = np.array_equal(preplace_positions, descent_positions) and np.array_equal(
        preplace_orientations, descent_orientations
    )
    if controller.use_batch:
        pre_result = controller.test_batch_forward(
            preplace_positions, preplace_orientations
        )
        pre_count = _success_count(pre_result)
        if not pre_count:
            return PlacePlanEvaluation(
                False, None, 0, 0, 0, "NO_COLLISION_FREE_PREPLACE_PLAN"
            )
        if same_target:
            index = int(select_index_by_priority_single(pre_result))
            return PlacePlanEvaluation(True, index, pre_count, pre_count, pre_count, None)
        if not hasattr(controller, "test_batch_forward_from_paths"):
            raise RuntimeError(
                "physics-schema Place controller lacks chained-start batch planning"
            )
        with collision_scene_manager.placement_descent_planning_world(
            object_name, support_name, controller.name, controller.lr_name
        ):
            descent_result = controller.test_batch_forward_from_paths(
                descent_positions,
                descent_orientations,
                pre_result.get_paths(),
            )
        descent_count = _success_count(descent_result)
        joint_count = int((pre_result.success & descent_result.success).sum().item())
        if not joint_count:
            return PlacePlanEvaluation(
                False,
                None,
                pre_count,
                descent_count,
                0,
                "NO_COLLISION_FREE_PLACE_DESCENT_PLAN",
            )
        index = int(select_index_by_priority_dual(pre_result, descent_result))
        return PlacePlanEvaluation(
            True, index, pre_count, descent_count, joint_count, None
        )

    if test_mode != "forward":
        raise RuntimeError(
            "physics-schema Place chained planning requires test_mode=forward"
        )
    pre_results: list[Any] = []
    pre_count = 0
    for position, orientation in zip(preplace_positions, preplace_orientations):
        result = controller.test_single_forward_result(position, orientation)
        pre_results.append(result)
        pre_count += int(bool(result.success.item()))
    if not pre_count:
        return PlacePlanEvaluation(
            False, None, 0, 0, 0, "NO_COLLISION_FREE_PREPLACE_PLAN"
        )
    if same_target:
        index = next(
            index
            for index, result in enumerate(pre_results)
            if bool(result.success.item())
        )
        return PlacePlanEvaluation(True, index, pre_count, pre_count, pre_count, None)

    descent_count = 0
    joint_count = 0
    selected_index = None
    with collision_scene_manager.placement_descent_planning_world(
        object_name, support_name, controller.name, controller.lr_name
    ):
        for index, (position, orientation, pre_result) in enumerate(
            zip(descent_positions, descent_orientations, pre_results)
        ):
            if not bool(pre_result.success.item()):
                continue
            descent_result = controller.test_single_forward_from_path(
                position,
                orientation,
                pre_result.get_interpolated_plan(),
            )
            descent_ok = bool(descent_result.success.item())
            descent_count += int(descent_ok)
            joint_count += int(descent_ok)
            if descent_ok and selected_index is None:
                selected_index = index
    if selected_index is None:
        return PlacePlanEvaluation(
            False,
            None,
            pre_count,
            descent_count,
            joint_count,
            "NO_COLLISION_FREE_PLACE_DESCENT_PLAN",
        )
    return PlacePlanEvaluation(
        True, selected_index, pre_count, descent_count, joint_count, None
    )
