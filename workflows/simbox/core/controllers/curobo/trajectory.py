from __future__ import annotations
from typing import Any
import numpy as np
from core.planning.domain_types import JointTrajectory
def normalize_named_trajectory(
    positions: Any,
    joint_names: Any,
    tensor_args: Any,
    *,
    context: str = "trajectory endpoint",
):
    if joint_names is None or isinstance(joint_names, (str, bytes)):
        raise ValueError(f"{context} requires explicit joint_names")
    names = list(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{context} joint_names must be non-empty and unique")
    if positions is None:
        raise ValueError(f"{context} requires position values")
    position = tensor_args.to_device(positions)
    try:
        shape = tuple(int(value) for value in position.shape)
    except AttributeError:
        shape = tuple(int(value) for value in np.asarray(position).shape)
    if len(shape) < 2 or len(shape) > 4:
        raise ValueError(
            f"{context} position must have shape [time, dof] with at most "
            f"leading singleton batch/seed dimensions, got shape={shape}"
        )
    leading_shape = shape[:-2]
    if any(size != 1 for size in leading_shape):
        raise ValueError(
            f"{context} position has non-singleton leading dimensions; "
            f"select a batch/seed candidate before conversion, got shape={shape}"
        )
    for _ in leading_shape:
        position = position[0]
    try:
        normalized_shape = tuple(int(value) for value in position.shape)
    except AttributeError:
        normalized_shape = tuple(int(value) for value in np.asarray(position).shape)
    if (
        len(normalized_shape) != 2
        or normalized_shape[0] == 0
        or normalized_shape[-1] != len(names)
    ):
        raise ValueError(
            f"{context} position must be non-empty with shape [time, dof] "
            f"matching joint_names: shape={normalized_shape}, "
            f"joint_names={names!r}"
        )
    return position, names
def execution_trajectory_tensor(
    trajectory: JointTrajectory,
    tensor_args: Any,
    *,
    target_joint_names: Any = None,
    context: str = "controller execution trajectory",
):
    if not isinstance(trajectory, JointTrajectory):
        raise TypeError(
            f"{context} requires JointTrajectory, got {type(trajectory).__name__}"
        )
    positions, names = normalize_named_trajectory(
        trajectory.positions,
        trajectory.joint_names,
        tensor_args,
        context=context,
    )
    if target_joint_names is None:
        return positions, tuple(names)
    target = tuple(str(name) for name in target_joint_names)
    if not target:
        raise ValueError(f"{context} target_joint_names must not be empty")
    if set(target) != set(names):
        raise ValueError(
            f"{context} joint_names do not match target order: "
            f"trajectory={names!r}, target={target!r}"
        )
    if tuple(names) != target:
        positions = positions[..., [names.index(name) for name in target]]
    return positions, target
__all__ = ["execution_trajectory_tensor", "normalize_named_trajectory"]
