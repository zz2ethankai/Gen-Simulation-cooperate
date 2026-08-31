"""Pure trajectory helpers retained for offline callers.

Planner result normalization belongs to :mod:`core.planning.domain_types`.
This module intentionally does not inspect native planner result objects or
select candidates from them.
"""

from __future__ import annotations

import torch

from core.controllers.curobo.trajectory import normalize_named_trajectory


class _TorchTensorArgs:
    """Minimal tensor boundary for offline helpers without a DeviceCfg."""

    @staticmethod
    def to_device(value):
        return torch.as_tensor(value)


def _trajectory_positions(path, tensor_args=None):
    if tensor_args is None:
        tensor_args = _TorchTensorArgs()
    return normalize_named_trajectory(
        path.positions,
        path.joint_names,
        tensor_args,
        context="plan utility trajectory",
    )[0]


def sort_by_difference_js(paths, weights=None, *, tensor_args=None):
    """Return path indices ordered by cumulative joint-space movement."""

    if not paths:
        return torch.empty(0, dtype=torch.long)
    position = _trajectory_positions(paths[0], tensor_args)
    if weights is None:
        weights = torch.ones(
            position.shape[-1], device=position.device, dtype=position.dtype
        )
    else:
        weights = torch.as_tensor(weights, device=position.device, dtype=position.dtype)
    if weights.shape[0] != position.shape[-1]:
        raise ValueError("weights must match the trajectory joint dimension")
    costs = []
    for path in paths:
        path_position = _trajectory_positions(path, tensor_args)
        delta = torch.abs(path_position[1:, :] - path_position[:-1, :])
        costs.append((delta.sum(dim=0) * weights).sum())
    return torch.argsort(torch.stack(costs))


def filter_paths_by_position_error(paths, position_errors):
    """Return mean-threshold mask for already normalized position metrics."""

    if len(paths) != position_errors.shape[0]:
        raise ValueError("paths and position_errors must have equal length")
    threshold = torch.mean(position_errors)
    return [error <= threshold for error in position_errors]


def filter_paths_by_rotation_error(paths, rotation_errors):
    """Return mean-threshold mask for already normalized rotation metrics."""

    if len(paths) != rotation_errors.shape[0]:
        raise ValueError("paths and rotation_errors must have equal length")
    threshold = torch.mean(rotation_errors)
    return [error <= threshold for error in rotation_errors]
