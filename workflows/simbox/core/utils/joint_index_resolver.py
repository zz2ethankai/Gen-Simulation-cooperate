"""Resolve Isaac articulation DOF indices from stable joint-name contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


JOINT_GROUP_FIELDS = {
    "left_joint": ("left_joint_indices", "left_joint_names"),
    "right_joint": ("right_joint_indices", "right_joint_names"),
    "left_gripper": ("left_gripper_indices", "left_gripper_names"),
    "right_gripper": ("right_gripper_indices", "right_gripper_names"),
    "body": ("body_indices", "body_names"),
    "head": ("head_indices", "head_names"),
    "lift": ("lift_indices", "lift_names"),
}


class JointIndexResolutionError(ValueError):
    """Raised when a configured joint contract cannot be mapped unambiguously."""


def _as_name_list(values: object, field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise JointIndexResolutionError(f"{field} must be a sequence of joint names")
    names = [str(value) for value in values]
    if not names:
        raise JointIndexResolutionError(f"{field} must not be empty when provided")
    if any(not name for name in names):
        raise JointIndexResolutionError(f"{field} contains an empty joint name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise JointIndexResolutionError(f"{field} contains duplicate joint names: {duplicates}")
    return names


def resolve_joint_names(dof_names: Sequence[str], joint_names: Sequence[str], *, group: str) -> list[int]:
    """Resolve one ordered joint-name group against Isaac's runtime DOF order."""

    runtime_names = [str(name) for name in dof_names]
    requested_names = _as_name_list(joint_names, group)
    duplicates = sorted({name for name in runtime_names if runtime_names.count(name) > 1 and name in requested_names})
    if duplicates:
        raise JointIndexResolutionError(
            f"runtime DOF names are ambiguous for {group}: {duplicates}; all DOF names={runtime_names}"
        )
    missing = [name for name in requested_names if name not in runtime_names]
    if missing:
        raise JointIndexResolutionError(
            f"runtime DOFs are missing {group} names {missing}; all DOF names={runtime_names}"
        )
    return [runtime_names.index(name) for name in requested_names]


def resolve_configured_joint_groups(dof_names: Sequence[str], config: Mapping[str, object]) -> dict[str, list[int]]:
    """Resolve named groups and validate legacy numeric groups without trusting their order."""

    runtime_names = [str(name) for name in dof_names]
    if not runtime_names:
        raise JointIndexResolutionError("Isaac returned an empty DOF-name list")

    resolved: dict[str, list[int]] = {}
    owners: dict[int, str] = {}
    for group, (indices_field, names_field) in JOINT_GROUP_FIELDS.items():
        if names_field in config:
            indices = resolve_joint_names(runtime_names, config[names_field], group=names_field)
        else:
            raw_indices = config.get(indices_field, [])
            if isinstance(raw_indices, (str, bytes)) or not isinstance(raw_indices, Sequence):
                raise JointIndexResolutionError(f"{indices_field} must be a sequence of integer indices")
            indices = [int(index) for index in raw_indices]
            invalid = [index for index in indices if index < 0 or index >= len(runtime_names)]
            if invalid:
                raise JointIndexResolutionError(
                    f"{indices_field} contains out-of-range indices {invalid} for {len(runtime_names)} runtime DOFs"
                )
            if len(indices) != len(set(indices)):
                raise JointIndexResolutionError(f"{indices_field} contains duplicate indices: {indices}")

        for index in indices:
            previous_owner = owners.get(index)
            if previous_owner is not None:
                raise JointIndexResolutionError(
                    f"runtime DOF {index} ({runtime_names[index]}) is assigned to both {previous_owner} and {group}"
                )
            owners[index] = group
        resolved[group] = indices
    return resolved
